"""Module 3 feature layer: match state at a checkpoint, and the labels to predict.

Everything here obeys one rule, and the rule is the whole reason this module is a separate
file with its own test:

    **A feature at checkpoint t must be computable from actions with minute <= t, and
    nothing else.**

Full-match aggregates leak the future. So do full-match *network* metrics -- which is why
B2's structural features are built from windows that have already closed rather than from
`centrality_teams.parquet`. A leak here would not raise an error; it would silently make
every Module 3 number look excellent and mean nothing. `tests/test_match_state.py` enforces
the rule by truncating the action stream and asserting the feature row is unchanged.

Labels come from the match index, which needs one repair first: the Wyscout half of
`games.parquet` has null scores (the loader does not expose them), recoverable in full from
the raw match JSON.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tacticalgraph.config import Paths
from tacticalgraph.data.download import load_json
from tacticalgraph.data.recipient import PASS_LIKE_TYPES
from tacticalgraph.data.schema import PROVIDER_COMPARABLE_TYPES
from tacticalgraph.graphs.passing_network import window_bounds

log = logging.getLogger(__name__)

# Outcome encoding, always from the HOME team's perspective.
OUTCOME_LABELS: tuple[str, ...] = ("home_win", "draw", "away_win")
OUTCOME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(OUTCOME_LABELS)}

MATCH_MINUTES = 90.0

# Feature groups, in ladder order. Each rung is a superset of the previous one, so the
# comparison isolates exactly what the added features buy.
B0_FEATURES: tuple[str, ...] = (
    "goal_diff",
    "minutes_remaining",
    "goal_diff_x_minutes_remaining",
)

B1_FEATURES: tuple[str, ...] = B0_FEATURES + (
    "shots_home",
    "shots_away",
    "passes_home",
    "passes_away",
    "pass_completion_home",
    "pass_completion_away",
    "possession_share_home",
    "xt_home",
    "xt_away",
    "xt_diff",
)

B2_FEATURES: tuple[str, ...] = B1_FEATURES + (
    "mean_x_home",
    "mean_x_away",
    "form_ppg_home",
    "form_ppg_away",
    "form_gd_home",
    "form_gd_away",
    "network_density_home",
    "network_density_away",
    "network_centralization_home",
    "network_centralization_away",
)

FEATURE_LADDER: dict[str, tuple[str, ...]] = {
    "B0": B0_FEATURES,
    "B1": B1_FEATURES,
    "B2": B2_FEATURES,
}


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------


def wyscout_scores(paths: Paths) -> pd.DataFrame:
    """Read final scores for Serie A 2017/18 from the raw Wyscout match file.

    `PublicWyscoutLoader.games()` does not expose scores, so `games.parquet` has them null
    for all 380 Wyscout rows. They are present in `matches_Italy.json` under
    `teamsData[*].score`, keyed by side.
    """
    matches = load_json(paths.raw_wyscout / "matches_Italy.json")
    rows = []
    for match in matches:
        sides = {team["side"]: team for team in match["teamsData"].values()}
        if not {"home", "away"} <= sides.keys():
            log.warning("match %s has no home/away sides; skipped", match.get("wyId"))
            continue
        rows.append(
            {
                "game_id": int(match["wyId"]),
                "home_score_fill": int(sides["home"]["score"]),
                "away_score_fill": int(sides["away"]["score"]),
                "home_team_id_check": int(sides["home"]["teamId"]),
                "away_team_id_check": int(sides["away"]["teamId"]),
            }
        )
    return pd.DataFrame(rows)


def backfill_wyscout_scores(paths: Paths, games: pd.DataFrame) -> pd.DataFrame:
    """Fill the null Wyscout scores, verifying home/away orientation while doing it.

    The orientation check matters: filling `home_score` from the wrong side would invert
    the label for half the test season and be nearly impossible to spot downstream.
    """
    games = games.copy()
    fill = wyscout_scores(paths)
    merged = games.merge(fill, on="game_id", how="left")

    is_wyscout = merged["provider"] == "wyscout"
    matched = is_wyscout & merged["home_score_fill"].notna()

    orientation_ok = (
        merged.loc[matched, "home_team_id"].astype(int)
        == merged.loc[matched, "home_team_id_check"].astype(int)
    )
    if not bool(orientation_ok.all()):
        wrong = int((~orientation_ok).sum())
        raise ValueError(
            f"{wrong} Wyscout match(es) disagree on which team is home between the loader "
            "and the raw match file; refusing to backfill scores that may be inverted."
        )

    merged.loc[matched, "home_score"] = merged.loc[matched, "home_score_fill"].to_numpy()
    merged.loc[matched, "away_score"] = merged.loc[matched, "away_score_fill"].to_numpy()

    still_missing = int(merged["home_score"].isna().sum())
    log.info(
        "backfilled %d Wyscout scores (%d/%d Wyscout rows); %d games still without a score",
        int(matched.sum()),
        int(matched.sum()),
        int(is_wyscout.sum()),
        still_missing,
    )
    return merged.drop(
        columns=[
            "home_score_fill",
            "away_score_fill",
            "home_team_id_check",
            "away_team_id_check",
        ]
    )


def match_outcomes(games: pd.DataFrame) -> pd.DataFrame:
    """Per-game 3-class label from the home team's perspective."""
    frame = games[games["home_score"].notna() & games["away_score"].notna()].copy()
    difference = frame["home_score"].astype(int) - frame["away_score"].astype(int)
    frame["outcome"] = np.where(difference > 0, "home_win", np.where(difference < 0, "away_win", "draw"))
    frame["outcome_index"] = frame["outcome"].map(OUTCOME_TO_INDEX).astype("int64")
    return frame[
        ["game_id", "season", "provider", "game_day", "home_team_id", "away_team_id",
         "home_score", "away_score", "outcome", "outcome_index"]
    ].reset_index(drop=True)


def action_minutes(actions: pd.DataFrame) -> pd.Series:
    """Match minute of each action, offsetting the per-period clock reset."""
    period_offset = {1: 0.0, 2: 45.0, 3: 90.0, 4: 105.0, 5: 120.0}
    offsets = actions["period_id"].map(period_offset).fillna(0.0)
    return offsets + actions["time_seconds"] / 60.0


def derive_goals(actions: pd.DataFrame) -> pd.DataFrame:
    """Goal events derived from SPADL, with own goals credited to the opposing team.

    Verified against the StatsBomb match index: this rule reproduces the final score
    exactly for 80/80 games sampled, own goals included.
    """
    minutes = action_minutes(actions)
    frame = actions.assign(_minute=minutes)

    scored = frame[
        frame["type_name"].str.startswith("shot") & (frame["result_name"] == "success")
    ][["game_id", "team_id", "_minute"]].copy()
    scored["scoring_team_id"] = scored["team_id"].astype("int64")

    own = frame[frame["result_name"] == "owngoal"][["game_id", "team_id", "_minute"]].copy()
    # An own goal is credited to the *other* team, so the scorer is resolved per game from
    # the pair of teams involved.
    teams_per_game = frame.groupby("game_id")["team_id"].unique().to_dict()
    own["scoring_team_id"] = [
        int(next(t for t in teams_per_game[game] if int(t) != int(team)))
        for game, team in zip(own["game_id"], own["team_id"])
    ]

    goals = pd.concat([scored, own], ignore_index=True)
    goals = goals.rename(columns={"_minute": "minute", "team_id": "acting_team_id"})
    return goals.sort_values(["game_id", "minute"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------


def checkpoints() -> list[float]:
    """The evaluation grid: the closing minute of each 15-minute window.

    Deliberately identical to `window_bounds()` ends so the tabular ladder and the graph
    model are scored on the same support -- a five-minute grid would give the first two
    checkpoints no complete window.
    """
    return [end for _, end in window_bounds()]


# --------------------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------------------


def _rolling_form(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Points and goal difference per game from strictly earlier matchweeks.

    Computed within a season only: the two seasons have different squads, and carrying
    2015/16 form into the 2017/18 test set would leak across the split.
    """
    rows = []
    for (season, provider), season_games in outcomes.groupby(["season", "provider"]):
        for side in ("home", "away"):
            frame = season_games.rename(columns={f"{side}_team_id": "team_id"})
            frame = frame[["game_id", "season", "provider", "game_day", "team_id",
                           "home_score", "away_score"]].copy()
            frame["is_home"] = side == "home"
            frame["goals_for"] = np.where(frame["is_home"], frame["home_score"], frame["away_score"])
            frame["goals_against"] = np.where(frame["is_home"], frame["away_score"], frame["home_score"])
            rows.append(frame)

    long = pd.concat(rows, ignore_index=True)
    long["points"] = np.where(
        long["goals_for"] > long["goals_against"], 3, np.where(long["goals_for"] == long["goals_against"], 1, 0)
    )
    long["goal_difference"] = long["goals_for"] - long["goals_against"]
    long = long.sort_values(["season", "provider", "team_id", "game_day"])

    grouped = long.groupby(["season", "provider", "team_id"], sort=False)
    # shift(1) so a match never sees its own result; expanding mean over prior matches only.
    long["form_ppg"] = grouped["points"].transform(lambda s: s.shift(1).expanding().mean())
    long["form_gd"] = grouped["goal_difference"].transform(lambda s: s.shift(1).expanding().mean())
    # Matchweek 1 has no history: 1.0 point/game is the neutral prior for a 3-1-0 system.
    long["form_ppg"] = long["form_ppg"].fillna(1.0)
    long["form_gd"] = long["form_gd"].fillna(0.0)

    return long[["game_id", "team_id", "is_home", "form_ppg", "form_gd"]]


def _cumulative_team_stats(
    actions: pd.DataFrame, checkpoint: float, minutes: pd.Series
) -> pd.DataFrame:
    """Per-(game, team) aggregates over actions strictly up to `checkpoint`."""
    upto = actions[minutes.to_numpy() <= checkpoint]
    if upto.empty:
        return pd.DataFrame(
            columns=["game_id", "team_id", "shots", "passes", "pass_completion",
                     "comparable_actions", "mean_x", "xt"]
        )

    frame = upto.copy()
    frame["_is_pass"] = frame["type_name"].isin(PASS_LIKE_TYPES)
    frame["_pass_ok"] = frame["_is_pass"] & (frame["result_name"] == "success")
    frame["_is_shot"] = frame["type_name"].str.startswith("shot")
    # Only provider-comparable types count towards volume, so possession share does not
    # inherit StatsBomb's 8.7x dribble inflation.
    frame["_comparable"] = frame["type_name"].isin(PROVIDER_COMPARABLE_TYPES)

    aggregated = frame.groupby(["game_id", "team_id"]).agg(
        shots=("_is_shot", "sum"),
        passes=("_is_pass", "sum"),
        passes_ok=("_pass_ok", "sum"),
        comparable_actions=("_comparable", "sum"),
        mean_x=("start_x", "mean"),
        xt=("xt_value", "sum") if "xt_value" in frame.columns else ("_is_pass", "size"),
    ).reset_index()

    if "xt_value" not in frame.columns:
        aggregated["xt"] = 0.0

    aggregated["pass_completion"] = (
        aggregated["passes_ok"] / aggregated["passes"].replace(0, np.nan)
    ).fillna(0.0)
    return aggregated.drop(columns=["passes_ok"])


def _cumulative_network_stats(
    window_nodes: pd.DataFrame, window_edges: pd.DataFrame, window_index: int
) -> pd.DataFrame:
    """Density and centralisation from the window that just closed.

    Uses only `window_index` (whose window ends at the checkpoint), never the full-match
    network -- that would be a direct future leak.
    """
    nodes = window_nodes[window_nodes["window_index"] == window_index]
    edges = window_edges[window_edges["window_index"] == window_index]
    if nodes.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "network_density",
                                     "network_centralization"])

    node_counts = nodes.groupby(["game_id", "team_id"]).size().rename("n_nodes")
    edge_counts = edges.groupby(["game_id", "team_id"]).size().rename("n_edges")
    degree = (
        edges.groupby(["game_id", "team_id", "source"]).size().rename("degree").reset_index()
    )
    max_degree = degree.groupby(["game_id", "team_id"])["degree"].max().rename("max_degree")
    mean_degree = degree.groupby(["game_id", "team_id"])["degree"].mean().rename("mean_degree")

    frame = pd.concat([node_counts, edge_counts, max_degree, mean_degree], axis=1).reset_index()
    frame["n_edges"] = frame["n_edges"].fillna(0.0)
    possible = frame["n_nodes"] * (frame["n_nodes"] - 1)
    frame["network_density"] = (frame["n_edges"] / possible.replace(0, np.nan)).fillna(0.0)
    frame["network_centralization"] = (
        (frame["max_degree"] - frame["mean_degree"]) / frame["max_degree"].replace(0, np.nan)
    ).fillna(0.0)
    return frame[["game_id", "team_id", "network_density", "network_centralization"]]


def build_state_table(
    actions: pd.DataFrame,
    outcomes: pd.DataFrame,
    window_nodes: pd.DataFrame | None = None,
    window_edges: pd.DataFrame | None = None,
    xt_values: pd.Series | None = None,
) -> pd.DataFrame:
    """One row per (game, checkpoint) with the full feature ladder and the label.

    `xt_values` must be aligned to `actions.index` and produced by an xThreat model fit on
    the training split only. Passing None yields zero xT columns, which keeps the function
    usable in tests without fitting a model.
    """
    frame = actions.copy()
    if xt_values is not None:
        frame["xt_value"] = pd.Series(xt_values, index=actions.index).fillna(0.0).to_numpy()
    minutes = action_minutes(frame)

    goals = derive_goals(frame)
    form = _rolling_form(outcomes)
    games = outcomes.set_index("game_id")

    rows = []
    grid = checkpoints()

    for window_index, checkpoint in enumerate(grid):
        stats = _cumulative_team_stats(frame, checkpoint, minutes)
        stats = stats.set_index(["game_id", "team_id"])

        if window_nodes is not None and window_edges is not None:
            network = _cumulative_network_stats(window_nodes, window_edges, window_index)
            network = network.set_index(["game_id", "team_id"])
        else:
            network = None

        scored = goals[goals["minute"] <= checkpoint]
        goal_counts = scored.groupby(["game_id", "scoring_team_id"]).size()

        for game_id, game in games.iterrows():
            home, away = int(game["home_team_id"]), int(game["away_team_id"])

            def _stat(team: int, column: str, default: float = 0.0) -> float:
                try:
                    return float(stats.loc[(game_id, team), column])
                except KeyError:
                    return default

            def _net(team: int, column: str) -> float:
                if network is None:
                    return 0.0
                try:
                    return float(network.loc[(game_id, team), column])
                except KeyError:
                    return 0.0

            goals_home = float(goal_counts.get((game_id, home), 0))
            goals_away = float(goal_counts.get((game_id, away), 0))
            comparable_home = _stat(home, "comparable_actions")
            comparable_away = _stat(away, "comparable_actions")
            total_comparable = comparable_home + comparable_away

            form_home = form[(form["game_id"] == game_id) & form["is_home"]]
            form_away = form[(form["game_id"] == game_id) & ~form["is_home"]]

            xt_home, xt_away = _stat(home, "xt"), _stat(away, "xt")
            minutes_remaining = MATCH_MINUTES - checkpoint
            goal_diff = goals_home - goals_away

            rows.append(
                {
                    "game_id": int(game_id),
                    "season": game["season"],
                    "provider": game["provider"],
                    "window_index": window_index,
                    "checkpoint_minute": checkpoint,
                    "outcome_index": int(game["outcome_index"]),
                    "goals_home": goals_home,
                    "goals_away": goals_away,
                    # --- B0
                    "goal_diff": goal_diff,
                    "minutes_remaining": minutes_remaining,
                    "goal_diff_x_minutes_remaining": goal_diff * minutes_remaining,
                    # --- B1
                    "shots_home": _stat(home, "shots"),
                    "shots_away": _stat(away, "shots"),
                    "passes_home": _stat(home, "passes"),
                    "passes_away": _stat(away, "passes"),
                    "pass_completion_home": _stat(home, "pass_completion"),
                    "pass_completion_away": _stat(away, "pass_completion"),
                    "possession_share_home": (
                        comparable_home / total_comparable if total_comparable else 0.5
                    ),
                    "xt_home": xt_home,
                    "xt_away": xt_away,
                    "xt_diff": xt_home - xt_away,
                    # --- B2
                    "mean_x_home": _stat(home, "mean_x", 52.5),
                    "mean_x_away": _stat(away, "mean_x", 52.5),
                    "form_ppg_home": float(form_home["form_ppg"].iloc[0]) if len(form_home) else 1.0,
                    "form_ppg_away": float(form_away["form_ppg"].iloc[0]) if len(form_away) else 1.0,
                    "form_gd_home": float(form_home["form_gd"].iloc[0]) if len(form_home) else 0.0,
                    "form_gd_away": float(form_away["form_gd"].iloc[0]) if len(form_away) else 0.0,
                    "network_density_home": _net(home, "network_density"),
                    "network_density_away": _net(away, "network_density"),
                    "network_centralization_home": _net(home, "network_centralization"),
                    "network_centralization_away": _net(away, "network_centralization"),
                }
            )

    table = pd.DataFrame(rows).sort_values(["game_id", "window_index"]).reset_index(drop=True)
    log.info(
        "state table: %d rows (%d games x %d checkpoints)",
        len(table),
        table["game_id"].nunique(),
        len(grid),
    )
    return table
