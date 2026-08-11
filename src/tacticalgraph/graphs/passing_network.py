"""Passing-network construction.

A passing network is built per (game, team): nodes are players, and a directed edge u->v
carries the passes u completed to v. Node positions are the player's mean action location,
which is what makes the networks plottable on a pitch and comparable across matches.

Two flavours, both from the same core routine:

* **full-match** -- one network per team-match. Feeds Module 2 (centrality, role embeddings).
* **windowed**   -- a 15-minute window sliding with a 5-minute stride, ~16 steps per match.
  Feeds Module 3. The window is long enough that a team's network is not mostly isolated
  nodes (~75 passes per team per window) while the stride still resolves in-match shifts.

Only **completed** passes with a confidently inferred recipient become edges. Incomplete
passes have no recipient by definition, and unresolved ones would otherwise invent an edge.
The share dropped is reported per provider by the harmonisation report, because it is the
main channel through which the two seasons could differ for non-football reasons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tacticalgraph.data.recipient import PASS_LIKE_TYPES
from tacticalgraph.data.schema import PROVIDER_COMPARABLE_TYPES

log = logging.getLogger(__name__)

# Minimum minutes on the pitch for a player to become a node. Substitutes with a handful
# of touches produce degenerate nodes (degree 0-2) that dominate centrality distributions
# without carrying tactical meaning. 20' is a judgement call; `scripts/sensitivity.py`
# reports how much the Module 2 results move as this varies.
DEFAULT_MIN_MINUTES = 20.0

WINDOW_MINUTES = 15.0
STRIDE_MINUTES = 5.0
MATCH_MINUTES = 90.0


@dataclass(frozen=True)
class TeamNetwork:
    """One team's passing network for one game (or one window of one game)."""

    game_id: int
    team_id: int
    season: str
    provider: str
    nodes: pd.DataFrame  # player_id, mean_x, mean_y, spread, touches, passes, ...
    edges: pd.DataFrame  # source, target, weight, success_rate, mean_length
    window_index: int | None = None

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def _minutes(actions: pd.DataFrame) -> pd.Series:
    """Approximate match minute of each action.

    SPADL `time_seconds` restarts each period, so periods must be offset. Serie A data has
    two 45' periods; extra time never occurs in league play but is handled for safety.
    """
    period_offset = {1: 0.0, 2: 45.0, 3: 90.0, 4: 105.0, 5: 120.0}
    offsets = actions["period_id"].map(period_offset).fillna(0.0)
    return offsets + actions["time_seconds"] / 60.0


def player_minutes_from_actions(actions: pd.DataFrame) -> pd.DataFrame:
    """Estimate per-player time on the pitch from their first/last action.

    A deliberate approximation. StatsBomb lineups give exact spells, but Wyscout does not
    expose per-match position spells in a comparable way, so deriving minutes the same way
    for both providers keeps the node filter symmetric. It underestimates for players
    substituted on late who barely touch the ball -- which is precisely the population the
    filter is meant to exclude anyway.
    """
    minutes = _minutes(actions)
    frame = actions.assign(_minute=minutes)
    grouped = frame.groupby(["game_id", "team_id", "player_id"], dropna=True)["_minute"]
    out = grouped.agg(first_minute="min", last_minute="max").reset_index()
    out["minutes_played"] = out["last_minute"] - out["first_minute"]
    return out


def _node_table(
    actions: pd.DataFrame, eligible_players: pd.Index | None = None
) -> pd.DataFrame:
    """Per-player node attributes from their own actions."""
    frame = actions[actions["player_id"].notna()].copy()
    if eligible_players is not None:
        frame = frame[frame["player_id"].isin(eligible_players)]
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "mean_x",
                "mean_y",
                "spread_x",
                "spread_y",
                "touches",
                "passes_attempted",
                "passes_completed",
            ]
        )

    is_pass = frame["type_name"].isin(PASS_LIKE_TYPES)
    frame["_is_pass"] = is_pass
    frame["_is_completed_pass"] = is_pass & (frame["result_name"] == "success")

    # `touches` counts only provider-comparable action types. Counting *all* actions would
    # make the feature mean 790 dribbles/game in 2015/16 and 90 in 2017/18 -- i.e. it would
    # encode the provider rather than the player. See schema.PROVIDER_COMPARABLE_TYPES.
    frame["_is_comparable"] = frame["type_name"].isin(PROVIDER_COMPARABLE_TYPES)

    nodes = (
        frame.groupby("player_id")
        .agg(
            mean_x=("start_x", "mean"),
            mean_y=("start_y", "mean"),
            spread_x=("start_x", "std"),
            spread_y=("start_y", "std"),
            touches=("_is_comparable", "sum"),
            actions_all_types=("action_id", "count"),
            passes_attempted=("_is_pass", "sum"),
            passes_completed=("_is_completed_pass", "sum"),
        )
        .reset_index()
    )
    # A player with a single action has undefined std; 0 spread is the honest reading.
    nodes[["spread_x", "spread_y"]] = nodes[["spread_x", "spread_y"]].fillna(0.0)
    nodes["player_id"] = nodes["player_id"].astype("int64")
    return nodes


def _edge_table(actions: pd.DataFrame, eligible_players: pd.Index | None = None) -> pd.DataFrame:
    """Directed pass edges with weights and attributes."""
    passes = actions[
        actions["type_name"].isin(PASS_LIKE_TYPES)
        & (actions["result_name"] == "success")
        & actions["recipient_confident"].fillna(False)
        & actions["recipient_id"].notna()
        & actions["player_id"].notna()
    ].copy()

    if eligible_players is not None:
        passes = passes[
            passes["player_id"].isin(eligible_players)
            & passes["recipient_id"].isin(eligible_players)
        ]
    if passes.empty:
        return pd.DataFrame(columns=["source", "target", "weight", "mean_length", "mean_dx"])

    passes["_length"] = np.hypot(
        passes["end_x"] - passes["start_x"], passes["end_y"] - passes["start_y"]
    )
    passes["_dx"] = passes["end_x"] - passes["start_x"]

    edges = (
        passes.groupby(["player_id", "recipient_id"])
        .agg(weight=("action_id", "count"), mean_length=("_length", "mean"), mean_dx=("_dx", "mean"))
        .reset_index()
        .rename(columns={"player_id": "source", "recipient_id": "target"})
    )
    edges["source"] = edges["source"].astype("int64")
    edges["target"] = edges["target"].astype("int64")
    # Self-passes are a symptom of a bad recipient inference, not a real event.
    edges = edges[edges["source"] != edges["target"]].reset_index(drop=True)
    return edges


def build_team_network(
    actions: pd.DataFrame,
    game_id: int,
    team_id: int,
    season: str,
    provider: str,
    min_minutes: float = DEFAULT_MIN_MINUTES,
    window_index: int | None = None,
    apply_minute_filter: bool = True,
) -> TeamNetwork:
    """Build one team's network from that team's actions in a game (or window)."""
    team_actions = actions[actions["team_id"] == team_id]

    eligible: pd.Index | None = None
    if apply_minute_filter and min_minutes > 0:
        minutes = player_minutes_from_actions(team_actions)
        eligible = pd.Index(
            minutes.loc[minutes["minutes_played"] >= min_minutes, "player_id"]
            .astype("int64")
            .unique()
        )

    nodes = _node_table(team_actions, eligible)
    edges = _edge_table(team_actions, eligible)

    # Keep node/edge sets consistent: an edge must not reference a filtered-out node.
    if not nodes.empty and not edges.empty:
        known = set(nodes["player_id"])
        edges = edges[edges["source"].isin(known) & edges["target"].isin(known)]
        edges = edges.reset_index(drop=True)

    return TeamNetwork(
        game_id=int(game_id),
        team_id=int(team_id),
        season=season,
        provider=provider,
        nodes=nodes,
        edges=edges,
        window_index=window_index,
    )


def build_match_networks(
    actions: pd.DataFrame, min_minutes: float = DEFAULT_MIN_MINUTES
) -> list[TeamNetwork]:
    """Full-match networks: one per team per game, for every game in `actions`."""
    networks = []
    for (game_id, season, provider), game_actions in actions.groupby(
        ["game_id", "season", "provider"], sort=False
    ):
        for team_id in game_actions["team_id"].dropna().unique():
            networks.append(
                build_team_network(
                    game_actions,
                    game_id=int(game_id),
                    team_id=int(team_id),
                    season=str(season),
                    provider=str(provider),
                    min_minutes=min_minutes,
                )
            )
    return networks


def window_bounds(
    window_minutes: float = WINDOW_MINUTES,
    stride_minutes: float = STRIDE_MINUTES,
    match_minutes: float = MATCH_MINUTES,
) -> list[tuple[float, float]]:
    """Sliding-window bounds, e.g. (0,15), (5,20), ... (75,90)."""
    bounds = []
    start = 0.0
    while start + window_minutes <= match_minutes + 1e-9:
        bounds.append((start, start + window_minutes))
        start += stride_minutes
    return bounds


def build_windowed_networks(
    actions: pd.DataFrame,
    window_minutes: float = WINDOW_MINUTES,
    stride_minutes: float = STRIDE_MINUTES,
) -> list[TeamNetwork]:
    """Windowed networks for Module 3's graph sequence.

    The minute filter is *not* applied here: within a 15-minute window nobody has 20
    minutes of play, so the full-match eligibility rule would empty every graph. Sparsity
    within a window is handled by the window length itself.
    """
    bounds = window_bounds(window_minutes, stride_minutes)
    networks = []

    for (game_id, season, provider), game_actions in actions.groupby(
        ["game_id", "season", "provider"], sort=False
    ):
        minutes = _minutes(game_actions)
        for window_index, (low, high) in enumerate(bounds):
            mask = (minutes >= low) & (minutes < high)
            window_actions = game_actions[mask.to_numpy()]
            if window_actions.empty:
                continue
            for team_id in window_actions["team_id"].dropna().unique():
                networks.append(
                    build_team_network(
                        window_actions,
                        game_id=int(game_id),
                        team_id=int(team_id),
                        season=str(season),
                        provider=str(provider),
                        window_index=window_index,
                        apply_minute_filter=False,
                    )
                )
    return networks


def networks_to_frames(networks: list[TeamNetwork]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten a list of networks into (nodes, edges) tables for parquet storage."""
    node_frames, edge_frames = [], []
    for network in networks:
        keys = {
            "game_id": network.game_id,
            "team_id": network.team_id,
            "season": network.season,
            "provider": network.provider,
            "window_index": network.window_index,
        }
        if not network.nodes.empty:
            node_frames.append(network.nodes.assign(**keys))
        if not network.edges.empty:
            edge_frames.append(network.edges.assign(**keys))

    nodes = (
        pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()
    )
    edges = (
        pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    )
    return nodes, edges
