"""Module 4: possession chains as feature vectors and as action sequences.

A "chain" is one possession, already segmented by `data.possession.reconstruct_possessions`
(scored at ARI 0.832 against StatsBomb's native counter).

Two representation choices are forced by the data rather than by taste:

**Chains are built from `PROVIDER_COMPARABLE_TYPES` only.** Measured on the full corpus, raw
chain length differs between the providers by a factor of 1.44 (mean 7.91 actions in 2015/16
against 5.51 in 2017/18) purely because StatsBomb logs carries as dribbles. Filtering to
comparable types collapses that to 0.90 (4.38 vs 4.87, median 3 vs 3). Without the filter,
any clustering would partly be clustering *the data provider*.

**Chains shorter than 3 actions are dropped.** A one- or two-action possession has no
sequence structure to model.

The reporting target is `ends_in_shot`. Its base rate is **9.7% over all 186,318 raw chains**,
but **12.4% over the 109,912 chains that survive the >=3-comparable-action filter** -- longer
possessions are likelier to produce a shot, so filtering raises the base rate. Every cluster
is measured against the *filtered* 12.4%, since that is the population being clustered.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tacticalgraph.data.schema import PROVIDER_COMPARABLE_TYPES
from tacticalgraph.features.match_state import action_minutes

log = logging.getLogger(__name__)

MIN_CHAIN_ACTIONS = 3

# Hand-crafted features: the interpretable baseline the learned encoder must beat, and also
# what makes a cluster nameable rather than an anonymous index.
CHAIN_FEATURES: tuple[str, ...] = (
    "n_actions",
    "duration_seconds",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "net_dx",
    "path_length",
    "directness",
    "mean_pass_length",
    "xt_gain",
    "started_with_set_piece",
    "share_final_third",
    "width_used",
)

SET_PIECE_TYPES: frozenset[str] = frozenset(
    {"throw_in", "corner_crossed", "corner_short", "freekick_short", "freekick_crossed", "goalkick"}
)

# Coarse pitch thirds, for naming clusters.
THIRD_EDGES = (35.0, 70.0)


def zone_of(x: float) -> str:
    if x < THIRD_EDGES[0]:
        return "defensive"
    if x < THIRD_EDGES[1]:
        return "middle"
    return "final"


def shot_possessions(actions: pd.DataFrame) -> set[tuple[int, int]]:
    """The `(game_id, possession_id)` pairs whose possession contains at least one shot.

    Membership is a `type_name` prefix test, not an equality test: SPADL spells the restart
    variants `shot_penalty` and `shot_freekick` alongside the plain `shot`, and matching on
    equality would quietly drop every penalty and direct free kick from the target.

    The pairs are cast to `(int, int)` rather than left as whatever `to_numpy()` produced, so
    that the key type is fixed by this function instead of by the caller's column dtypes. Every
    call site looks membership up as `(int(g), int(p))`; leaving the set keyed on a float or
    object array would make those lookups depend on Python's cross-type equality happening to
    hold, which is correct today and silently is not the moment a column arrives as a nullable
    or string dtype.
    """
    if actions.empty:
        return set()
    is_shot = actions["type_name"].str.startswith("shot")
    pairs = actions.loc[is_shot, ["game_id", "possession_id"]].to_numpy()
    return {(int(game_id), int(possession_id)) for game_id, possession_id in pairs}


def build_chain_table(
    actions: pd.DataFrame,
    xt_values: pd.Series | None = None,
    min_actions: int = MIN_CHAIN_ACTIONS,
) -> pd.DataFrame:
    """One row per possession chain, restricted to provider-comparable actions.

    `xt_values` should come from an xThreat model fit on the training split only; passing
    None yields zero xT gain, which keeps the function testable without a fitted model.
    """
    frame = actions.copy()
    if xt_values is not None:
        frame["xt_value"] = pd.Series(xt_values, index=actions.index).fillna(0.0).to_numpy()
    else:
        frame["xt_value"] = 0.0

    frame["_minute"] = action_minutes(frame)

    # Shot membership is determined BEFORE the comparable-type filter, because `shot` is
    # itself comparable but we want the flag even for chains whose shot is the only surviving
    # action of its kind.
    shot_chains = shot_possessions(frame)

    frame = frame[frame["type_name"].isin(PROVIDER_COMPARABLE_TYPES)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["game_id", "possession_id", *CHAIN_FEATURES])

    frame["_dx"] = frame["end_x"] - frame["start_x"]
    frame["_dy"] = frame["end_y"] - frame["start_y"]
    frame["_step"] = np.hypot(frame["_dx"], frame["_dy"])
    frame["_final_third"] = frame["start_x"] >= THIRD_EDGES[1]
    frame["_is_pass"] = frame["type_name"].isin({"pass", "cross"})

    grouped = frame.sort_values(["game_id", "possession_id", "period_id", "time_seconds"]).groupby(
        ["game_id", "possession_id"], sort=False
    )

    table = grouped.agg(
        season=("season", "first"),
        provider=("provider", "first"),
        team_id=("team_id", "first"),
        period_id=("period_id", "first"),
        start_minute=("_minute", "min"),
        end_minute=("_minute", "max"),
        n_actions=("action_id", "size"),
        start_x=("start_x", "first"),
        start_y=("start_y", "first"),
        end_x=("end_x", "last"),
        end_y=("end_y", "last"),
        path_length=("_step", "sum"),
        mean_pass_length=("_step", "mean"),
        xt_gain=("xt_value", "sum"),
        first_type=("type_name", "first"),
        share_final_third=("_final_third", "mean"),
        min_y=("start_y", "min"),
        max_y=("start_y", "max"),
    ).reset_index()

    table["duration_seconds"] = (table["end_minute"] - table["start_minute"]) * 60.0
    table["net_dx"] = table["end_x"] - table["start_x"]
    # Directness: how much of the distance travelled was forward progress. 1.0 = a straight
    # vertical line towards goal, near 0 = sideways circulation.
    table["directness"] = (
        table["net_dx"] / table["path_length"].replace(0.0, np.nan)
    ).fillna(0.0).clip(-1.0, 1.0)
    table["started_with_set_piece"] = table["first_type"].isin(SET_PIECE_TYPES).astype(float)
    table["width_used"] = table["max_y"] - table["min_y"]

    table["ends_in_shot"] = [
        (int(g), int(p)) in shot_chains
        for g, p in zip(table["game_id"], table["possession_id"])
    ]
    table["start_zone"] = table["start_x"].map(zone_of)
    table["end_zone"] = table["end_x"].map(zone_of)

    table = table[table["n_actions"] >= min_actions].reset_index(drop=True)

    log.info(
        "chain table: %d chains (>=%d actions) | shot rate %.3f | by season %s",
        len(table),
        min_actions,
        table["ends_in_shot"].mean(),
        table.groupby("season").size().to_dict(),
    )
    return table


def shot_chain_involvement(
    actions: pd.DataFrame,
    group_keys: tuple[str, ...] = ("game_id", "team_id", "season", "provider"),
) -> pd.DataFrame:
    """Two ways of asking whether a player was involved in shots, with different denominators.

    `shot_involvement` is the player's share of the group's shot-ending possessions.
    `shot_conversion` is the share of the possessions *the player was in* that ended in a shot.

    **The denominator is the whole point, and the first one does not do what it was built to do.**
    `shot_involvement` divides by a quantity that is constant within a team-match, so it never
    normalises away the player's own touch frequency: a player who is in more possessions is in
    more shot-ending possessions, and the metric inherits that. Measured on the Premier League
    corpus it correlates **+0.71** with `degree_total`, against a pre-registered bar of < 0.70 --
    i.e. it is substantially a third volume proxy, which is exactly what it was meant not to be.
    `shot_conversion` conditions on the player's own involvement instead, and the volume signal
    collapses: **rho vs `degree_total` falls from +0.69 to +0.04** on the same cohort.

    What survives that is *not* role-neutral. `shot_conversion`'s top 50 is 74% forwards against a
    26% population share, because forwards are in the box when shots happen. Removing the volume
    component leaves the positional one, which is the finding these features were built to test
    rather than a defect in them. See `features/centrality.residualise_against_position`.

    This is the one proposed metric that ranks a player on the *outcome* of the possessions they
    appear in, rather than on pass volume or on receiving direction. That is the limitation it
    exists to attack: with pass-only edges, centrality is a positional proxy -- midfielders are
    31% of players with >=10 matches but **84% of the top 50 by `degree_total`**, and goalkeepers
    take **0%** of the top 50 on all ten metrics. A forward with six touches and a share in three
    of his team's shot chains is unrankable by every graph feature in this project. Involvement
    counts possessions rather than actions, so one touch in a shot-ending chain scores exactly as
    much as five, which is what keeps it from collapsing back into a volume measure.

    **Nothing here is fitted, so there is no train-fold rule to obey.** Worth stating explicitly
    because this sits next to features that do have one: `features/xthreat.fit_xthreat` learns a
    value surface and must see training games only, and `xthreat.player_threat` inherits that
    constraint through the values handed to it. `shot_involvement` is a count over whatever
    actions it is given, so it can be computed on any fold, in any order, and the only thing the
    input decides is which matches get summarised -- there is nothing for the test fold to shape.

    **Possessions are not reassigned to an owning team.** `data.possession.reconstruct_possessions`
    segments the ball, so a defensive touch by the other team falls inside the same
    `possession_id`. Rather than pick a single owner, each player is counted within their own
    group's rows and the denominator is the distinct shot-ending possessions appearing in those
    same rows. This is a deliberate simplification, and it means a defender who touches the ball
    during the *opponent's* shot-ending possession is credited with involvement inside his own
    team's group. The metric therefore reads as "was part of the play that ended in a shot",
    attacking or defending, and not as "helped create a shot"; separating the two needs the
    possession's owning team, which this layer does not carry.

    `group_keys` mirrors `xthreat.player_threat` and `models/role_gnn.engineer_node_features`.
    Passing `window_index` among them requires the caller to have assigned that column onto
    `actions` first -- it is a property of the windowed network tables, not of the action log --
    and its absence raises here rather than silently returning full-match shares.

    A group with no shot-ending possession gives every one of its players 0.0 on both metrics,
    not NaN: the same convention as the `_safe_ratio` shares elsewhere, where "no evidence" reads
    as zero and a NaN would drop the player out of any downstream sort. `shot_conversion` is
    additionally 0.0 for a player with no actions in the group at all, which cannot arise from
    this function's own universe but can if a caller reindexes the result.
    """
    keys = list(dict.fromkeys(group_keys))
    columns = [*keys, "player_id", "shot_involvement", "shot_conversion"]

    chains = shot_possessions(actions)
    if not chains:
        return pd.DataFrame(columns=columns)

    chain_keys = list(dict.fromkeys([*keys, "game_id", "possession_id"]))
    frame = actions[[*chain_keys, "player_id"]].copy()
    frame = frame[frame["player_id"].notna()]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["_in_shot_chain"] = [
        (int(game_id), int(possession_id)) in chains
        for game_id, possession_id in zip(frame["game_id"], frame["possession_id"])
    ]
    universe = frame[[*keys, "player_id"]].drop_duplicates()

    involved = frame[frame["_in_shot_chain"]].drop_duplicates([*chain_keys, "player_id"])
    per_player = (
        involved.groupby([*keys, "player_id"], dropna=False)
        .size()
        .rename("_player_chains")
        .reset_index()
    )
    # `shot_conversion`'s denominator: every distinct possession the player appeared in, shot or
    # not. Counted from the unfiltered frame, so it is the player's own exposure rather than the
    # team's shot count.
    touched = (
        frame.drop_duplicates([*chain_keys, "player_id"])
        .groupby([*keys, "player_id"], dropna=False)
        .size()
        .rename("_player_touched")
        .reset_index()
    )
    per_group = (
        involved.drop_duplicates(chain_keys)
        .groupby(keys, dropna=False)
        .size()
        .rename("_group_chains")
        .reset_index()
    )

    table = (
        universe.merge(per_player, on=[*keys, "player_id"], how="left")
        .merge(touched, on=[*keys, "player_id"], how="left")
        .merge(per_group, on=keys, how="left")
    )
    for column in ("_player_chains", "_group_chains", "_player_touched"):
        table[column] = table[column].fillna(0.0)
    table["shot_involvement"] = (
        table["_player_chains"] / table["_group_chains"].replace(0.0, np.nan)
    ).fillna(0.0)
    table["shot_conversion"] = (
        table["_player_chains"] / table["_player_touched"].replace(0.0, np.nan)
    ).fillna(0.0)
    table["player_id"] = table["player_id"].astype("int64")

    log.info(
        "shot involvement: %d players over %d groups | mean involvement %.3f, conversion %.3f",
        len(table),
        len(per_group),
        table["shot_involvement"].mean(),
        table["shot_conversion"].mean(),
    )
    return table[columns].reset_index(drop=True)


def chain_sequences(
    actions: pd.DataFrame,
    chain_table: pd.DataFrame,
    max_length: int = 12,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pad chains into a (n_chains, max_length, n_token_features) tensor.

    Token = [one-hot action type over the comparable vocabulary, start_x, start_y, dx, dy,
    success]. Coordinates are scaled to [0, 1] so the one-hot and the geometry are on
    comparable scales without a separate scaler.

    `max_length=12` covers well past the 90th percentile (9-10 actions); longer chains are
    truncated, which loses the tail of a handful of very long possessions.
    """
    vocabulary = sorted(PROVIDER_COMPARABLE_TYPES)
    type_index = {name: i for i, name in enumerate(vocabulary)}
    n_features = len(vocabulary) + 5

    wanted = set(map(tuple, chain_table[["game_id", "possession_id"]].to_numpy()))
    frame = actions[actions["type_name"].isin(PROVIDER_COMPARABLE_TYPES)].copy()
    frame = frame[
        [
            (int(g), int(p)) in wanted
            for g, p in zip(frame["game_id"], frame["possession_id"])
        ]
    ]
    frame = frame.sort_values(["game_id", "possession_id", "period_id", "time_seconds"])

    position = {
        (int(g), int(p)): i
        for i, (g, p) in enumerate(chain_table[["game_id", "possession_id"]].to_numpy())
    }

    tensor = np.zeros((len(chain_table), max_length, n_features), dtype=np.float32)
    lengths = np.zeros(len(chain_table), dtype=np.int64)

    for (game_id, possession_id), group in frame.groupby(["game_id", "possession_id"], sort=False):
        row = position.get((int(game_id), int(possession_id)))
        if row is None:
            continue
        group = group.head(max_length)
        for step, action in enumerate(group.itertuples(index=False)):
            token = tensor[row, step]
            token[type_index[action.type_name]] = 1.0
            offset = len(vocabulary)
            token[offset + 0] = action.start_x / 105.0
            token[offset + 1] = action.start_y / 68.0
            token[offset + 2] = (action.end_x - action.start_x) / 105.0
            token[offset + 3] = (action.end_y - action.start_y) / 68.0
            token[offset + 4] = 1.0 if action.result_name == "success" else 0.0
        lengths[row] = len(group)

    log.info(
        "chain sequences: %s | mean length %.2f | truncated %d chains",
        tensor.shape,
        lengths.mean(),
        int((chain_table["n_actions"] > max_length).sum()),
    )
    return tensor, lengths, vocabulary


def cluster_profiles(chain_table: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per-cluster summary, so each cluster can be named and judged.

    This is what turns "cluster 3" into "long build-up from the defensive third", and it is
    what the human review harness prints alongside the sampled chains.
    """
    frame = chain_table.copy()
    frame["cluster"] = labels

    profile = frame.groupby("cluster").agg(
        n_chains=("game_id", "size"),
        shot_rate=("ends_in_shot", "mean"),
        n_actions=("n_actions", "mean"),
        duration_s=("duration_seconds", "mean"),
        start_x=("start_x", "mean"),
        end_x=("end_x", "mean"),
        net_dx=("net_dx", "mean"),
        directness=("directness", "mean"),
        xt_gain=("xt_gain", "mean"),
        set_piece=("started_with_set_piece", "mean"),
        width=("width_used", "mean"),
    ).reset_index()

    profile["share_of_chains"] = profile["n_chains"] / len(frame)
    profile["label"] = _name_clusters(profile)
    return profile.round(4)


def _name_clusters(profile: pd.DataFrame) -> list[str]:
    """Human-readable names, assigned *relative to the other clusters*.

    Absolute thresholds produce useless output when a representation splits one region of the
    space several ways -- an early version labelled four of eight clusters "set-piece restart
    from middle third". Ranking each cluster against its siblings on length, directness and
    origin keeps the names distinct and informative.

    These names are a reading aid, explicitly **not** validation: they are generated from the
    same profile numbers printed beside them, so they cannot corroborate anything. Human
    judgement is `scripts/review_patterns.py`.
    """
    def rank(column: str) -> pd.Series:
        return profile[column].rank(pct=True)

    length_rank, direct_rank, xt_rank = rank("n_actions"), rank("directness"), rank("xt_gain")

    names = []
    for i, row in enumerate(profile.itertuples(index=False)):
        parts = []
        if row.set_piece > 0.6:
            parts.append("set-piece")
        elif row.set_piece < 0.2:
            parts.append("open-play")

        if length_rank.iloc[i] >= 0.75:
            parts.append("long")
        elif length_rank.iloc[i] <= 0.25:
            parts.append("brief")

        if direct_rank.iloc[i] >= 0.75:
            parts.append("direct")
        elif direct_rank.iloc[i] <= 0.25:
            parts.append("lateral")

        stem = " ".join(parts) if parts else "mixed"
        suffix = ", high threat" if xt_rank.iloc[i] >= 0.85 else ""
        names.append(f"{stem} from {zone_of(row.start_x)} third{suffix}")
    return names
