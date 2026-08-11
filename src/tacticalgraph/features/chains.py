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
    frame["_is_shot"] = frame["type_name"].str.startswith("shot")

    # Shot membership is determined BEFORE the comparable-type filter, because `shot` is
    # itself comparable but we want the flag even for chains whose shot is the only surviving
    # action of its kind.
    shot_chains = set(
        map(tuple, frame.loc[frame["_is_shot"], ["game_id", "possession_id"]].to_numpy())
    )

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
