"""Expected Threat (xT), fitted once and shared by every module that needs it.

`socceraction`'s `ExpectedThreat` learns a value surface over a coarse pitch grid: the probability
that a possession starting in a cell ends in a goal. The delta across a pass is then "how much
more dangerous did this move make us", which is the closest thing this project has to a
tactical-value label that nobody hand-annotated.

**The one rule that governs this module: fit on the training fold only.** The surface is learned
from data, so fitting it on the whole corpus lets the test fold shape the features that are then
used to predict it. That is a subtle leak -- the surface is only 192 numbers, so it looks harmless
-- but it is a leak, and every call site here passes an explicit set of training game ids so the
constraint is visible at the call rather than buried.

This existed as three near-identical copies in `scripts/train_outcome.py`,
`scripts/train_patterns.py` and `scripts/estimate_ceiling.py`, all with the same `l=16, w=12` grid.
Three copies of a leakage-sensitive fit is three places for the rule to be broken independently.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tacticalgraph.data.recipient import PASS_LIKE_TYPES

log = logging.getLogger(__name__)

# 16 x 12 cells over a 105 x 68 m pitch: ~6.6 m x 5.7 m per cell. Matches the grid the three
# original copies used, so the numbers already reported stay reproducible.
GRID_LENGTH = 16
GRID_WIDTH = 12


def fit_xthreat(
    actions: pd.DataFrame,
    train_game_ids: set[int] | frozenset[int],
    grid_length: int = GRID_LENGTH,
    grid_width: int = GRID_WIDTH,
) -> pd.Series:
    """Fit xT on `train_game_ids` only, then rate *every* action.

    Returns a Series indexed like `actions`, so it can be assigned straight back onto the frame.
    Non-move actions (shots, fouls, ...) get NaN from `socceraction` and are the caller's problem
    to fill -- usually with 0.0, meaning "moved the ball nowhere".

    Imported lazily: `socceraction` pulls in a large dependency tree, and the scripts that do not
    need xT should not pay for it.
    """
    from socceraction.xthreat import ExpectedThreat

    if not train_game_ids:
        raise ValueError("train_game_ids is empty; xT must be fitted on a non-empty training fold")

    train_actions = actions[actions["game_id"].isin(train_game_ids)]
    if train_actions.empty:
        raise ValueError(
            "no actions match train_game_ids -- check the split was assigned before fitting"
        )

    model = ExpectedThreat(l=grid_length, w=grid_width)
    model.fit(train_actions)
    values = model.rate(actions)

    rated = int(np.sum(~np.isnan(values)))
    log.info(
        "xT fitted on %d train games (%d actions); rated %d/%d actions (rest are non-move types)",
        train_actions["game_id"].nunique(),
        len(train_actions),
        rated,
        len(values),
    )
    return pd.Series(values, index=actions.index)


def player_threat(
    actions: pd.DataFrame,
    xt_values: pd.Series,
    group_keys: tuple[str, ...] = ("game_id", "team_id", "season", "provider"),
) -> pd.DataFrame:
    """Per-player xT generated, as a share of their team's total in the same group.

    A *share* rather than a sum, for the reason every other node feature is a share: raw totals
    carry the provider's action-density signature, and Module 2's whole point is comparing across
    providers. `group_keys` mirrors `engineer_node_features` -- pass the window-aware keys for
    windowed networks or the aggregate is a full-match value repeated across every window.

    Only positive deltas count. A back-pass has negative xT delta, and summing signed values would
    let a player cancel their own progressive passes out to zero, which reads as "not involved"
    rather than "recycled possession".
    """
    keys = list(group_keys)
    frame = actions[[*keys, "player_id"]].copy()
    frame["xt"] = pd.Series(xt_values, index=actions.index).fillna(0.0).clip(lower=0.0).to_numpy()
    frame = frame[frame["player_id"].notna()]

    per_player = (
        frame.groupby([*keys, "player_id"], dropna=False)["xt"].sum().reset_index()
    )
    team_total = (
        per_player.groupby(keys, dropna=False)["xt"].sum().rename("team_xt").reset_index()
    )
    per_player = per_player.merge(team_total, on=keys, how="left")
    per_player["xt_generated"] = (
        per_player["xt"] / per_player["team_xt"].replace(0.0, np.nan)
    ).fillna(0.0)
    per_player["player_id"] = per_player["player_id"].astype("int64")
    return per_player[[*keys, "player_id", "xt_generated"]]


def xt_edge_weights(
    actions: pd.DataFrame,
    xt_values: pd.Series,
    group_keys: tuple[str, ...] = ("game_id", "team_id", "season", "provider"),
) -> pd.DataFrame:
    """Per-edge xT: how much threat flowed along each pass lane, on `_edge_table`'s exact edge set.

    The edge set is *reproduced* rather than inferred from the persisted edges, so the pass
    predicate here is the one `passing_network._edge_table` filters with, term for term, and the
    self-pass drop happens after the groupby for the same reason it does there. An xT weight
    attached to the wrong edge is invisible -- a plausible number in a plausible row -- so the two
    filters have to agree by construction rather than by coincidence.

    `weight` is not a feature. It is the recomputed pass count, returned so that the caller can
    assert it equals the persisted `weight` on `full_edges.parquet` after a left join; that
    assertion is the guard proving the join is aligned and the xT weights are not silently attached
    to the wrong edges. If it fails, `xt_weight` is meaningless rather than merely imprecise.

    Only positive deltas count. A back-pass has negative xT delta, and summing signed values would
    let a player cancel their own progressive passes out to zero, which reads as "not involved"
    rather than "recycled possession".

    `group_keys` has to match the granularity of the edge table this will be joined to. For the
    windowed tables that means including `window_index`, and the caller must have assigned a
    `window_index` column onto `actions` first -- raw SPADL actions carry no such column. Leaving
    the match-level default in place against windowed edges is the exact silent future leak
    `engineer_node_features` documents at role_gnn.py:152-163: every window of a match would
    receive that match's whole-match xT, so a model predicting at minute 15 sees threat that was
    generated at minute 90.
    """
    keys = list(group_keys)
    columns = [*keys, "source", "target", "xt_weight", "weight"]

    passes = actions[
        actions["type_name"].isin(PASS_LIKE_TYPES)
        & (actions["result_name"] == "success")
        & actions["recipient_confident"].fillna(False)
        & actions["recipient_id"].notna()
        & actions["player_id"].notna()
    ]
    if passes.empty:
        return pd.DataFrame(columns=columns)

    frame = passes[[*keys, "player_id", "recipient_id", "action_id"]].copy()
    frame["xt"] = (
        pd.Series(xt_values, index=actions.index)
        .reindex(passes.index)
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy()
    )

    edges = (
        frame.groupby([*keys, "player_id", "recipient_id"], dropna=False)
        .agg(xt_weight=("xt", "sum"), weight=("action_id", "count"))
        .reset_index()
        .rename(columns={"player_id": "source", "recipient_id": "target"})
    )
    edges["source"] = edges["source"].astype("int64")
    edges["target"] = edges["target"].astype("int64")
    edges = edges[edges["source"] != edges["target"]].reset_index(drop=True)
    return edges[columns]


def attach_xt_edge_weights(
    edges: pd.DataFrame,
    actions: pd.DataFrame,
    xt_values: pd.Series,
    group_keys: tuple[str, ...] = ("game_id", "team_id", "season", "provider"),
) -> pd.DataFrame:
    """Left-join per-edge xT onto a persisted edge table, proving the join is aligned.

    This is why the network tables never have to be rebuilt to carry xT. `xt_edge_weights`
    reproduces `_edge_table`'s edge set from the actions, and the persisted table keeps its
    min-minutes eligibility filter for free: the computed set is a superset, so the left join
    drops exactly the edges the filter already removed. The stored artifacts stay split-free,
    which matters because xT is fitted per fold and the network tables are not.

    Three assertions, because a misaligned join here is invisible -- a plausible xT number in a
    plausible row, attached to the wrong pair of players:

    1. the row count is unchanged (no fan-out from duplicate keys);
    2. every persisted edge found a match (the computed set really is a superset);
    3. the recomputed pass count equals the persisted `weight` **exactly** for every row.

    (3) is the load-bearing one. `weight` is an integer count derived from the same predicate on
    the same actions, so any disagreement means the two groupings are not the same grouping, and
    `xt_weight` is then meaningless rather than merely imprecise.
    """
    keys = list(group_keys) + ["source", "target"]
    computed = xt_edge_weights(actions, xt_values, group_keys).rename(
        columns={"weight": "_recomputed_weight"}
    )

    before = len(edges)
    merged = edges.merge(computed, on=keys, how="left")
    if len(merged) != before:
        raise ValueError(
            f"xT edge join changed the row count ({before} -> {len(merged)}); the computed edge "
            "table has duplicate (network, source, target) keys"
        )

    unmatched = int(merged["_recomputed_weight"].isna().sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} of {before} persisted edges had no recomputed counterpart; the pass "
            "predicate here has drifted from passing_network._edge_table"
        )

    mismatched = int((merged["_recomputed_weight"] != merged["weight"]).sum())
    if mismatched:
        raise ValueError(
            f"{mismatched} of {before} edges disagree on pass count between the persisted table "
            "and the recomputation; the xT weights would be attached to the wrong edges"
        )

    return merged.drop(columns=["_recomputed_weight"])
