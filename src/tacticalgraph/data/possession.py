"""Possession-chain reconstruction.

StatsBomb ships a native `possession` counter; Wyscout ships nothing. As with recipients,
we reconstruct chains identically for both providers so the two seasons are directly
comparable, and use StatsBomb's native field only to score the reconstruction.

Rule: a new chain starts when
  1. the game or the period changes, or
  2. a set-piece restart occurs (throw-in, corner, free-kick, goal-kick, penalty), or
  3. the controlling team changes *and holds the ball* -- a single opponent touch
     (a clearance, a failed interception, a duel) is treated as contest within the
     current chain rather than a change of possession.

Rule 3 matters: without the flicker guard, a defensive clearance that comes straight back
splits one attacking sequence into three chains, which would wreck the possession-level
sequence modelling in Module 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A set piece is a hard restart: the ball was dead, so the chain before it has ended.
SET_PIECE_TYPES: frozenset[str] = frozenset(
    {
        "throw_in",
        "corner_crossed",
        "corner_short",
        "freekick_crossed",
        "freekick_short",
        "goalkick",
        "shot_penalty",
        "shot_freekick",
    }
)

# Touches that are reactive rather than controlled: they signal contest, not possession.
CONTEST_TYPES: frozenset[str] = frozenset(
    {"clearance", "interception", "tackle", "keeper_save", "keeper_punch", "bad_touch"}
)

DEFAULT_MIN_HOLD = 2


def reconstruct_possessions(
    actions: pd.DataFrame, min_hold: int = DEFAULT_MIN_HOLD
) -> pd.DataFrame:
    """Add a `possession_id` column (monotonic within a game)."""
    if actions.empty:
        return actions.assign(possession_id=pd.Series(dtype="int64"))

    frame = actions.sort_values(["period_id", "time_seconds", "action_id"]).reset_index(
        drop=True
    )

    team = frame["team_id"].to_numpy()
    period = frame["period_id"].to_numpy()
    type_name = frame["type_name"].to_numpy()

    # Effective controlling team, with single-touch contests absorbed into the run around
    # them. A contest action by team B sandwiched between team A actions stays with A.
    controller = team.copy()
    is_contest = np.isin(type_name, list(CONTEST_TYPES))
    for index in range(1, len(frame) - 1):
        if not is_contest[index]:
            continue
        if controller[index - 1] == controller[index + 1] != controller[index]:
            controller[index] = controller[index - 1]

    # Suppress remaining short flickers: a run of the same controller shorter than
    # `min_hold`, flanked by the same other team, is absorbed.
    change_points = np.flatnonzero(np.diff(controller) != 0) + 1
    run_starts = np.concatenate(([0], change_points))
    run_ends = np.concatenate((change_points, [len(controller)]))
    for start, end in zip(run_starts, run_ends):
        if end - start >= min_hold:
            continue
        before = controller[start - 1] if start > 0 else None
        after = controller[end] if end < len(controller) else None
        if before is not None and before == after:
            controller[start:end] = before

    new_chain = np.zeros(len(frame), dtype=bool)
    new_chain[0] = True
    new_chain[1:] |= controller[1:] != controller[:-1]
    new_chain[1:] |= period[1:] != period[:-1]
    new_chain |= np.isin(type_name, list(SET_PIECE_TYPES))

    frame["possession_id"] = np.cumsum(new_chain) - 1
    frame["possession_team_id"] = controller
    return frame


def evaluate_possessions(
    reconstructed: pd.DataFrame, truth: pd.Series, context: str = ""
) -> dict[str, float]:
    """Score reconstructed chains against StatsBomb's native possession counter.

    Reported as boundary agreement (do the two segmentations start chains in the same
    places?) plus the Rand-style pair agreement, which is less sensitive to a constant
    offset in chain numbering.
    """
    if len(reconstructed) == 0 or truth.notna().sum() == 0:
        return {"context": context, "n": 0}

    ours = reconstructed["possession_id"].to_numpy()
    theirs = truth.to_numpy()

    ours_boundary = np.concatenate(([True], ours[1:] != ours[:-1]))
    theirs_boundary = np.concatenate(([True], theirs[1:] != theirs[:-1]))

    both = int((ours_boundary & theirs_boundary).sum())
    either = int((ours_boundary | theirs_boundary).sum())

    from sklearn.metrics import adjusted_rand_score

    return {
        "context": context,
        "n": int(len(reconstructed)),
        "n_chains_ours": int(ours_boundary.sum()),
        "n_chains_statsbomb": int(theirs_boundary.sum()),
        "boundary_jaccard": both / either if either else float("nan"),
        "adjusted_rand": float(adjusted_rand_score(theirs, ours)),
    }
