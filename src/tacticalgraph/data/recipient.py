"""Pass-recipient inference, applied identically to both providers.

Passing networks need to know *who received each pass*. StatsBomb records this directly;
Wyscout does not record it at all, and SPADL drops it for both. Rather than use the true
recipient where available and an inferred one elsewhere -- which would make the 2015/16
networks systematically better than the 2017/18 ones and confound the cross-season test --
we infer it the same way everywhere and keep StatsBomb's ground truth purely for
measuring how good the inference is.

The rule:

    recipient = player of the next action, within the next `lookahead` actions,
                same team, different player

Measured against StatsBomb ground truth on an event stream degraded to Wyscout-like
coverage: **95.9% correct, 1.2% wrong, 2.9% unresolved** (single match, during planning;
`scripts/validate_harmonization.py` reruns this over the full season).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# SPADL action types after which a team-mate may receive the ball.
PASS_LIKE_TYPES: frozenset[str] = frozenset(
    {
        "pass",
        "cross",
        "throw_in",
        "freekick_crossed",
        "freekick_short",
        "corner_crossed",
        "corner_short",
        "goalkick",
    }
)

DEFAULT_LOOKAHEAD = 3


def infer_recipients(
    actions: pd.DataFrame, lookahead: int = DEFAULT_LOOKAHEAD
) -> pd.DataFrame:
    """Add `recipient_id` / `recipient_confident` to a single game's SPADL actions.

    Only successful pass-like actions get a recipient. Vectorised over the lookahead
    window rather than looped per row -- at 760 games x ~1.7k actions the difference is
    minutes, not seconds.
    """
    if actions.empty:
        return actions.assign(recipient_id=pd.NA, recipient_confident=False)

    frame = actions.sort_values(["period_id", "time_seconds", "action_id"]).reset_index(
        drop=True
    )

    is_pass = frame["type_name"].isin(PASS_LIKE_TYPES).to_numpy()
    is_success = (frame["result_name"] == "success").to_numpy()
    eligible = is_pass & is_success

    team = frame["team_id"].to_numpy()
    player = frame["player_id"].to_numpy(dtype="float64")
    period = frame["period_id"].to_numpy()

    recipient = np.full(len(frame), np.nan)

    # Walk the lookahead offsets, filling only slots still unresolved. Offset 1 handles
    # the overwhelming majority; deeper offsets mop up cases where a duel or an
    # interruption is logged between the pass and the touch.
    for offset in range(1, lookahead + 1):
        shifted_team = np.roll(team, -offset).astype("float64")
        shifted_player = np.roll(player, -offset)
        shifted_period = np.roll(period, -offset).astype("float64")
        # Rolled-around tail values are meaningless; mask them out.
        shifted_team[-offset:] = np.nan
        shifted_player[-offset:] = np.nan
        shifted_period[-offset:] = np.nan

        candidate = (
            eligible
            & np.isnan(recipient)
            & (shifted_team == team)
            & (shifted_period == period)  # never cross a half boundary
            & (shifted_player != player)  # a player cannot pass to themselves
            & ~np.isnan(shifted_player)
        )
        recipient = np.where(candidate, shifted_player, recipient)

    frame["recipient_id"] = pd.array(recipient, dtype="Float64").astype("Int64")
    frame["recipient_confident"] = eligible & ~np.isnan(recipient)
    return frame


def evaluate_inference(
    inferred: pd.DataFrame, truth: pd.Series, context: str = ""
) -> dict[str, float]:
    """Score inferred recipients against StatsBomb ground truth.

    `truth` must be indexed like `inferred` and hold the real recipient id (nullable).
    Returns the correct/wrong/unresolved breakdown quoted in the README.
    """
    known = truth.notna()
    eligible = inferred["type_name"].isin(PASS_LIKE_TYPES) & (
        inferred["result_name"] == "success"
    )
    mask = known & eligible
    total = int(mask.sum())
    if total == 0:
        return {"n": 0, "accuracy": float("nan"), "wrong": float("nan"), "unresolved": float("nan")}

    predicted = inferred.loc[mask, "recipient_id"]
    actual = truth.loc[mask]
    resolved = predicted.notna()

    correct = int((resolved & (predicted == actual)).sum())
    wrong = int((resolved & (predicted != actual)).sum())
    unresolved = int((~resolved).sum())

    return {
        "context": context,
        "n": total,
        "accuracy": correct / total,
        "wrong": wrong / total,
        "unresolved": unresolved / total,
    }
