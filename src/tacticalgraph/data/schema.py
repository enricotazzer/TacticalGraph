"""Canonical schema definitions and the intersection rule.

The project's central data constraint: Module 3 trains on 2015/16 (StatsBomb) and tests
on 2017/18 (Wyscout). Any feature that exists in only one provider would silently vanish
at test time, so **model inputs are restricted to the intersection of the two providers**.

SPADL already is that intersection -- it is a provider-agnostic action representation, and
socceraction ships official converters for both sources. We therefore adopt SPADL as the
canonical schema rather than inventing one, and add only the derived columns both
providers can support (inferred recipient, reconstructed possession).

StatsBomb-only richness (carries, pressures, true pass recipients, 24-class positions,
`statsbomb_xg`, 360 freeze-frames) is written to a *separate* enrichment table and is
used exclusively for validating the harmonization. It must never become a model input for
Modules 2-4; `assert_no_enrichment_leakage` enforces that mechanically.
"""

from __future__ import annotations

import pandas as pd

# Columns produced by socceraction's SPADL converters.
SPADL_COLUMNS: tuple[str, ...] = (
    "game_id",
    "original_event_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "type_id",
    "result_id",
    "bodypart_id",
)

# Columns this project derives on top of SPADL, identically for both providers.
DERIVED_COLUMNS: tuple[str, ...] = (
    "action_id",
    "season",
    "provider",
    "recipient_id",
    "recipient_confident",
    "possession_id",
    "type_name",
    "result_name",
)

CANONICAL_COLUMNS: tuple[str, ...] = SPADL_COLUMNS + DERIVED_COLUMNS

# Action types whose *per-game rate* is comparable between the two providers, measured by
# `scripts/validate_harmonization.py` on the full 760-game corpus (ratio within 0.75-1.33):
#
#     pass    847.7 vs 874.0  (0.97)      throw_in  43.4 vs 42.2  (1.03)
#     shot     24.7 vs  23.2  (1.07)      goalkick  17.1 vs 17.2  (0.99)
#     take_on  30.8 vs  27.5  (1.12)      corners    ~10 vs  ~10  (~0.9)
#
# Excluded because the providers disagree by multiples, not margins:
#
#     bad_touch    29.6 vs  0.1  (296x!)  dribble     790.0 vs 90.4  (8.7x)
#     tackle       36.7 vs  8.7  (4.2x)   interception 26.3 vs 86.0  (0.31x)
#
# Those gaps are annotation convention, not football. Any feature that *counts actions*
# must therefore be built from this set only, or it will encode "which provider is this"
# and collapse when Module 3 tests on 2017/18.
#
# Pass-like types are safe in aggregate (974.1 vs 1005.9 per game, ratio 0.968) even though
# the pass/cross split differs by convention, which is why passing networks are viable.
PROVIDER_COMPARABLE_TYPES: frozenset[str] = frozenset(
    {
        "pass",
        "cross",
        "throw_in",
        "corner_crossed",
        "corner_short",
        "freekick_short",
        "goalkick",
        "shot",
        "shot_freekick",
        "shot_penalty",
        "take_on",
        "keeper_save",
        "foul",
    }
)

# Fields that exist only in StatsBomb. Quarantined to the enrichment table; validation
# use only. Kept as a named constant so the leakage guard and the docs cannot drift apart.
ENRICHMENT_ONLY_COLUMNS: frozenset[str] = frozenset(
    {
        "true_recipient_id",
        "statsbomb_xg",
        "statsbomb_possession",
        "position_name_24",
        "position_id_24",
        "under_pressure",
        "carry_end_x",
        "carry_end_y",
        "play_pattern",
    }
)


class LeakageError(AssertionError):
    """Raised when validation-only data reaches a modelling code path."""


def assert_no_enrichment_leakage(frame: pd.DataFrame, context: str = "") -> None:
    """Fail loudly if a StatsBomb-only column reaches a model input frame.

    Cheap to call and worth calling everywhere a feature matrix is assembled: the failure
    it prevents (a model that works in 2015/16 and collapses in 2017/18) is expensive and
    hard to diagnose after the fact.
    """
    offenders = sorted(ENRICHMENT_ONLY_COLUMNS.intersection(frame.columns))
    if offenders:
        raise LeakageError(
            f"{context or 'frame'} contains StatsBomb-only column(s) {offenders}. "
            "These do not exist in the Wyscout test season and must not be model inputs. "
            "They belong in the enrichment table (validation only)."
        )


def validate_canonical(frame: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Check a canonical action frame has the expected columns and sane values."""
    missing = [c for c in CANONICAL_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{context or 'frame'} missing canonical columns: {missing}")

    if frame["period_id"].isna().any():
        raise ValueError(f"{context}: null period_id")
    if (frame["time_seconds"] < 0).any():
        raise ValueError(f"{context}: negative time_seconds")

    # SPADL normalises to a 105x68 pitch. A few units of overshoot happens at the
    # touchline; anything wilder means the converter or the flip went wrong.
    for axis, limit in (("start_x", 105.0), ("end_x", 105.0), ("start_y", 68.0), ("end_y", 68.0)):
        values = frame[axis].dropna()
        if not values.empty and (values.min() < -2.0 or values.max() > limit + 2.0):
            raise ValueError(
                f"{context}: {axis} out of pitch bounds "
                f"[{values.min():.1f}, {values.max():.1f}] vs [0, {limit}]"
            )
    return frame
