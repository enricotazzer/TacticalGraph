"""Position/role harmonisation across providers.

The two providers describe player position at very different resolutions:

* **StatsBomb** -- a 25-position vocabulary, per match, with time spells
  ("Right Center Midfield", "Center Attacking Midfield", ...). 24 of the 25 actually occur
  in Serie A 2015/16; "Secondary Striker" never does.
* **Wyscout** -- 4 static roles attached to the player, not the match
  (GK / DF / MD / FW).

The intersection is therefore the 4-class vocabulary, and that is what Module 2 trains
on. This is a real loss of information, but it turns into the experiment: the GNN is
supervised with 4 coarse classes, then evaluated on whether its embedding recovers
StatsBomb's fine-grained position structure *that it never saw*. If it does, "functional
role is finer than nominal position" stops being a slogan and becomes a measurement.

`COARSE_ROLES` is the canonical label space. `STATSBOMB_POSITION_TO_COARSE` is the only place
the fine -> coarse collapse is defined.
"""

from __future__ import annotations

import pandas as pd

COARSE_ROLES: tuple[str, ...] = ("GK", "DEF", "MID", "FWD")
ROLE_TO_INDEX: dict[str, int] = {role: i for i, role in enumerate(COARSE_ROLES)}

# StatsBomb's 25-position vocabulary collapsed onto the 4 classes Wyscout also supports.
# Wing-backs go to DEF and wingers to FWD, following the usual reading of those roles;
# both are judgement calls and are the kind of boundary the Module 2 embedding is meant
# to say something interesting about.
STATSBOMB_POSITION_TO_COARSE: dict[str, str] = {
    "Goalkeeper": "GK",
    # Defenders
    "Right Back": "DEF",
    "Right Center Back": "DEF",
    "Center Back": "DEF",
    "Left Center Back": "DEF",
    "Left Back": "DEF",
    "Right Wing Back": "DEF",
    "Left Wing Back": "DEF",
    # Midfielders
    "Right Defensive Midfield": "MID",
    "Center Defensive Midfield": "MID",
    "Left Defensive Midfield": "MID",
    "Right Midfield": "MID",
    "Right Center Midfield": "MID",
    "Center Midfield": "MID",
    "Left Center Midfield": "MID",
    "Left Midfield": "MID",
    "Right Attacking Midfield": "MID",
    "Center Attacking Midfield": "MID",
    "Left Attacking Midfield": "MID",
    # Forwards
    "Right Wing": "FWD",
    "Left Wing": "FWD",
    "Right Center Forward": "FWD",
    "Center Forward": "FWD",
    "Left Center Forward": "FWD",
    "Secondary Striker": "FWD",
}

# Wyscout's own 2-letter codes (`role.code2` in players.json).
WYSCOUT_CODE_TO_COARSE: dict[str, str] = {
    "GK": "GK",
    "DF": "DEF",
    "MD": "MID",
    "FW": "FWD",
}


class UnknownPositionError(KeyError):
    """A position string outside the mapping. Never silently bucket these."""


def statsbomb_to_coarse(position: str | None, strict: bool = True) -> str | None:
    """Map one fine-grained StatsBomb position onto the coarse vocabulary."""
    if position is None or (isinstance(position, float) and pd.isna(position)):
        return None
    try:
        return STATSBOMB_POSITION_TO_COARSE[position]
    except KeyError:
        if strict:
            raise UnknownPositionError(
                f"unmapped StatsBomb position {position!r}. Add it to "
                "STATSBOMB_POSITION_TO_COARSE rather than defaulting, so the collapse stays "
                "auditable."
            ) from None
        return None


def wyscout_to_coarse(code: str | None, strict: bool = True) -> str | None:
    """Map one Wyscout role code onto the coarse vocabulary."""
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return None
    try:
        return WYSCOUT_CODE_TO_COARSE[code]
    except KeyError:
        if strict:
            raise UnknownPositionError(f"unmapped Wyscout role code {code!r}") from None
        return None


def add_coarse_role(
    frame: pd.DataFrame, source_column: str, provider: str, strict: bool = True
) -> pd.DataFrame:
    """Add a `coarse_role` column derived from a provider-specific position column."""
    mapper = statsbomb_to_coarse if provider == "statsbomb" else wyscout_to_coarse
    out = frame.copy()
    out["coarse_role"] = [
        mapper(value, strict=strict) for value in out[source_column].tolist()
    ]
    return out


def coarse_role_index(roles: pd.Series) -> pd.Series:
    """Integer-encode coarse roles for a classification head."""
    return roles.map(ROLE_TO_INDEX).astype("Int64")


def verify_mapping_coverage(observed_positions: set[str], provider: str) -> list[str]:
    """Return positions present in the data but absent from the mapping.

    Called by the harmonisation report so a provider adding a position label shows up as
    a reported gap rather than a crash deep inside training.
    """
    known = (
        set(STATSBOMB_POSITION_TO_COARSE) if provider == "statsbomb" else set(WYSCOUT_CODE_TO_COARSE)
    )
    return sorted(p for p in observed_positions if p is not None and p not in known)
