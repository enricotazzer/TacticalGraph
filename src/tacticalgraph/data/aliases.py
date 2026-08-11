"""Cross-provider identity resolution for clubs and players.

Team and player ids are provider-private: Juventus is 224 in StatsBomb 2015/16 and 3159 in
Wyscout 2017/18. Module 2's cross-season validation ("is a player's functional-role
embedding stable across seasons?") is impossible without a bridge between the two.

Clubs are handled with an explicit hand-checked table. There are only 20 per season and the
names differ in ways no normaliser gets right reliably ("Inter Milan" vs "Internazionale",
"AC Milan" vs "Milan"), so a table is both cheaper and more trustworthy than fuzzy matching.

Players cannot be tabulated by hand -- there are ~3,600 in the Wyscout file alone -- so they
are matched on normalised names, and the match *rate* is reported rather than assumed. Any
analysis that depends on player linkage is restricted to confidently matched players.

Of the 20 clubs per season, **16 appear in both**; 4 were relegated after 2015/16 (Empoli,
Frosinone, Carpi, Palermo) and 4 promoted for 2017/18 (Cagliari, Crotone, SPAL, Benevento).
"""

from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd

log = logging.getLogger(__name__)

# Canonical club key -> provider-specific ids. Verified against both providers' team lists.
CLUB_IDS: dict[str, dict[str, int | None]] = {
    "juventus": {"statsbomb": 224, "wyscout": 3159},
    "hellas_verona": {"statsbomb": 226, "wyscout": 3194},
    "napoli": {"statsbomb": 227, "wyscout": 3187},
    "atalanta": {"statsbomb": 228, "wyscout": 3172},
    "roma": {"statsbomb": 229, "wyscout": 3158},
    "udinese": {"statsbomb": 230, "wyscout": 3163},
    "chievo": {"statsbomb": 231, "wyscout": 3165},
    "sassuolo": {"statsbomb": 232, "wyscout": 3315},
    "genoa": {"statsbomb": 233, "wyscout": 3193},
    "sampdoria": {"statsbomb": 234, "wyscout": 3164},
    "lazio": {"statsbomb": 236, "wyscout": 3162},
    "internazionale": {"statsbomb": 238, "wyscout": 3161},
    "fiorentina": {"statsbomb": 239, "wyscout": 3176},
    "bologna": {"statsbomb": 240, "wyscout": 3166},
    "torino": {"statsbomb": 241, "wyscout": 3185},
    "milan": {"statsbomb": 243, "wyscout": 3157},
    # Relegated after 2015/16 -- present in StatsBomb only.
    "empoli": {"statsbomb": 290, "wyscout": None},
    "frosinone": {"statsbomb": 291, "wyscout": None},
    "carpi": {"statsbomb": 1683, "wyscout": None},
    "palermo": {"statsbomb": 2256, "wyscout": None},
    # Promoted for 2017/18 -- present in Wyscout only.
    "cagliari": {"statsbomb": None, "wyscout": 3173},
    "crotone": {"statsbomb": None, "wyscout": 3197},
    "spal": {"statsbomb": None, "wyscout": 3204},
    "benevento": {"statsbomb": None, "wyscout": 3219},
}

# Display names for plots and reports.
CLUB_DISPLAY: dict[str, str] = {
    "juventus": "Juventus",
    "hellas_verona": "Hellas Verona",
    "napoli": "Napoli",
    "atalanta": "Atalanta",
    "roma": "Roma",
    "udinese": "Udinese",
    "chievo": "Chievo",
    "sassuolo": "Sassuolo",
    "genoa": "Genoa",
    "sampdoria": "Sampdoria",
    "lazio": "Lazio",
    "internazionale": "Internazionale",
    "fiorentina": "Fiorentina",
    "bologna": "Bologna",
    "torino": "Torino",
    "milan": "Milan",
    "empoli": "Empoli",
    "frosinone": "Frosinone",
    "carpi": "Carpi",
    "palermo": "Palermo",
    "cagliari": "Cagliari",
    "crotone": "Crotone",
    "spal": "SPAL",
    "benevento": "Benevento",
}


def clubs_in_both_seasons() -> list[str]:
    """Canonical keys for clubs present in both seasons (the cross-season cohort)."""
    return sorted(
        key
        for key, ids in CLUB_IDS.items()
        if ids["statsbomb"] is not None and ids["wyscout"] is not None
    )


def team_id_to_club(provider: str) -> dict[int, str]:
    """Provider team_id -> canonical club key."""
    return {
        ids[provider]: key
        for key, ids in CLUB_IDS.items()
        if ids.get(provider) is not None
    }


def club_to_team_id(provider: str) -> dict[str, int]:
    return {
        key: ids[provider]
        for key, ids in CLUB_IDS.items()
        if ids.get(provider) is not None
    }


def add_club_key(frame: pd.DataFrame, provider_column: str = "provider") -> pd.DataFrame:
    """Attach a `club` column derived from (provider, team_id)."""
    out = frame.copy()
    mappings = {p: team_id_to_club(p) for p in ("statsbomb", "wyscout")}
    out["club"] = [
        mappings.get(provider, {}).get(int(team_id))
        if pd.notna(team_id)
        else None
        for provider, team_id in zip(out[provider_column], out["team_id"])
    ]
    return out


# --------------------------------------------------------------------------------------
# Player matching
# --------------------------------------------------------------------------------------

_PUNCT = re.compile(r"[^a-z ]+")
_SPACES = re.compile(r"\s+")


def normalise_name(name: str | None) -> str:
    """Fold accents, drop punctuation, lowercase, collapse spaces.

    'Gonzalo Higuaín' -> 'gonzalo higuain'; 'M. Pjanić' -> 'm pjanic'.
    """
    if not name or not isinstance(name, str):
        return ""
    folded = unicodedata.normalize("NFKD", name)
    ascii_only = folded.encode("ascii", "ignore").decode()
    cleaned = _PUNCT.sub(" ", ascii_only.lower())
    return _SPACES.sub(" ", cleaned).strip()


def _surname_key(name: str) -> str:
    """Last token of a normalised name, used as a weaker fallback key.

    Wyscout's `shortName` is often 'G. Higuain' while StatsBomb has the full name, so the
    given name is frequently just an initial and cannot be relied on.
    """
    parts = normalise_name(name).split()
    return parts[-1] if parts else ""


def match_players(
    statsbomb: pd.DataFrame,
    wyscout: pd.DataFrame,
    sb_name_col: str = "player_name",
    wy_name_col: str = "shortName",
) -> pd.DataFrame:
    """Match players across providers on normalised full name, then surname.

    Returns one row per matched pair with the rule that produced it, so downstream code can
    require `match_rule == "full_name"` when precision matters more than coverage.
    Surname collisions (two players sharing a surname on either side) are dropped rather
    than resolved arbitrarily.
    """
    sb = statsbomb[["player_id", sb_name_col]].drop_duplicates().copy()
    wy = wyscout[["player_id", wy_name_col]].drop_duplicates().copy()
    sb["_full"] = sb[sb_name_col].map(normalise_name)
    wy["_full"] = wy[wy_name_col].map(normalise_name)

    exact = sb.merge(wy, on="_full", how="inner", suffixes=("_sb", "_wy"))
    exact["match_rule"] = "full_name"

    matched_sb = set(exact["player_id_sb"])
    matched_wy = set(exact["player_id_wy"])

    sb_left = sb[~sb["player_id"].isin(matched_sb)].copy()
    wy_left = wy[~wy["player_id"].isin(matched_wy)].copy()
    sb_left["_surname"] = sb_left[sb_name_col].map(_surname_key)
    wy_left["_surname"] = wy_left[wy_name_col].map(_surname_key)

    # Only surnames that are unique on both sides can be matched without guessing.
    sb_unique = sb_left["_surname"].value_counts()
    wy_unique = wy_left["_surname"].value_counts()
    safe = set(sb_unique[sb_unique == 1].index) & set(wy_unique[wy_unique == 1].index)
    safe.discard("")

    surname = sb_left[sb_left["_surname"].isin(safe)].merge(
        wy_left[wy_left["_surname"].isin(safe)],
        on="_surname",
        how="inner",
        suffixes=("_sb", "_wy"),
    )
    surname["match_rule"] = "surname"

    columns = ["player_id_sb", "player_id_wy", "match_rule"]
    frames = [exact[columns]]
    if not surname.empty:
        frames.append(surname[columns])
    matches = pd.concat(frames, ignore_index=True)

    log.info(
        "player matching: %d full-name + %d surname = %d pairs "
        "(%d StatsBomb / %d Wyscout candidates)",
        int((matches["match_rule"] == "full_name").sum()),
        int((matches["match_rule"] == "surname").sum()),
        len(matches),
        len(sb),
        len(wy),
    )
    return matches
