"""Unified player directory across providers.

Module 2 needs, for every (season, player): a display name, a coarse role label to train
on, and -- on the StatsBomb season only -- the 24-class position kept aside as a held-out
validation signal.

The two providers supply this very differently:

* StatsBomb: per-match position spells in the lineups file, so a player's nominal position
  is match-specific and we take their most-played position across the season.
* Wyscout: a single static `role.code2` per player in `players.json`, with no per-match
  detail at all.

The coarse (4-class) label is therefore the only vocabulary both can express, which is why
it is what the GNN is supervised with. See `roles.py`.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from tacticalgraph.config import Paths
from tacticalgraph.data.download import load_json
from tacticalgraph.data.enrichment import load_enrichment
from tacticalgraph.data.roles import statsbomb_to_coarse, wyscout_to_coarse

log = logging.getLogger(__name__)


def statsbomb_players(paths: Paths) -> pd.DataFrame:
    """Season-level player directory from the StatsBomb lineup enrichment."""
    lineups = load_enrichment(paths, "statsbomb_lineup_positions")

    # Most-played 24-class position across the season becomes the player's nominal one.
    by_position = (
        lineups.groupby(["player_id", "position_name_24"])["minutes_played"]
        .sum()
        .reset_index()
        .sort_values("minutes_played", ascending=False)
    )
    nominal = by_position.drop_duplicates(subset=["player_id"])

    totals = (
        lineups.groupby("player_id")
        .agg(
            player_name=("player_name", "first"),
            minutes_total=("minutes_played", "sum"),
            matches=("game_id", "nunique"),
        )
        .reset_index()
    )

    frame = totals.merge(
        nominal[["player_id", "position_name_24"]], on="player_id", how="left"
    )
    frame["coarse_role"] = [
        statsbomb_to_coarse(p, strict=False) for p in frame["position_name_24"]
    ]
    # Season key comes from the corpus, not a Serie A constant: the Premier League corpus is
    # also provider=statsbomb, so a hardcoded key would tag its players with the wrong
    # season and silently break every (season, provider, player_id) join.
    frame["season"] = _season_key(paths, "statsbomb")
    frame["provider"] = "statsbomb"
    return frame


def _season_key(paths: Paths, provider: str) -> str:
    """The corpus's season key for one provider."""
    for season in paths.spec.seasons:
        if season.provider == provider:
            return season.key
    raise ValueError(
        f"corpus {paths.corpus!r} ({paths.spec.label}) has no {provider!r} season"
    )


_ESCAPED_UNICODE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _fix_double_encoded(name: object) -> object:
    r"""Repair names in the Wyscout dump that carry literal ``\uXXXX`` text.

    The published players.json was serialised with the escapes already stringified, so
    'Pjanić' arrives as the 12 characters 'Pjanić'. Substituting the escapes directly
    is safer than `unicode_escape`, which would also mangle genuine non-ASCII bytes
    elsewhere in the same string.
    """
    if not isinstance(name, str):
        return name
    return _ESCAPED_UNICODE.sub(lambda m: chr(int(m.group(1), 16)), name)


def wyscout_players(paths: Paths) -> pd.DataFrame:
    """Season-level player directory from Wyscout's players.json.

    Wyscout has no per-match position, so `position_name_24` is left null -- the 24-class
    signal genuinely does not exist for this season, and filling it would be fabrication.
    """
    players = pd.DataFrame(load_json(paths.raw_wyscout / "players.json"))
    for column in ("shortName", "firstName", "lastName"):
        if column in players.columns:
            players[column] = players[column].map(_fix_double_encoded)
    players["coarse_role"] = [
        wyscout_to_coarse((role or {}).get("code2"), strict=False)
        for role in players["role"]
    ]
    frame = players.rename(columns={"wyId": "player_id", "shortName": "player_name"})[
        ["player_id", "player_name", "coarse_role"]
    ].copy()
    frame["position_name_24"] = None
    frame["season"] = _season_key(paths, "wyscout")
    frame["provider"] = "wyscout"
    return frame


def build_player_directory(paths: Paths, actions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Concatenate the corpus's providers' directories, restricted to players who played.

    Wyscout's players.json covers all five European leagues (~3,600 players), so it must be
    filtered down to those appearing in this corpus's actions or the role-label distribution
    is badly skewed.

    Only providers the corpus declares are built. Building Wyscout unconditionally would
    inject ~3,600 Serie A players into the Premier League directory -- they would carry a
    season key no PL action has, so nothing would raise; the directory would just be wrong.
    """
    providers = {season.provider for season in paths.spec.seasons}
    frames = []
    if "statsbomb" in providers:
        frames.append(statsbomb_players(paths))
    if "wyscout" in providers:
        frames.append(wyscout_players(paths))
    if not frames:
        raise ValueError(f"corpus {paths.corpus!r} declares no known provider")

    if actions is not None:
        appeared = (
            actions.groupby(["season", "provider"])["player_id"]
            .apply(lambda s: set(s.dropna().astype(int)))
            .to_dict()
        )
        kept = []
        for frame in frames:
            key = (frame["season"].iloc[0], frame["provider"].iloc[0])
            ids = appeared.get(key)
            kept.append(frame if ids is None else frame[frame["player_id"].astype(int).isin(ids)])
        frames = kept

    directory = pd.concat(frames, ignore_index=True)
    directory["player_id"] = directory["player_id"].astype("int64")

    log.info(
        "player directory: %d rows (%s), roles: %s",
        len(directory),
        directory.groupby("provider").size().to_dict(),
        directory["coarse_role"].value_counts(dropna=False).to_dict(),
    )
    return directory


def load_player_directory(paths: Paths) -> pd.DataFrame:
    path = paths.spadl / "players.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python scripts/build_networks.py` first"
        )
    return pd.read_parquet(path)
