"""Canonical action store: partitioned parquet under DATA_ROOT/spadl.

Layout mirrors the analysis splits so a season can be read without touching the other:

    spadl/season=2015-2016/provider=statsbomb/actions.parquet
    spadl/season=2017-2018/provider=wyscout/actions.parquet
    spadl/games.parquet          # match index for both seasons

All directory listings go through `config.clean_glob` because the store lives on an exFAT
volume, where macOS drops `._`-prefixed AppleDouble sidecars next to every real file. Those
are not parquet and will crash a reader that globs naively.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tacticalgraph.config import Paths, clean_glob
from tacticalgraph.data.schema import validate_canonical

log = logging.getLogger(__name__)

ACTIONS_FILENAME = "actions.parquet"
GAMES_FILENAME = "games.parquet"


def partition_dir(paths: Paths, season: str, provider: str) -> Path:
    return paths.spadl / f"season={season}" / f"provider={provider}"


def write_actions(
    paths: Paths, actions: pd.DataFrame, season: str, provider: str
) -> Path:
    """Persist one season's canonical actions."""
    validate_canonical(actions, context=f"{season}/{provider}")
    directory = partition_dir(paths, season, provider)
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / ACTIONS_FILENAME
    actions.to_parquet(dest, index=False)
    log.info("wrote %d actions -> %s", len(actions), dest)
    return dest


def read_actions(
    paths: Paths, season: str | None = None, provider: str | None = None
) -> pd.DataFrame:
    """Read canonical actions, optionally restricted to one partition.

    Partition keys are stored as real columns (not just directory names), so no
    reconstruction from the path is needed.
    """
    season_glob = f"season={season}" if season else "season=*"
    provider_glob = f"provider={provider}" if provider else "provider=*"

    frames = []
    for directory in clean_glob(paths.spadl, f"{season_glob}/{provider_glob}"):
        for file in clean_glob(directory, ACTIONS_FILENAME):
            frames.append(pd.read_parquet(file))

    if not frames:
        raise FileNotFoundError(
            f"no actions under {paths.spadl} matching season={season} provider={provider}; "
            "run `python scripts/build_spadl.py` first"
        )
    return pd.concat(frames, ignore_index=True)


def write_games(paths: Paths, games: pd.DataFrame) -> Path:
    paths.spadl.mkdir(parents=True, exist_ok=True)
    dest = paths.spadl / GAMES_FILENAME
    games.to_parquet(dest, index=False)
    log.info("wrote %d games -> %s", len(games), dest)
    return dest


def read_games(paths: Paths, season: str | None = None) -> pd.DataFrame:
    dest = paths.spadl / GAMES_FILENAME
    if not dest.exists():
        raise FileNotFoundError(
            f"{dest} not found; run `python scripts/build_spadl.py` first"
        )
    games = pd.read_parquet(dest)
    if season:
        games = games[games["season"] == season].reset_index(drop=True)
    return games


TEAMS_FILENAME = "teams.parquet"


def write_teams(paths: Paths, teams: pd.DataFrame) -> Path:
    """Persist (provider, season, team_id, team_name) for the corpus.

    The hand-maintained alias table in `data.aliases` exists only to reconcile StatsBomb
    team ids with Wyscout ones for Serie A. A single-provider corpus needs no reconciliation,
    just the provider's own names -- so read them from the loader once and store them, rather
    than extending a hardcoded table per competition.
    """
    paths.spadl.mkdir(parents=True, exist_ok=True)
    dest = paths.spadl / TEAMS_FILENAME
    teams.to_parquet(dest, index=False)
    log.info("wrote %d teams -> %s", len(teams), dest)
    return dest


def read_teams(paths: Paths) -> pd.DataFrame:
    dest = paths.spadl / TEAMS_FILENAME
    if not dest.exists():
        raise FileNotFoundError(
            f"{dest} not found; run `python scripts/build_spadl.py --corpus {paths.corpus}`"
        )
    return pd.read_parquet(dest)


def team_name_lookup(paths: Paths) -> dict[tuple[str, int], str]:
    """(provider, team_id) -> display name, empty if the corpus has no teams table yet.

    Keyed by provider, not team id alone: the two providers number teams independently
    (StatsBomb Serie A starts at 224, Wyscout at 3157). They happen not to collide today, but
    a collision would silently label one club with another's name rather than raise.

    Returns empty rather than raising so presentation code can fall back to the alias table
    (Serie A, built before this store existed) without a try/except at every call site.
    """
    try:
        teams = read_teams(paths)
    except FileNotFoundError:
        return {}
    return {
        (str(r.provider), int(r.team_id)): str(r.team_name)
        for r in teams.itertuples(index=False)
    }


def store_summary(paths: Paths) -> pd.DataFrame:
    """Quick inventory of what has been built, for logging and the README."""
    rows = []
    for directory in clean_glob(paths.spadl, "season=*/provider=*"):
        file = directory / ACTIONS_FILENAME
        if not file.exists():
            continue
        frame = pd.read_parquet(file, columns=["game_id", "type_name"])
        season = directory.parent.name.split("=", 1)[1]
        provider = directory.name.split("=", 1)[1]
        rows.append(
            {
                "season": season,
                "provider": provider,
                "games": frame["game_id"].nunique(),
                "actions": len(frame),
                "actions_per_game": round(len(frame) / max(frame["game_id"].nunique(), 1), 1),
                "size_mb": round(file.stat().st_size / 1e6, 1),
            }
        )
    return pd.DataFrame(rows)
