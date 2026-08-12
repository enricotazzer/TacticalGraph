"""Provider adapters: raw provider data -> canonical SPADL actions.

`socceraction` gives us two things that make this layer thin on purpose:

* `EventDataLoader` subclasses (`StatsBombLoader`, `PublicWyscoutLoader`) that expose the
  *same* interface (`games`, `teams`, `players`, `events`) over very different raw formats;
* `spadl.{statsbomb,wyscout}.convert_to_actions`, the official converters into SPADL.

So there is one conversion routine here, parameterised by provider, rather than two
parallel implementations that could drift apart. Drift is the main risk in this project:
2015/16 comes from StatsBomb and 2017/18 from Wyscout, and any asymmetry we introduce
ourselves would be indistinguishable from a real change in how Serie A was played.

On top of SPADL we add, identically for both providers:
  * `play_left_to_right` normalisation (every team attacks +x),
  * inferred pass recipients (`recipient.py`),
  * reconstructed possession chains (`possession.py`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader
from socceraction.data.wyscout import PublicWyscoutLoader
from socceraction.spadl import statsbomb as statsbomb_spadl
from socceraction.spadl import wyscout as wyscout_spadl

from tacticalgraph.config import (
    CORPORA,
    DEFAULT_CORPUS,
    Paths,
    SeasonSpec,
)
from tacticalgraph.data.possession import reconstruct_possessions
from tacticalgraph.data.recipient import infer_recipients

log = logging.getLogger(__name__)

# Wyscout public dataset identifiers for Serie A 2017/18 (from PublicWyscoutLoader._index).
WYSCOUT_COMPETITION_ID = 524
WYSCOUT_SEASON_ID = 181248


@dataclass(frozen=True)
class ProviderBinding:
    """Everything provider-specific, in one place."""

    spec: SeasonSpec
    competition_id: int
    season_id: int
    make_loader: Callable[[Paths], Any]
    convert: Callable[..., pd.DataFrame]


def _statsbomb_loader(paths: Paths) -> StatsBombLoader:
    return StatsBombLoader(getter="local", root=str(paths.raw_statsbomb))


def _wyscout_loader(paths: Paths) -> PublicWyscoutLoader:
    # download=False: our own downloader already placed the Italy members and the
    # players/teams reference tables. Letting the loader download would pull all seven
    # competitions (~950 MB unpacked) when we need one.
    return PublicWyscoutLoader(root=str(paths.raw_wyscout), download=False)


def bindings_for(corpus: str = DEFAULT_CORPUS) -> dict[str, ProviderBinding]:
    """Provider bindings for one corpus.

    Built per corpus rather than as a module constant because the same provider serves
    different competitions: `statsbomb` means (competition 12, season 27) for Serie A and
    (competition 2, season 27) for the Premier League. A single global mapping would quietly
    convert the wrong competition into the right-looking partition.
    """
    spec = CORPORA.get(corpus)
    if spec is None:
        raise ValueError(f"unknown corpus {corpus!r}; known: {sorted(CORPORA)}")

    out: dict[str, ProviderBinding] = {}
    statsbomb_seasons = [s for s in spec.seasons if s.provider == "statsbomb"]
    if statsbomb_seasons:
        if len(statsbomb_seasons) != len(spec.statsbomb_ids):
            raise ValueError(
                f"corpus {corpus!r}: {len(statsbomb_seasons)} StatsBomb season(s) but "
                f"{len(spec.statsbomb_ids)} id pair(s) -- these must correspond"
            )
        competition_id, season_id = spec.statsbomb_ids[0]
        out["statsbomb"] = ProviderBinding(
            spec=statsbomb_seasons[0],
            competition_id=competition_id,
            season_id=season_id,
            make_loader=_statsbomb_loader,
            convert=statsbomb_spadl.convert_to_actions,
        )
    wyscout_seasons = [s for s in spec.seasons if s.provider == "wyscout"]
    if wyscout_seasons:
        out["wyscout"] = ProviderBinding(
            spec=wyscout_seasons[0],
            competition_id=WYSCOUT_COMPETITION_ID,
            season_id=WYSCOUT_SEASON_ID,
            make_loader=_wyscout_loader,
            convert=wyscout_spadl.convert_to_actions,
        )
    return out


def get_binding(provider: str, corpus: str = DEFAULT_CORPUS) -> ProviderBinding:
    bindings = bindings_for(corpus)
    try:
        return bindings[provider]
    except KeyError:
        raise ValueError(
            f"corpus {corpus!r} has no provider {provider!r}; "
            f"it has {sorted(bindings)}"
        ) from None


def load_games(paths: Paths, provider: str) -> pd.DataFrame:
    """Match index for a provider's season, chronologically ordered."""
    binding = get_binding(provider, paths.corpus)
    loader = binding.make_loader(paths)
    games = loader.games(binding.competition_id, binding.season_id)
    sort_keys = [k for k in ("game_date", "game_day") if k in games.columns]
    if sort_keys:
        games = games.sort_values(sort_keys).reset_index(drop=True)
    return games


def convert_game(
    paths: Paths,
    provider: str,
    game: pd.Series,
    loader: Any | None = None,
) -> pd.DataFrame:
    """Convert one game to canonical actions.

    Returns a frame with SPADL columns plus `type_name`/`result_name`, an inferred
    `recipient_id`, a reconstructed `possession_id`, and season/provider partition keys.
    """
    binding = get_binding(provider, paths.corpus)
    loader = loader or binding.make_loader(paths)
    game_id = int(game["game_id"])

    events = loader.events(game_id)
    actions = binding.convert(events, int(game["home_team_id"]))

    # Normalise direction of play before anything spatial is derived from coordinates.
    actions = spadl.play_left_to_right(actions, int(game["home_team_id"]))
    actions = spadl.add_names(actions)

    actions = actions.reset_index(drop=True)
    actions["action_id"] = actions.index.astype("int64")
    actions["season"] = binding.spec.key
    actions["provider"] = provider

    actions = infer_recipients(actions)
    actions = reconstruct_possessions(actions)
    return actions


def team_names(paths: Paths, provider: str) -> pd.DataFrame:
    """Distinct (team_id, team_name) for a season, for the alias table and plots."""
    binding = get_binding(provider, paths.corpus)
    loader = binding.make_loader(paths)
    games = load_games(paths, provider)
    frames = []
    for game_id in games["game_id"].astype(int):
        try:
            frames.append(loader.teams(game_id))
        except Exception as exc:  # noqa: BLE001
            log.debug("teams(%s) failed: %s", game_id, exc)
    if not frames:
        return pd.DataFrame(columns=["team_id", "team_name"])
    teams = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["team_id"])
    teams["provider"] = provider
    teams["season"] = binding.spec.key
    return teams.reset_index(drop=True)


def player_frame(paths: Paths, provider: str) -> pd.DataFrame:
    """Per game-player rows: minutes played, team, nominal position.

    This is where the two providers differ most and where the harmonisation cost is
    concentrated -- see `roles.py` for how the position vocabularies are reconciled.
    """
    binding = get_binding(provider, paths.corpus)
    loader = binding.make_loader(paths)
    games = load_games(paths, provider)
    frames = []
    for game_id in games["game_id"].astype(int):
        try:
            players = loader.players(game_id)
            players["game_id"] = game_id
            frames.append(players)
        except Exception as exc:  # noqa: BLE001
            log.debug("players(%s) failed: %s", game_id, exc)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["provider"] = provider
    out["season"] = binding.spec.key
    return out
