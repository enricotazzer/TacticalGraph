"""Regression tests for the bugs found while porting Modules 1-4 to a second corpus.

Every bug here had the same shape: code written when there was one corpus hardcoded a Serie A
constant or assumed both providers existed. None of them raised. They produced a table that
looked right and was attributed to the wrong competition — the worst possible failure mode for
a project whose whole claim is that its numbers are trustworthy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tacticalgraph.config import Paths
from tacticalgraph.data.players import _season_key
from tacticalgraph.eval.clustering import half_season_stability


# --------------------------------------------------------------------------------------
# Season keys must come from the corpus, not from a Serie A constant
# --------------------------------------------------------------------------------------


def test_season_key_resolves_per_corpus(tmp_path):
    """Both corpora are provider=statsbomb season=2015-2016, so a hardcoded Serie A key
    would tag Premier League players with a season their actions do not have, breaking every
    (season, provider, player_id) join without raising."""
    serie_a = Paths(root=tmp_path, corpus="serie_a")
    premier = Paths(root=tmp_path, corpus="premier_league")

    assert _season_key(serie_a, "statsbomb") == "2015-2016"
    assert _season_key(premier, "statsbomb") == "2015-2016"
    assert _season_key(serie_a, "wyscout") == "2017-2018"


def test_season_key_rejects_a_provider_the_corpus_lacks(tmp_path):
    premier = Paths(root=tmp_path, corpus="premier_league")
    with pytest.raises(ValueError, match="no 'wyscout' season"):
        _season_key(premier, "wyscout")


def test_premier_league_declares_only_statsbomb(tmp_path):
    """The guard behind `build_player_directory` skipping Wyscout: building it
    unconditionally injected ~3,600 Serie A players into the Premier League directory."""
    premier = Paths(root=tmp_path, corpus="premier_league")
    assert {s.provider for s in premier.spec.seasons} == {"statsbomb"}


# --------------------------------------------------------------------------------------
# Half-season stability (the single-season replacement for cross-season stability)
# --------------------------------------------------------------------------------------


def _games(n_weeks: int = 38, per_week: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": range(1, n_weeks * per_week + 1),
            "game_day": [w for w in range(1, n_weeks + 1) for _ in range(per_week)],
        }
    )


def _meta(n_players: int, games: pd.DataFrame, rows_per_player: int = 12) -> pd.DataFrame:
    """One row per (player, game), spread across both halves of the season."""
    game_ids = games["game_id"].to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for player_id in range(1, n_players + 1):
        # Half the appearances in each half of the season, so every player is comparable.
        first = rng.choice(game_ids[: len(game_ids) // 2], rows_per_player // 2, replace=False)
        second = rng.choice(game_ids[len(game_ids) // 2 :], rows_per_player // 2, replace=False)
        for game_id in np.concatenate([first, second]):
            rows.append({"player_id": player_id, "game_id": int(game_id)})
    return pd.DataFrame(rows)


def test_half_season_stability_detects_a_stable_representation():
    """A per-player constant embedding must score near 1 matched and near 0 shuffled."""
    games = _games()
    meta = _meta(40, games)
    rng = np.random.default_rng(1)
    signatures = rng.normal(size=(41, 8))
    features = np.vstack([signatures[p] for p in meta["player_id"]])

    result = half_season_stability(features, meta, games, label="stable")
    assert result["n_players"] == 40
    assert result["matched_cosine"] > 0.9
    assert result["lift"] > 0.5


def test_half_season_stability_detects_an_unstable_representation():
    """Pure noise must not look stable -- the shuffled baseline is what makes that visible."""
    games = _games()
    meta = _meta(40, games)
    rng = np.random.default_rng(2)
    features = rng.normal(size=(len(meta), 8))

    result = half_season_stability(features, meta, games, label="noise")
    assert result["n_players"] == 40
    assert abs(result["lift"]) < 0.35


def test_half_season_stability_excludes_players_in_only_one_half():
    """January arrivals must be excluded, not counted as unstable: their absence is a
    transfer-window artefact rather than a property of the representation."""
    games = _games()
    game_ids = games["game_id"].to_numpy()
    first_half, second_half = game_ids[:190], game_ids[190:]

    meta = pd.DataFrame(
        [{"player_id": 1, "game_id": int(g)} for g in first_half[:6]]
        + [{"player_id": 1, "game_id": int(g)} for g in second_half[:6]]
        # Player 2 only ever plays in the second half.
        + [{"player_id": 2, "game_id": int(g)} for g in second_half[6:12]]
    )
    features = np.random.default_rng(3).normal(size=(len(meta), 8))

    result = half_season_stability(features, meta, games, label="partial")
    assert result["n_players"] == 1


def test_half_season_stability_needs_a_matchweek_column():
    games = _games().rename(columns={"game_day": "kickoff"})
    meta = _meta(5, _games())
    features = np.zeros((len(meta), 4))
    with pytest.raises(KeyError, match="no matchweek column"):
        half_season_stability(features, meta, games)


def test_team_name_lookup_is_keyed_by_provider(tmp_path):
    """The two providers number teams independently. Keying by team id alone would label one
    club with another's name if the ranges ever overlapped -- silently, with no error."""
    from tacticalgraph.data.spadl_store import team_name_lookup, write_teams

    paths = Paths(root=tmp_path, corpus="serie_a")
    write_teams(
        paths,
        pd.DataFrame(
            [
                {"provider": "statsbomb", "team_id": 7, "team_name": "Statsbomb Seven"},
                {"provider": "wyscout", "team_id": 7, "team_name": "Wyscout Seven"},
            ]
        ),
    )
    lookup = team_name_lookup(paths)
    assert lookup[("statsbomb", 7)] == "Statsbomb Seven"
    assert lookup[("wyscout", 7)] == "Wyscout Seven"


def test_club_labeller_falls_back_and_never_raises(tmp_path):
    """A missing name must degrade a label, not break a figure mid-render."""
    from tacticalgraph.data.aliases import club_labeller

    paths = Paths(root=tmp_path, corpus="premier_league")  # no teams table written
    label = club_labeller(paths)
    assert label("statsbomb", 999999) == "team 999999"


def test_half_season_stability_handles_no_overlap():
    """Empty rather than a crash: a corpus where nobody spans both halves is degenerate,
    not a bug, and the report should say n_players=0."""
    games = _games()
    meta = pd.DataFrame([{"player_id": 1, "game_id": 1}])
    features = np.zeros((1, 4))
    result = half_season_stability(features, meta, games, label="none")
    assert result == {"representation": "none", "n_players": 0}
