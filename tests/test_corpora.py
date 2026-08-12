"""Multi-corpus isolation and the matchweek split.

Two competitions in this project share a provider *and* a season key: Serie A 2015/16 and
Premier League 2015/16 are both `statsbomb` / `2015-2016`. Every test here exists because a
collision between them would not raise -- it would silently merge two competitions into one
partition and produce plausible-looking numbers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tacticalgraph.config import (
    ALL_SPLIT_KINDS,
    CORPORA,
    PREMIER_LEAGUE_CORPUS,
    SERIE_A_CORPUS,
    Paths,
)
from tacticalgraph.data.adapters import bindings_for, get_binding
from tacticalgraph.eval.splits import reject_random_split, temporal_split


# --------------------------------------------------------------------------------------
# Path isolation
# --------------------------------------------------------------------------------------


def test_derived_paths_differ_between_corpora(tmp_path):
    """The whole point of the corpus namespace: no derived path may be shared."""
    a = Paths(root=tmp_path, corpus="serie_a")
    b = Paths(root=tmp_path, corpus="premier_league")
    for attribute in ("spadl", "networks", "models", "reports", "figures", "enrichment"):
        assert getattr(a, attribute) != getattr(b, attribute), attribute


def test_raw_cache_is_shared_between_corpora(tmp_path):
    """StatsBomb events are keyed by globally unique match id, so sharing is safe --
    and deliberate, because duplicating it would cost 1.4 GB."""
    a = Paths(root=tmp_path, corpus="serie_a")
    b = Paths(root=tmp_path, corpus="premier_league")
    assert a.raw_statsbomb == b.raw_statsbomb
    assert a.raw_wyscout == b.raw_wyscout


def test_unknown_corpus_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown corpus"):
        Paths.load("bundesliga")


# --------------------------------------------------------------------------------------
# Provider bindings
# --------------------------------------------------------------------------------------


def test_statsbomb_binding_points_at_different_competitions():
    """The regression this guards: one global BINDINGS dict would convert Serie A events
    into the Premier League partition, since both are provider=statsbomb season=2015-2016."""
    serie_a = get_binding("statsbomb", "serie_a")
    premier = get_binding("statsbomb", "premier_league")

    assert serie_a.competition_id == 12
    assert premier.competition_id == 2
    # Same season *id* and same season *key* -- competition_id is the only thing separating
    # them, which is exactly why the corpus namespace exists.
    assert serie_a.season_id == premier.season_id == 27
    assert serie_a.spec.key == premier.spec.key == "2015-2016"
    assert serie_a.spec.label != premier.spec.label


def test_premier_league_has_no_wyscout_provider():
    assert sorted(bindings_for("premier_league")) == ["statsbomb"]
    with pytest.raises(ValueError, match="no provider 'wyscout'"):
        get_binding("wyscout", "premier_league")


def test_serie_a_has_both_providers():
    assert sorted(bindings_for("serie_a")) == ["statsbomb", "wyscout"]


def test_every_corpus_declares_matching_statsbomb_ids():
    """A corpus with two StatsBomb seasons but one id pair would bind the wrong season."""
    for slug in CORPORA:
        bindings_for(slug)  # raises if the counts disagree


# --------------------------------------------------------------------------------------
# The matchweek split
# --------------------------------------------------------------------------------------


def _premier_league_games(n_weeks: int = 38, per_week: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": range(1, n_weeks * per_week + 1),
            "game_day": [w for w in range(1, n_weeks + 1) for _ in range(per_week)],
            "season": "2015-2016",
            "provider": "statsbomb",
        }
    )


def test_matchweek_split_sizes():
    split = temporal_split(
        _premier_league_games(), kind="matchweek", corpus="premier_league"
    )
    assert (len(split.train), len(split.val), len(split.test)) == (260, 70, 50)
    assert len(split.train | split.val | split.test) == 380


def test_matchweek_split_is_chronological():
    """Every test match must fall strictly later than every training match, or the split
    leaks the future in exactly the way a random split would."""
    games = _premier_league_games()
    split = temporal_split(games, kind="matchweek", corpus="premier_league")
    week = games.set_index("game_id")["game_day"]
    assert week[list(split.train)].max() < week[list(split.val)].min()
    assert week[list(split.val)].max() < week[list(split.test)].min()


def test_cross_season_split_rejected_on_single_season_corpus():
    """Without this guard the folds come back empty and the run looks like a success."""
    games = _premier_league_games()
    with pytest.raises(ValueError, match="does not support split kind"):
        temporal_split(games, kind="cross_season", corpus="premier_league")


def test_matchweek_split_rejected_on_serie_a():
    with pytest.raises(ValueError, match="does not support split kind"):
        temporal_split(_premier_league_games(), kind="matchweek", corpus="serie_a")


def test_matchweek_split_needs_the_declared_season():
    games = _premier_league_games()
    games["season"] = "2019-2020"
    with pytest.raises(ValueError, match="no games for season"):
        temporal_split(games, kind="matchweek", corpus="premier_league")


def test_matchweek_split_passes_the_random_split_guard():
    """A match must land in exactly one fold even at per-row granularity."""
    games = _premier_league_games()
    split = temporal_split(games, kind="matchweek", corpus="premier_league")
    # 16 checkpoint rows per match, the Module 3 granularity.
    rows = games.loc[games.index.repeat(16)].reset_index(drop=True)
    fold = split.assign(rows["game_id"])
    reject_random_split(fold, rows["game_id"])
    assert set(fold.unique()) == {"train", "val", "test"}


def test_split_kinds_registry_covers_argparse_choices():
    """`ALL_SPLIT_KINDS` feeds --split; a corpus kind missing from it is unreachable."""
    declared = {k for spec in CORPORA.values() for k in spec.split_kinds}
    assert declared == set(ALL_SPLIT_KINDS)


def test_corpus_split_kinds_are_what_the_modules_expect():
    assert SERIE_A_CORPUS.split_kinds == ("cross_season", "within_season")
    assert PREMIER_LEAGUE_CORPUS.split_kinds == ("matchweek",)
    assert PREMIER_LEAGUE_CORPUS.n_matches_expected == 380
    assert SERIE_A_CORPUS.n_matches_expected == 760
