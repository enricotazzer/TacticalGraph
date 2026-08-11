"""Tests for temporal splitting and the leakage guards.

The most valuable test in the suite is `test_random_split_is_rejected`: a random split over
per-player rows is the single failure that would silently inflate every metric in the
project, and the guard exists specifically to make it loud.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tacticalgraph.config import SERIE_A_STATSBOMB, SERIE_A_WYSCOUT
from tacticalgraph.data.schema import LeakageError as SchemaLeakageError
from tacticalgraph.data.schema import assert_no_enrichment_leakage
from tacticalgraph.eval.splits import (
    LeakageError,
    Split,
    assert_no_overlap,
    reject_random_split,
    temporal_split,
)


def make_games() -> pd.DataFrame:
    """38 matchweeks x 10 games for both seasons."""
    rows = []
    game_id = 1000
    for season in (SERIE_A_STATSBOMB.key, SERIE_A_WYSCOUT.key):
        provider = "statsbomb" if season == SERIE_A_STATSBOMB.key else "wyscout"
        for week in range(1, 39):
            for _ in range(10):
                rows.append(
                    {"game_id": game_id, "season": season, "provider": provider, "game_day": week}
                )
                game_id += 1
    return pd.DataFrame(rows)


def test_cross_season_split_puts_second_season_in_test():
    split = temporal_split(make_games(), kind="cross_season")
    assert len(split.train) == 300  # weeks 1-30
    assert len(split.val) == 80  # weeks 31-38
    assert len(split.test) == 380  # all of 2017/18
    assert not split.train & split.test


def test_within_season_split_is_single_provider_and_ordered():
    games = make_games()
    split = temporal_split(games, kind="within_season")
    weeks = games.set_index("game_id")["game_day"]
    seasons = games.set_index("game_id")["season"]

    # Test fold must be strictly later than train, and stay inside 2015/16.
    assert max(weeks[list(split.train)]) < min(weeks[list(split.test)])
    assert set(seasons[list(split.test)]) == {SERIE_A_STATSBOMB.key}


def test_overlapping_folds_are_rejected():
    bad = Split(name="bad", train={1, 2}, val={2, 3}, test={4}, description="")
    with pytest.raises(LeakageError, match="both train and val"):
        assert_no_overlap(bad)


def test_random_split_is_rejected():
    """The core guard: a match must never straddle two folds.

    Simulates the classic mistake -- shuffling per-player rows instead of per-match ones.
    """
    game_ids = pd.Series([1, 1, 1, 2, 2, 2])
    fold = pd.Series(["train", "test", "train", "test", "train", "test"])
    with pytest.raises(LeakageError, match="more than one fold"):
        reject_random_split(fold, game_ids)


def test_clean_split_passes_the_guard():
    game_ids = pd.Series([1, 1, 1, 2, 2, 2])
    fold = pd.Series(["train", "train", "train", "test", "test", "test"])
    reject_random_split(fold, game_ids)  # must not raise


def test_unused_rows_do_not_trigger_the_guard():
    game_ids = pd.Series([1, 1, 2, 2])
    fold = pd.Series(["train", "unused", "test", "unused"])
    reject_random_split(fold, game_ids)


def test_missing_matchweek_column_is_a_clear_error():
    games = make_games().drop(columns=["game_day"])
    with pytest.raises(KeyError, match="no matchweek column"):
        temporal_split(games)


def test_enrichment_columns_are_blocked_from_model_inputs():
    """StatsBomb-only fields must not reach a feature matrix."""
    frame = pd.DataFrame({"mean_x": [1.0], "statsbomb_xg": [0.2]})
    with pytest.raises(SchemaLeakageError, match="statsbomb_xg"):
        assert_no_enrichment_leakage(frame, context="test features")


def test_clean_feature_frame_passes_leakage_check():
    assert_no_enrichment_leakage(pd.DataFrame({"mean_x": [1.0], "touches": [3]}))
