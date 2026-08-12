"""Tests for the Module 4 chain layer.

The provider-comparability test is the important one here: without the
`PROVIDER_COMPARABLE_TYPES` filter, chain length differs between the two providers by a
factor of 1.44 purely because StatsBomb logs carries as dribbles, and any clustering would
partly be clustering the data provider.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tacticalgraph.eval.patterns import shot_lift, wilson_interval
from tacticalgraph.features.chains import (
    CHAIN_FEATURES,
    MIN_CHAIN_ACTIONS,
    build_chain_table,
    chain_sequences,
    cluster_profiles,
    zone_of,
)


def make_actions() -> pd.DataFrame:
    """Three chains: a long build-up ending in a shot, a short one, a 2-action one."""
    rows: list[dict] = []

    def add(possession: int, team: int, type_name: str, x: float, y: float, second: float,
            result: str = "success"):
        rows.append(
            {
                "possession_id": possession,
                "team_id": team,
                "type_name": type_name,
                "result_name": result,
                "start_x": x,
                "start_y": y,
                "end_x": x + 8,
                "end_y": y,
                "time_seconds": second,
                "period_id": 1,
                "game_id": 1,
                "season": "2015-2016",
                "provider": "statsbomb",
            }
        )

    # Chain 0: five passes advancing up the pitch, then a shot -> ends_in_shot
    for i in range(5):
        add(0, 1, "pass", 20.0 + 12 * i, 34.0, 10.0 + i)
    add(0, 1, "shot", 90.0, 34.0, 16.0)
    # Chain 1: three lateral passes, no shot
    for i in range(3):
        add(1, 2, "pass", 50.0, 20.0 + 5 * i, 30.0 + i)
    # Chain 2: only two actions -> must be filtered out
    add(2, 1, "pass", 40.0, 34.0, 50.0)
    add(2, 1, "pass", 45.0, 34.0, 51.0)
    # A dribble that must NOT be counted (not provider-comparable)
    add(0, 1, "dribble", 60.0, 34.0, 13.5)

    frame = pd.DataFrame(rows)
    frame["action_id"] = range(len(frame))
    return frame


def test_chains_shorter_than_the_minimum_are_dropped():
    table = build_chain_table(make_actions())
    assert 2 not in set(table["possession_id"]), "a 2-action chain has no sequence structure"
    assert set(table["possession_id"]) == {0, 1}


def test_dribbles_are_excluded_from_the_action_count():
    """Chain 0 has 5 passes + 1 shot + 1 dribble; only the 6 comparable actions count."""
    table = build_chain_table(make_actions()).set_index("possession_id")
    assert table.loc[0, "n_actions"] == 6


def test_ends_in_shot_is_detected():
    table = build_chain_table(make_actions()).set_index("possession_id")
    assert bool(table.loc[0, "ends_in_shot"]) is True
    assert bool(table.loc[1, "ends_in_shot"]) is False


def test_directness_separates_forward_from_lateral():
    table = build_chain_table(make_actions()).set_index("possession_id")
    # Chain 0 advances 20 -> 98 in x; chain 1 stays at x=50 and moves sideways.
    assert table.loc[0, "directness"] > table.loc[1, "directness"]
    assert table.loc[0, "net_dx"] > 0


def test_all_declared_features_are_produced():
    table = build_chain_table(make_actions())
    missing = [c for c in CHAIN_FEATURES if c not in table.columns]
    assert not missing, f"chain table missing declared features: {missing}"
    assert table[list(CHAIN_FEATURES)].notna().all().all()


def test_zone_boundaries():
    assert zone_of(10.0) == "defensive"
    assert zone_of(50.0) == "middle"
    assert zone_of(90.0) == "final"


def test_set_piece_origin_is_flagged():
    actions = make_actions()
    actions.loc[actions.index[0], "type_name"] = "corner_crossed"
    table = build_chain_table(actions).set_index("possession_id")
    assert table.loc[0, "started_with_set_piece"] == 1.0


def test_xt_gain_uses_supplied_values():
    actions = make_actions()
    values = pd.Series(np.ones(len(actions)) * 0.01, index=actions.index)
    table = build_chain_table(actions, xt_values=values).set_index("possession_id")
    # 6 comparable actions x 0.01
    assert table.loc[0, "xt_gain"] == pytest.approx(0.06)


def test_sequences_are_padded_and_masked_by_length():
    actions = make_actions()
    table = build_chain_table(actions)
    tensor, lengths, vocabulary = chain_sequences(actions, table, max_length=8)
    assert tensor.shape == (len(table), 8, len(vocabulary) + 5)
    assert lengths.tolist() == [6, 3]
    # Padding beyond a chain's length must be exactly zero.
    assert np.allclose(tensor[1, 3:], 0.0)


def test_sequences_encode_action_type_as_one_hot():
    actions = make_actions()
    table = build_chain_table(actions)
    tensor, lengths, vocabulary = chain_sequences(actions, table, max_length=8)
    onehot = tensor[0, 0, : len(vocabulary)]
    assert onehot.sum() == pytest.approx(1.0), "exactly one action type per token"


def test_cluster_profiles_name_every_cluster_distinctly():
    actions = make_actions()
    table = build_chain_table(actions)
    labels = np.array([0, 1])
    profile = cluster_profiles(table, labels)
    assert len(profile) == 2
    assert profile["label"].nunique() == 2, "cluster names must discriminate"
    assert set(profile.columns) >= {"cluster", "n_chains", "shot_rate", "label"}


# ----------------------------------------------------------------------- evaluation


def test_wilson_interval_brackets_the_estimate():
    low, high = wilson_interval(10, 100)
    assert low < 0.10 < high
    assert low >= 0.0 and high <= 1.0


def test_wilson_interval_stays_in_range_at_the_extremes():
    """The reason Wilson is used instead of the normal approximation."""
    low, high = wilson_interval(0, 50)
    assert low >= 0.0
    low, high = wilson_interval(50, 50)
    assert high <= 1.0


def test_wilson_interval_narrows_with_more_data():
    narrow = wilson_interval(100, 1000)
    wide = wilson_interval(1, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_shot_lift_compares_against_the_subset_base_rate():
    table = pd.DataFrame(
        {
            "game_id": range(100),
            "ends_in_shot": [True] * 30 + [False] * 70,
        }
    )
    labels = np.array([0] * 50 + [1] * 50)
    lift = shot_lift(table, labels).set_index("cluster")
    # Cluster 0 holds all 30 shots; base rate is 0.30 over the whole subset.
    assert lift.loc[0, "shot_rate"] == pytest.approx(0.6)
    assert lift.loc[1, "shot_rate"] == pytest.approx(0.0)
    assert lift.loc[0, "base_rate"] == pytest.approx(0.3)
    assert bool(lift.loc[0, "differs_from_base"]) is True
