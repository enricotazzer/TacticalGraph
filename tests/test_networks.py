"""Tests for passing-network construction, possession chains and role mapping.

The network fixture is hand-checked: a 4-player team with a known pass pattern, so node and
edge counts can be asserted exactly rather than approximately.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tacticalgraph.data.possession import reconstruct_possessions
from tacticalgraph.data.recipient import infer_recipients
from tacticalgraph.data.roles import (
    COARSE_ROLES,
    STATSBOMB_POSITION_TO_COARSE,
    UnknownPositionError,
    statsbomb_to_coarse,
    wyscout_to_coarse,
)
from tacticalgraph.graphs.passing_network import (
    build_team_network,
    player_minutes_from_actions,
    window_bounds,
)


def fixture_actions() -> pd.DataFrame:
    """Team 1 passes 10 -> 20 -> 30 -> 10 -> 20 -> 30; team 2 makes one clearance.

    Expected team-1 edges: (10,20) weight 2, (20,30) 2, (30,10) 1.

    Every player must act at least twice, far enough apart in time to clear the 20-minute
    eligibility filter -- minutes are estimated from first-to-last action, so a player with
    a single action is credited 0 minutes and filtered out. See
    `test_single_action_player_is_credited_zero_minutes`.
    """
    rows = [
        {"player_id": 10, "team_id": 1, "type_name": "pass"},
        {"player_id": 20, "team_id": 1, "type_name": "pass"},
        {"player_id": 30, "team_id": 1, "type_name": "pass"},
        {"player_id": 10, "team_id": 1, "type_name": "pass"},
        {"player_id": 20, "team_id": 1, "type_name": "pass"},
        {"player_id": 30, "team_id": 1, "type_name": "pass"},
        {"player_id": 99, "team_id": 2, "type_name": "clearance"},
    ]
    frame = pd.DataFrame(rows)
    frame["action_id"] = range(len(frame))
    frame["period_id"] = 1
    # Spread over 50 minutes so every player clears the 20-minute filter.
    frame["time_seconds"] = [0.0, 300.0, 600.0, 1800.0, 2100.0, 2400.0, 2500.0]
    frame["result_name"] = "success"
    frame["start_x"] = [20.0, 40.0, 60.0, 25.0, 45.0, 65.0, 80.0]
    frame["start_y"] = [30.0, 34.0, 40.0, 30.0, 34.0, 40.0, 20.0]
    frame["end_x"] = frame["start_x"] + 10.0
    frame["end_y"] = frame["start_y"]
    frame["game_id"] = 1
    frame["season"] = "2015-2016"
    frame["provider"] = "statsbomb"
    return infer_recipients(frame)


def test_edge_weights_match_the_hand_checked_pattern():
    network = build_team_network(
        fixture_actions(), game_id=1, team_id=1, season="2015-2016", provider="statsbomb"
    )
    edges = network.edges.set_index(["source", "target"])["weight"].to_dict()
    assert edges[(10, 20)] == 2
    assert edges[(20, 30)] == 2
    assert edges[(30, 10)] == 1
    assert network.n_nodes == 3  # opponent excluded


def test_opponent_actions_are_excluded():
    network = build_team_network(
        fixture_actions(), game_id=1, team_id=1, season="2015-2016", provider="statsbomb"
    )
    assert 99 not in set(network.nodes["player_id"])


def test_node_mean_position_is_the_action_mean():
    network = build_team_network(
        fixture_actions(), game_id=1, team_id=1, season="2015-2016", provider="statsbomb"
    )
    node10 = network.nodes.set_index("player_id").loc[10]
    assert node10["mean_x"] == pytest.approx((20.0 + 25.0) / 2)


def test_no_self_loops_survive():
    frame = fixture_actions()
    network = build_team_network(
        frame, game_id=1, team_id=1, season="2015-2016", provider="statsbomb"
    )
    assert not (network.edges["source"] == network.edges["target"]).any()


def test_minute_filter_drops_a_late_substitute():
    frame = fixture_actions()
    late = frame.iloc[[0]].copy()
    late["player_id"] = 77
    late["time_seconds"] = 2600.0
    late["action_id"] = 99
    combined = infer_recipients(pd.concat([frame, late], ignore_index=True))

    strict = build_team_network(
        combined, game_id=1, team_id=1, season="2015-2016", provider="statsbomb", min_minutes=20
    )
    assert 77 not in set(strict.nodes["player_id"])

    loose = build_team_network(
        combined,
        game_id=1,
        team_id=1,
        season="2015-2016",
        provider="statsbomb",
        apply_minute_filter=False,
    )
    assert 77 in set(loose.nodes["player_id"])


def test_window_bounds_are_15min_sliding_on_a_5min_stride():
    bounds = window_bounds()
    assert bounds[0] == (0.0, 15.0)
    assert bounds[1] == (5.0, 20.0)
    assert bounds[-1] == (75.0, 90.0)
    assert len(bounds) == 16


def test_possession_breaks_on_set_piece_and_team_change():
    rows = [
        {"player_id": 10, "team_id": 1, "type_name": "pass"},
        {"player_id": 20, "team_id": 1, "type_name": "pass"},
        {"player_id": 30, "team_id": 1, "type_name": "throw_in"},  # hard restart
        {"player_id": 91, "team_id": 2, "type_name": "pass"},
        {"player_id": 92, "team_id": 2, "type_name": "pass"},
    ]
    frame = pd.DataFrame(rows)
    frame["action_id"] = range(len(frame))
    frame["period_id"] = 1
    frame["time_seconds"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    out = reconstruct_possessions(frame)
    assert out.loc[0, "possession_id"] == out.loc[1, "possession_id"]
    assert out.loc[2, "possession_id"] != out.loc[1, "possession_id"]
    assert out.loc[3, "possession_id"] != out.loc[2, "possession_id"]


def test_possession_absorbs_a_single_contest_touch():
    """A lone opponent clearance between team-1 actions must not split the chain."""
    rows = [
        {"player_id": 10, "team_id": 1, "type_name": "pass"},
        {"player_id": 91, "team_id": 2, "type_name": "clearance"},
        {"player_id": 20, "team_id": 1, "type_name": "pass"},
    ]
    frame = pd.DataFrame(rows)
    frame["action_id"] = range(len(frame))
    frame["period_id"] = 1
    frame["time_seconds"] = [0.0, 1.0, 2.0]
    out = reconstruct_possessions(frame)
    assert out["possession_id"].nunique() == 1


def test_possession_always_breaks_between_periods():
    rows = [
        {"player_id": 10, "team_id": 1, "type_name": "pass", "period_id": 1},
        {"player_id": 20, "team_id": 1, "type_name": "pass", "period_id": 2},
    ]
    frame = pd.DataFrame(rows)
    frame["action_id"] = range(len(frame))
    frame["time_seconds"] = [0.0, 0.0]
    out = reconstruct_possessions(frame)
    assert out["possession_id"].nunique() == 2


def test_single_action_player_is_credited_zero_minutes():
    """Documents a known limitation of the symmetric minutes estimate.

    Minutes are derived from first-to-last action because Wyscout exposes no per-match
    position spells, so estimating them the same way for both providers is what keeps the
    node filter symmetric. The cost: a player with exactly one action is credited 0 minutes.
    That population is late substitutes with a single touch -- exactly what the filter is
    meant to exclude -- but the behaviour should be explicit, not incidental.
    """
    frame = fixture_actions()
    cameo = frame.iloc[[0]].copy()
    cameo["player_id"] = 55
    cameo["time_seconds"] = 2450.0
    cameo["action_id"] = 500
    combined = infer_recipients(pd.concat([frame, cameo], ignore_index=True))

    minutes = player_minutes_from_actions(combined[combined["team_id"] == 1])
    assert minutes.set_index("player_id").loc[55, "minutes_played"] == 0.0


def test_role_mapping_covers_the_statsbomb_vocabulary():
    """StatsBomb publishes 25 positions; 24 occur in Serie A 2015/16."""
    assert len(STATSBOMB_POSITION_TO_COARSE) == 25
    assert set(STATSBOMB_POSITION_TO_COARSE.values()) == set(COARSE_ROLES)


def test_role_mapping_examples():
    assert statsbomb_to_coarse("Right Center Midfield") == "MID"
    assert statsbomb_to_coarse("Left Wing Back") == "DEF"
    assert statsbomb_to_coarse("Secondary Striker") == "FWD"
    assert statsbomb_to_coarse("Goalkeeper") == "GK"
    assert wyscout_to_coarse("MD") == "MID"
    assert wyscout_to_coarse("DF") == "DEF"


def test_unknown_position_raises_rather_than_bucketing_silently():
    with pytest.raises(UnknownPositionError):
        statsbomb_to_coarse("Sweeper Keeper Deluxe")
    assert statsbomb_to_coarse("Sweeper Keeper Deluxe", strict=False) is None


# --------------------------------------------------------------------------------------
# Role-relative centrality
#
# Raw centrality is largely positional: on the Premier League corpus midfielders are 31% of
# players with >=10 matches and 84% of the top 50 by degree, and goalkeepers take 0% of the top
# 50 on every one of the ten metrics. These tests pin the within-role normalisation that makes
# "central for a centre-back" expressible.
# --------------------------------------------------------------------------------------


def _aggregated_fixture() -> pd.DataFrame:
    """Four roles, three players each, with deliberately role-dependent scales."""
    rows = []
    scale = {"GK": 0.01, "DEF": 0.05, "MID": 0.20, "FWD": 0.08}
    for role, base in scale.items():
        for offset, player in enumerate(range(3)):
            rows.append({
                "season": "2015-2016", "provider": "statsbomb",
                "player_id": len(rows), "coarse_role": role,
                "pagerank": base * (1.0 + 0.5 * offset),
                "betweenness": base * (1.0 + 0.25 * offset),
                "n_matches": 20,
            })
    return pd.DataFrame(rows)


def test_role_relative_z_scores_are_centred_within_each_role():
    from tacticalgraph.features.centrality import role_relative_metrics

    frame = role_relative_metrics(_aggregated_fixture(), metrics=("pagerank", "betweenness"))
    for role, group in frame.groupby("coarse_role"):
        assert group["pagerank_z"].mean() == pytest.approx(0.0, abs=1e-9), role
        # The top player in every role must be positive, so each role has a rankable leader --
        # the property raw centrality lacks, where no goalkeeper ever ranks at all.
        assert group["pagerank_z"].max() > 0, role


def test_role_relative_ranking_surfaces_every_role():
    """The point of the change: the top of the table stops being a list of midfielders."""
    from tacticalgraph.features.centrality import role_relative_metrics

    frame = role_relative_metrics(_aggregated_fixture(), metrics=("pagerank",))
    # Three players per role, so the raw top 3 is the ceiling for a single role monopolising it.
    raw_top = set(frame.nlargest(3, "pagerank")["coarse_role"])
    z_top = set(frame.nlargest(4, "pagerank_z")["coarse_role"])

    assert raw_top == {"MID"}, "fixture should reproduce the positional artefact"
    assert len(z_top) == 4, f"role-relative top 4 should span all roles, got {z_top}"


def test_zero_variance_role_scores_zero_not_nan():
    """A NaN would silently drop the player from any downstream sort."""
    import numpy as np

    from tacticalgraph.features.centrality import role_relative_metrics

    frame = _aggregated_fixture()
    frame.loc[frame["coarse_role"] == "GK", "pagerank"] = 0.01  # no spread at all
    out = role_relative_metrics(frame, metrics=("pagerank",))
    gk = out[out["coarse_role"] == "GK"]["pagerank_z"]
    assert not gk.isna().any()
    assert np.allclose(gk.to_numpy(), 0.0)


def test_missing_role_column_raises_rather_than_ranking_globally():
    from tacticalgraph.features.centrality import role_relative_metrics

    frame = _aggregated_fixture().drop(columns=["coarse_role"])
    with pytest.raises(KeyError, match="coarse_role"):
        role_relative_metrics(frame)


# --------------------------------------------------------------------------------------
# Shared xThreat
# --------------------------------------------------------------------------------------


def test_player_threat_is_a_share_of_the_team_total():
    """Shares, not sums: raw totals carry the provider's action-density signature."""
    from tacticalgraph.features.xthreat import player_threat

    actions = pd.DataFrame({
        "game_id": [1] * 4,
        "team_id": [1, 1, 1, 1],
        "season": ["2015-2016"] * 4,
        "provider": ["statsbomb"] * 4,
        "player_id": [10, 10, 20, 30],
    })
    xt = pd.Series([0.10, 0.10, 0.20, 0.0], index=actions.index)

    out = player_threat(actions, xt).set_index("player_id")["xt_generated"]
    assert out.sum() == pytest.approx(1.0)
    assert out.loc[10] == pytest.approx(0.5)
    assert out.loc[20] == pytest.approx(0.5)
    assert out.loc[30] == pytest.approx(0.0)


def test_player_threat_ignores_negative_deltas():
    """Signed sums would let a player cancel their own progressive passes to zero."""
    from tacticalgraph.features.xthreat import player_threat

    actions = pd.DataFrame({
        "game_id": [1, 1], "team_id": [1, 1],
        "season": ["2015-2016"] * 2, "provider": ["statsbomb"] * 2,
        "player_id": [10, 20],
    })
    xt = pd.Series([0.2, -0.2], index=actions.index)
    out = player_threat(actions, xt).set_index("player_id")["xt_generated"]
    assert out.loc[10] == pytest.approx(1.0)
    assert out.loc[20] == pytest.approx(0.0)


def test_fit_xthreat_rejects_an_empty_training_fold():
    """Fitting on everything is the leak this module exists to prevent."""
    from tacticalgraph.features.xthreat import fit_xthreat

    actions = pd.DataFrame({"game_id": [1, 2]})
    with pytest.raises(ValueError, match="non-empty training fold"):
        fit_xthreat(actions, set())
    with pytest.raises(ValueError, match="split was assigned"):
        fit_xthreat(actions, {999})
