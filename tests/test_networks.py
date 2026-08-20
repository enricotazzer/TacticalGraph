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
    build_match_networks,
    build_team_network,
    networks_to_frames,
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


# --------------------------------------------------------------------------------------
# Direction features
#
# These exist because every TOPOLOGY_FEATURE is a volume measure, and volume is positional:
# midfielders take 84% of the top 50 by degree against a 31% population share. A target man and
# a deep recycler can share a degree while differing entirely in the *kind* of pass they touch.
# --------------------------------------------------------------------------------------


def _direction_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keeper hits long forward passes; a forward receives them and lays the ball back."""
    keys = {"game_id": 1, "team_id": 1, "season": "2015-2016", "provider": "statsbomb"}
    nodes = pd.DataFrame([
        {**keys, "player_id": pid, "mean_x": x, "mean_y": 34.0, "spread_x": 1.0,
         "spread_y": 1.0, "touches": 10, "actions_all_types": 10,
         "passes_attempted": 10, "passes_completed": 8}
        for pid, x in ((1, 5.0), (2, 50.0), (3, 90.0))
    ])
    edges = pd.DataFrame([
        # keeper -> midfielder: long and very forward
        {**keys, "source": 1, "target": 2, "weight": 10, "mean_length": 40.0, "mean_dx": 35.0},
        # midfielder -> forward: forward, shorter
        {**keys, "source": 2, "target": 3, "weight": 10, "mean_length": 20.0, "mean_dx": 15.0},
        # forward -> midfielder: backward lay-off
        {**keys, "source": 3, "target": 2, "weight": 10, "mean_length": 10.0, "mean_dx": -12.0},
    ])
    return nodes, edges


def test_direction_features_separate_senders_from_receivers():
    from tacticalgraph.models.role_gnn import engineer_node_features

    nodes, edges = _direction_fixture()
    out = engineer_node_features(nodes, edges).set_index("player_id")

    # The keeper makes the most progressive passes; the forward makes the least (negative).
    assert out.loc[1, "progression_made"] == pytest.approx(35.0)
    assert out.loc[3, "progression_made"] == pytest.approx(-12.0)
    # The forward *receives* the most forward ball -- invisible to any volume metric, since all
    # three players have identical degree and weight here.
    assert out.loc[3, "progression_received"] == pytest.approx(15.0)
    assert out.loc[1, "progression_received"] == pytest.approx(0.0), "keeper receives nothing"
    # Degree is identical for players 1 and 3, so volume genuinely cannot tell them apart.
    assert out.loc[1, "degree_out_norm"] == out.loc[3, "degree_out_norm"]


def test_progressive_share_is_a_share_and_pass_length_is_weighted():
    from tacticalgraph.models.role_gnn import engineer_node_features

    nodes, edges = _direction_fixture()
    out = engineer_node_features(nodes, edges).set_index("player_id")

    assert out["progressive_share"].between(0.0, 1.0).all()
    assert out.loc[1, "progressive_share"] == pytest.approx(1.0), "35 m is progressive"
    assert out.loc[3, "progressive_share"] == pytest.approx(0.0), "-12 m is not"
    assert out.loc[1, "pass_length_made"] == pytest.approx(40.0)


def test_weighted_means_respect_edge_weight():
    """A weighted mean, not a mean of means -- one heavy edge must dominate a light one."""
    from tacticalgraph.models.role_gnn import engineer_node_features

    keys = {"game_id": 1, "team_id": 1, "season": "2015-2016", "provider": "statsbomb"}
    nodes = pd.DataFrame([
        {**keys, "player_id": pid, "mean_x": 50.0, "mean_y": 34.0, "spread_x": 1.0,
         "spread_y": 1.0, "touches": 1, "actions_all_types": 1,
         "passes_attempted": 1, "passes_completed": 1}
        for pid in (1, 2, 3)
    ])
    edges = pd.DataFrame([
        {**keys, "source": 1, "target": 2, "weight": 9, "mean_length": 10.0, "mean_dx": 10.0},
        {**keys, "source": 1, "target": 3, "weight": 1, "mean_length": 10.0, "mean_dx": 0.0},
    ])
    out = engineer_node_features(nodes, edges).set_index("player_id")
    assert out.loc[1, "progression_made"] == pytest.approx(9.0)  # not 5.0


def test_players_without_edges_get_zero_not_nan():
    """Zero reads as 'no evidence of direction'; a NaN would poison the scaler."""
    import numpy as np

    from tacticalgraph.models.role_gnn import DIRECTION_FEATURES, engineer_node_features

    nodes, edges = _direction_fixture()
    isolated = nodes.iloc[[0]].copy()
    isolated["player_id"] = 99
    nodes = pd.concat([nodes, isolated], ignore_index=True)

    out = engineer_node_features(nodes, edges).set_index("player_id")
    assert not out[list(DIRECTION_FEATURES)].isna().any().any()
    assert np.allclose(out.loc[99, list(DIRECTION_FEATURES)].to_numpy(dtype=float), 0.0)


def test_empty_edge_table_still_returns_direction_columns():
    from tacticalgraph.models.role_gnn import DIRECTION_FEATURES, engineer_node_features

    nodes, edges = _direction_fixture()
    out = engineer_node_features(nodes, edges.iloc[0:0])
    for column in DIRECTION_FEATURES:
        assert column in out.columns
        assert (out[column] == 0.0).all()


# --------------------------------------------------------------------------------------
# xT-weighted edges, shot-chain involvement, and the leak guards on both
# --------------------------------------------------------------------------------------


def _threat_windows() -> tuple[pd.DataFrame, pd.Series]:
    """Two windows of one team-match, with a shot in each.

    `window_index` is assigned directly rather than derived from `window_bounds`, because the
    real windows overlap on a 5-minute stride and an action belongs to up to three of them.
    What is under test here is the *grouping* semantics -- whether a window's features are
    computed from that window's actions alone -- and non-overlapping windows isolate that.
    """
    rows = [
        # window 0: 10 -> 20 twice, and a shot ending possession 0
        (0, 10, 20, "pass", 0),
        (0, 10, 20, "pass", 0),
        (0, 20, None, "shot", 0),
        # window 1: the SAME 10 -> 20 lane again, and a shot ending possession 1. Reusing the
        # lane is what makes the negative control below possible: with match-level keys the two
        # windows collapse into one edge, so window 0's weight absorbs window 1's xT.
        (1, 10, 20, "pass", 1),
        (1, 30, None, "shot", 1),
    ]
    frame = pd.DataFrame(
        rows, columns=["window_index", "player_id", "recipient_id", "type_name", "possession_id"]
    )
    frame["action_id"] = range(len(frame))
    frame["game_id"] = 1
    frame["team_id"] = 1
    frame["season"] = "2015-2016"
    frame["provider"] = "statsbomb"
    frame["result_name"] = "success"
    frame["recipient_confident"] = frame["recipient_id"].notna()
    # Window 0's passes are worth ten times window 1's, so a match-level aggregate is obvious.
    xt = pd.Series([0.10, 0.10, 0.0, 0.01, 0.0], index=frame.index)
    return frame, xt


def test_xt_edge_weights_reproduce_the_persisted_pass_counts():
    """The guard that makes the join safe: same predicate, same grouping, same counts.

    An xT weight on the wrong edge is invisible -- a plausible number in a plausible row -- so
    the two edge tables have to agree by construction rather than by coincidence.
    """
    from tacticalgraph.features.xthreat import attach_xt_edge_weights

    actions = fixture_actions()
    _, edges = networks_to_frames(build_match_networks(actions))
    xt = pd.Series(0.05, index=actions.index)

    merged = attach_xt_edge_weights(edges, actions, xt)
    assert len(merged) == len(edges)
    assert (merged["weight"].to_numpy() == edges["weight"].to_numpy()).all()
    # A flat 0.05 per pass makes xt_weight a direct multiple of the pass count.
    assert merged["xt_weight"].to_numpy() == pytest.approx(0.05 * edges["weight"].to_numpy())


def test_attach_xt_edge_weights_raises_when_the_edge_set_drifts():
    """Both failure modes the assertion exists to catch, exercised separately."""
    from tacticalgraph.features.xthreat import attach_xt_edge_weights

    actions = fixture_actions()
    _, edges = networks_to_frames(build_match_networks(actions))
    xt = pd.Series(0.05, index=actions.index)

    stray = edges.iloc[[0]].copy()
    stray["target"] = 999
    with pytest.raises(ValueError, match="no recomputed counterpart"):
        attach_xt_edge_weights(pd.concat([edges, stray], ignore_index=True), actions, xt)

    inflated = edges.copy()
    inflated.loc[0, "weight"] = int(inflated.loc[0, "weight"]) + 1
    with pytest.raises(ValueError, match="disagree on pass count"):
        attach_xt_edge_weights(inflated, actions, xt)


def test_xt_edge_weights_ignore_negative_deltas():
    """A lane used only for back-passes is worth 0, not a negative number."""
    from tacticalgraph.features.xthreat import attach_xt_edge_weights

    actions = fixture_actions()
    _, edges = networks_to_frames(build_match_networks(actions))
    merged = attach_xt_edge_weights(edges, actions, pd.Series(-0.5, index=actions.index))
    assert (merged["xt_weight"] == 0.0).all()


def test_shot_chain_involvement_is_a_share_of_the_teams_shot_chains():
    from tacticalgraph.features.chains import shot_chain_involvement

    actions, _ = _threat_windows()
    out = shot_chain_involvement(actions).set_index("player_id")["shot_involvement"]
    # Both possessions end in a shot. 10 passes in both, 20 shoots in one, 30 shoots in the other.
    assert out.loc[10] == pytest.approx(1.0)
    assert out.loc[20] == pytest.approx(0.5)
    assert out.loc[30] == pytest.approx(0.5)
    assert out.between(0.0, 1.0).all()


def test_threat_features_are_confined_to_their_window():
    """Rebuild from truncated history and assert nothing changed.

    The same shape as the state-table and node-feature truncation tests, applied to the two new
    features before either reaches a model. `test_match_level_keys_leak_the_future_into_a_window`
    below is its negative control.
    """
    from tacticalgraph.features.chains import shot_chain_involvement
    from tacticalgraph.features.xthreat import xt_edge_weights
    from tacticalgraph.models.role_gnn import WINDOW_KEYS

    actions, xt = _threat_windows()
    truncated = actions[actions["window_index"] <= 0]

    def window_zero(frame: pd.DataFrame) -> pd.DataFrame:
        columns = sorted(frame.columns)
        return (
            frame[frame["window_index"] == 0][columns]
            .sort_values(columns)
            .reset_index(drop=True)
        )

    pd.testing.assert_frame_equal(
        window_zero(xt_edge_weights(actions, xt, group_keys=WINDOW_KEYS)),
        window_zero(xt_edge_weights(truncated, xt.loc[truncated.index], group_keys=WINDOW_KEYS)),
    )
    pd.testing.assert_frame_equal(
        window_zero(shot_chain_involvement(actions, group_keys=WINDOW_KEYS)),
        window_zero(shot_chain_involvement(truncated, group_keys=WINDOW_KEYS)),
    )

    # And the value itself is the window's, not the match's: window 0 saw 0.20 of xT on the
    # 10 -> 20 lane, and the 0.01 added in window 1 must not reach it.
    windowed = xt_edge_weights(actions, xt, group_keys=WINDOW_KEYS)
    assert windowed.set_index(["window_index", "source", "target"]).loc[
        (0, 10, 20), "xt_weight"
    ] == pytest.approx(0.20)


def test_match_level_keys_leak_the_future_into_a_window():
    """The negative control: the test above must be able to fail on the bug it guards.

    With match-level keys on windowed actions, window 0's edge weights absorb window 1's xT --
    exactly the leak that put 7 of 10 node features on full-match values in Module 3.
    """
    from tacticalgraph.features.xthreat import xt_edge_weights

    actions, xt = _threat_windows()
    truncated = actions[actions["window_index"] <= 0]

    full = xt_edge_weights(actions, xt).set_index(["source", "target"])["xt_weight"]
    trunc = xt_edge_weights(truncated, xt.loc[truncated.index]).set_index(
        ["source", "target"]
    )["xt_weight"]

    # The 10 -> 20 lane is used in both windows. Grouped without `window_index` its weight is a
    # whole-match total, so knowing only window 0 gives a strictly smaller number: the extra is
    # xT that had not been generated yet.
    assert full.loc[(10, 20)] == pytest.approx(0.21)
    assert trunc.loc[(10, 20)] == pytest.approx(0.20)
    assert full.loc[(10, 20)] > trunc.loc[(10, 20)]


def test_threat_columns_are_absent_rather_than_zero_when_inputs_are_missing():
    """A zero column would produce a plausible-looking ablation row that measures nothing."""
    from tacticalgraph.models.role_gnn import THREAT_FEATURES, engineer_node_features

    nodes, edges = _direction_fixture()
    frame = engineer_node_features(nodes, edges)
    for name in THREAT_FEATURES:
        assert name not in frame.columns


def test_threat_features_appear_when_their_inputs_are_supplied():
    from tacticalgraph.models.role_gnn import THREAT_FEATURES, engineer_node_features

    nodes, edges = _direction_fixture()
    keys = {"game_id": 1, "team_id": 1, "season": "2015-2016", "provider": "statsbomb"}
    edges = edges.assign(xt_weight=[1.0, 3.0, 0.0])
    node_threat = pd.DataFrame([
        {**keys, "player_id": 1, "xt_generated": 0.5, "shot_involvement": 0.0},
        {**keys, "player_id": 2, "xt_generated": 0.5, "shot_involvement": 1.0},
        {**keys, "player_id": 3, "xt_generated": 0.0, "shot_involvement": 1.0},
    ])

    frame = engineer_node_features(nodes, edges, node_threat=node_threat).set_index("player_id")
    for name in THREAT_FEATURES:
        assert name in frame.columns

    # Shares of the team's xT-weighted strength, so each side sums to 1 across the network.
    assert frame["xt_strength_out_norm"].sum() == pytest.approx(1.0)
    assert frame["xt_strength_in_norm"].sum() == pytest.approx(1.0)
    # The forward's only outgoing lane is the backward lay-off, worth 0 xT.
    assert frame.loc[3, "xt_strength_out_norm"] == pytest.approx(0.0)
    # ...but it receives the most valuable lane in the network.
    assert frame.loc[3, "xt_strength_in_norm"] == pytest.approx(0.75)


def test_node_threat_with_duplicate_rows_raises_rather_than_fanning_out():
    """A duplicated (network, player) row would multiply nodes and go unnoticed downstream."""
    from tacticalgraph.models.role_gnn import engineer_node_features

    nodes, edges = _direction_fixture()
    keys = {"game_id": 1, "team_id": 1, "season": "2015-2016", "provider": "statsbomb"}
    node_threat = pd.DataFrame([
        {**keys, "player_id": 1, "xt_generated": 0.5, "shot_involvement": 0.0},
        {**keys, "player_id": 1, "xt_generated": 0.2, "shot_involvement": 0.0},
    ])
    with pytest.raises(ValueError, match="changed the row count"):
        engineer_node_features(nodes, edges, node_threat=node_threat)
