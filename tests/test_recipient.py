"""Tests for pass-recipient inference.

The inference rule underpins every passing network in the project, so its edge cases are
worth pinning down: half boundaries, self-passes, failed passes, and the lookahead limit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tacticalgraph.data.recipient import evaluate_inference, infer_recipients


def make_actions(rows: list[dict]) -> pd.DataFrame:
    """Minimal SPADL-shaped frame with sensible defaults."""
    defaults = {
        "period_id": 1,
        "team_id": 1,
        "result_name": "success",
        "type_name": "pass",
        "start_x": 50.0,
        "start_y": 34.0,
        "end_x": 60.0,
        "end_y": 34.0,
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    frame["action_id"] = range(len(frame))
    frame["time_seconds"] = [float(i) for i in range(len(frame))]
    return frame


def test_simple_pass_chain_resolves_recipients():
    actions = make_actions(
        [
            {"player_id": 10},
            {"player_id": 20},
            {"player_id": 30},
        ]
    )
    out = infer_recipients(actions)
    assert out.loc[0, "recipient_id"] == 20
    assert out.loc[1, "recipient_id"] == 30
    # The final action has no successor, so it must stay unresolved rather than wrap around.
    assert pd.isna(out.loc[2, "recipient_id"])
    assert not out.loc[2, "recipient_confident"]


def test_failed_pass_gets_no_recipient():
    actions = make_actions(
        [
            {"player_id": 10, "result_name": "fail"},
            {"player_id": 99, "team_id": 2},
        ]
    )
    out = infer_recipients(actions)
    assert pd.isna(out.loc[0, "recipient_id"])
    assert not out.loc[0, "recipient_confident"]


def test_opponent_next_action_blocks_resolution():
    """A completed pass followed only by opponent actions must not invent an edge."""
    actions = make_actions(
        [
            {"player_id": 10},
            {"player_id": 91, "team_id": 2},
            {"player_id": 92, "team_id": 2},
            {"player_id": 93, "team_id": 2},
        ]
    )
    out = infer_recipients(actions)
    assert pd.isna(out.loc[0, "recipient_id"])


def test_lookahead_skips_intervening_opponent_touch():
    actions = make_actions(
        [
            {"player_id": 10},
            {"player_id": 91, "team_id": 2, "type_name": "interception"},
            {"player_id": 20},
        ]
    )
    out = infer_recipients(actions)
    assert out.loc[0, "recipient_id"] == 20


def test_lookahead_is_bounded():
    """Beyond the lookahead window the rule must give up, not search the whole match."""
    actions = make_actions(
        [
            {"player_id": 10},
            {"player_id": 91, "team_id": 2},
            {"player_id": 92, "team_id": 2},
            {"player_id": 93, "team_id": 2},
            {"player_id": 20},
        ]
    )
    out = infer_recipients(actions, lookahead=3)
    assert pd.isna(out.loc[0, "recipient_id"])


def test_never_crosses_a_period_boundary():
    """The last pass of a half must not be linked to the first action of the next."""
    actions = make_actions(
        [
            {"player_id": 10, "period_id": 1},
            {"player_id": 20, "period_id": 2},
        ]
    )
    out = infer_recipients(actions)
    assert pd.isna(out.loc[0, "recipient_id"])


def test_self_pass_is_not_a_recipient():
    actions = make_actions(
        [
            {"player_id": 10},
            {"player_id": 10, "type_name": "dribble"},
            {"player_id": 20},
        ]
    )
    out = infer_recipients(actions)
    assert out.loc[0, "recipient_id"] == 20


def test_non_pass_actions_get_no_recipient():
    actions = make_actions(
        [
            {"player_id": 10, "type_name": "clearance"},
            {"player_id": 20},
        ]
    )
    out = infer_recipients(actions)
    assert pd.isna(out.loc[0, "recipient_id"])


def test_empty_frame_is_handled():
    out = infer_recipients(make_actions([]).iloc[0:0])
    assert "recipient_id" in out.columns
    assert out.empty


def test_evaluate_inference_scores_against_truth():
    actions = make_actions([{"player_id": 10}, {"player_id": 20}, {"player_id": 30}])
    out = infer_recipients(actions)
    truth = pd.Series([20, 99, pd.NA], dtype="Int64")  # second one deliberately wrong
    result = evaluate_inference(out, truth)
    assert result["n"] == 2
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["wrong"] == pytest.approx(0.5)
