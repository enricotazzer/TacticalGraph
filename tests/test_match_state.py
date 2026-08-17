"""Tests for the Module 3 feature layer.

`test_no_future_leakage_at_any_checkpoint` is the most important test in the project. Every
other Module 3 result is meaningless if it fails, and a leak would not raise an error --
it would simply make the metrics look excellent.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tacticalgraph.features.match_state import (
    B0_FEATURES,
    B1_FEATURES,
    B2_FEATURES,
    FEATURE_LADDER,
    OUTCOME_TO_INDEX,
    action_minutes,
    build_state_table,
    checkpoints,
    derive_goals,
    match_outcomes,
)
from tacticalgraph.graphs.passing_network import window_bounds

HOME, AWAY = 1, 2


def make_actions() -> pd.DataFrame:
    """Two teams acting across both halves, with goals at known minutes.

    Home scores at 10' and 80'; away scores at 50'. Both teams act inside the first window
    so own-goal team resolution always has both ids available.
    """
    rows: list[dict] = []

    def add(minute: float, team: int, type_name: str, result: str = "success", x: float = 50.0):
        period = 1 if minute <= 45 else 2
        seconds = (minute if period == 1 else minute - 45) * 60
        rows.append(
            {
                "period_id": period,
                "time_seconds": seconds,
                "team_id": team,
                "player_id": 100 + team,
                "type_name": type_name,
                "result_name": result,
                "start_x": x,
                "start_y": 34.0,
                "end_x": x + 5,
                "end_y": 34.0,
                "game_id": 7,
            }
        )

    # Dense-ish passing through the match so cumulative stats change per checkpoint.
    for minute in range(1, 90, 2):
        add(float(minute), HOME if minute % 4 else AWAY, "pass", x=40.0 + minute / 3)

    add(10.0, HOME, "shot", "success", x=95.0)   # home goal
    add(50.0, AWAY, "shot", "success", x=95.0)   # away goal
    add(80.0, HOME, "shot", "success", x=95.0)   # home goal
    add(30.0, AWAY, "shot", "fail", x=90.0)      # not a goal

    frame = pd.DataFrame(rows)
    frame["action_id"] = range(len(frame))
    frame["season"] = "2015-2016"
    frame["provider"] = "statsbomb"
    return frame.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)


def make_outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": 7,
                "season": "2015-2016",
                "provider": "statsbomb",
                "game_day": 5,
                "home_team_id": HOME,
                "away_team_id": AWAY,
                "home_score": 2,
                "away_score": 1,
                "outcome": "home_win",
                "outcome_index": OUTCOME_TO_INDEX["home_win"],
            }
        ]
    )


# ----------------------------------------------------------------------- checkpoints


def test_checkpoints_align_with_window_ends():
    """The tabular ladder and the graph model must share one evaluation grid."""
    assert checkpoints() == [end for _, end in window_bounds()]
    assert checkpoints()[0] == 15.0
    assert checkpoints()[-1] == 90.0
    assert len(checkpoints()) == 16


def test_feature_ladder_is_nested():
    """Each rung must be a superset of the previous, or the comparison is not an ablation."""
    assert set(B0_FEATURES) < set(B1_FEATURES) < set(B2_FEATURES)
    assert FEATURE_LADDER["B2"] == B2_FEATURES


# ----------------------------------------------------------------------- goals


def test_derive_goals_finds_the_right_minutes_and_teams():
    goals = derive_goals(make_actions())
    assert len(goals) == 3
    assert sorted(goals["minute"].tolist()) == [10.0, 50.0, 80.0]
    assert (goals[goals["minute"] == 10.0]["scoring_team_id"] == HOME).all()
    assert (goals[goals["minute"] == 50.0]["scoring_team_id"] == AWAY).all()


def test_own_goal_is_credited_to_the_opposing_team():
    actions = make_actions()
    own = actions.iloc[[0]].copy()
    own["type_name"] = "shot"
    own["result_name"] = "owngoal"
    own["team_id"] = HOME
    own["action_id"] = 9_999
    goals = derive_goals(pd.concat([actions, own], ignore_index=True))
    own_goals = goals[goals["acting_team_id"] == HOME]
    own_goals = own_goals[own_goals["scoring_team_id"] == AWAY]
    assert len(own_goals) == 1, "an own goal by home must credit away"


def test_action_minutes_offsets_the_second_half():
    actions = make_actions()
    minutes = action_minutes(actions)
    second_half = actions["period_id"] == 2
    assert minutes[second_half].min() >= 45.0


# ----------------------------------------------------------------------- labels


def test_match_outcomes_encodes_from_the_home_perspective():
    games = pd.DataFrame(
        [
            {"game_id": 1, "season": "s", "provider": "p", "game_day": 1,
             "home_team_id": 1, "away_team_id": 2, "home_score": 3, "away_score": 0},
            {"game_id": 2, "season": "s", "provider": "p", "game_day": 1,
             "home_team_id": 1, "away_team_id": 2, "home_score": 1, "away_score": 1},
            {"game_id": 3, "season": "s", "provider": "p", "game_day": 1,
             "home_team_id": 1, "away_team_id": 2, "home_score": 0, "away_score": 2},
        ]
    )
    outcomes = match_outcomes(games)
    assert outcomes.set_index("game_id").loc[1, "outcome"] == "home_win"
    assert outcomes.set_index("game_id").loc[2, "outcome"] == "draw"
    assert outcomes.set_index("game_id").loc[3, "outcome"] == "away_win"


def test_games_without_scores_are_dropped_not_guessed():
    games = pd.DataFrame(
        [
            {"game_id": 1, "season": "s", "provider": "p", "game_day": 1,
             "home_team_id": 1, "away_team_id": 2, "home_score": 1, "away_score": 0},
            {"game_id": 2, "season": "s", "provider": "p", "game_day": 1,
             "home_team_id": 1, "away_team_id": 2, "home_score": None, "away_score": None},
        ]
    )
    outcomes = match_outcomes(games)
    assert outcomes["game_id"].tolist() == [1]


# ----------------------------------------------------------------------- THE leakage test


@pytest.mark.parametrize("window_index", [0, 3, 7, 11, 15])
def test_no_future_leakage_at_any_checkpoint(window_index: int):
    """A feature row must be identical whether or not the future exists.

    Builds the state table from the whole match, then from an action stream truncated at the
    checkpoint, and compares the row for that checkpoint. Any feature that peeks past `t`
    changes between the two and fails here.

    This is the test that makes every other Module 3 number trustworthy.
    """
    actions = make_actions()
    outcomes = make_outcomes()
    checkpoint = checkpoints()[window_index]

    full = build_state_table(actions, outcomes)
    truncated_actions = actions[action_minutes(actions).to_numpy() <= checkpoint]
    truncated = build_state_table(truncated_actions, outcomes)

    row_full = full[full["window_index"] == window_index].reset_index(drop=True)
    row_truncated = truncated[truncated["window_index"] == window_index].reset_index(drop=True)

    assert len(row_full) == 1 and len(row_truncated) == 1

    feature_columns = list(B2_FEATURES) + ["goals_home", "goals_away"]
    for column in feature_columns:
        a = float(row_full.loc[0, column])
        b = float(row_truncated.loc[0, column])
        assert a == pytest.approx(b, abs=1e-9), (
            f"FUTURE LEAK in {column!r} at checkpoint {checkpoint}': "
            f"{a} with the full match vs {b} when truncated"
        )


def test_scoreline_tracks_the_checkpoint():
    """Sanity: the running scoreline must actually change over the match."""
    table = build_state_table(make_actions(), make_outcomes()).set_index("window_index")
    # Home scores at 10' -> already 1-0 by the 15' checkpoint.
    assert table.loc[0, "goals_home"] == 1
    assert table.loc[0, "goals_away"] == 0
    # Away equalises at 50'; checkpoint 7 closes at 50'.
    assert table.loc[7, "goals_away"] == 1
    # Home scores again at 80'; final checkpoint sees 2-1.
    assert table.loc[15, "goals_home"] == 2
    assert table.loc[15, "goals_away"] == 1


def test_minutes_remaining_counts_down_to_zero():
    table = build_state_table(make_actions(), make_outcomes()).set_index("window_index")
    assert table.loc[0, "minutes_remaining"] == 75.0
    assert table.loc[15, "minutes_remaining"] == 0.0


def test_possession_share_is_a_share():
    table = build_state_table(make_actions(), make_outcomes())
    assert table["possession_share_home"].between(0.0, 1.0).all()


def test_rolling_form_excludes_the_current_match():
    """Pre-match form must not contain the result it is used to predict."""
    games = pd.DataFrame(
        [
            {"game_id": g, "season": "s", "provider": "p", "game_day": g,
             "home_team_id": 1, "away_team_id": 2, "home_score": 5, "away_score": 0}
            for g in (1, 2, 3)
        ]
    )
    outcomes = match_outcomes(games)
    actions = make_actions()
    # Point the fixture actions at each game in turn so the table can be built.
    frames = []
    for game_id in (1, 2, 3):
        copy = actions.copy()
        copy["game_id"] = game_id
        frames.append(copy)
    table = build_state_table(pd.concat(frames, ignore_index=True), outcomes)

    first = table[(table["game_id"] == 1) & (table["window_index"] == 0)]
    # Matchweek 1 has no prior matches, so form falls back to the neutral prior.
    assert float(first["form_ppg_home"].iloc[0]) == 1.0
    assert float(first["form_gd_home"].iloc[0]) == 0.0

    # By matchweek 3 the home side has two prior 5-0 wins: 3 points and +5 goal difference.
    third = table[(table["game_id"] == 3) & (table["window_index"] == 0)]
    assert float(third["form_ppg_home"].iloc[0]) == pytest.approx(3.0)
    assert float(third["form_gd_home"].iloc[0]) == pytest.approx(5.0)
