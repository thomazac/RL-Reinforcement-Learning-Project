import copy
from types import SimpleNamespace as NS

import numpy as np

from playstyle.env import make_env
from playstyle.match import START_SCORE_WEIGHTS, MatchEngine, RandomMatchStart, new_match

DT = 8 / 120


def fake_state(ball=(500, 0, 93), ball_vel=(1000, 0, 0), goal=False, scoring_team=0):
    return NS(goal_scored=goal, scoring_team=scoring_team,
              ball=NS(position=np.array(ball, float), linear_velocity=np.array(ball_vel, float)))


def test_clock_waits_for_kickoff_then_runs():
    eng, m = MatchEngine(None), new_match(time_remaining=100)
    eng._update(m, fake_state(ball=(0, 0, 93), ball_vel=(0, 0, 0)), DT)
    assert m["time_remaining"] == 100
    eng._update(m, fake_state(), DT)
    assert m["time_remaining"] == 100 - DT


def test_zero_seconds_waits_for_ball_on_ground_then_ends_with_winner():
    eng, m = MatchEngine(None), new_match(blue=1, time_remaining=DT / 2)
    eng._update(m, fake_state(ball=(500, 0, 800)), DT)
    assert m["time_remaining"] == 0 and not m["over"]
    eng._update(m, fake_state(), DT)
    assert m["over"] and m["winner"] == "blue"


def test_tie_at_zero_goes_to_overtime_and_next_goal_wins():
    eng, m = MatchEngine(None), new_match(blue=1, orange=1, time_remaining=DT / 2)
    eng._update(m, fake_state(), DT)
    assert m["overtime"] and not m["over"] and eng._kickoff_pending
    eng._update(m, fake_state(), 100.0)
    assert m["time_remaining"] == 0  # overtime clock doesn't run
    eng._update(m, fake_state(goal=True, scoring_team=1), DT)
    assert m["over"] and m["winner"] == "orange" and m["goal"] == "orange"


def test_goal_before_zero_continues_with_kickoff():
    eng, m = MatchEngine(None), new_match(time_remaining=100)
    eng._update(m, fake_state(goal=True, scoring_team=0), DT)
    assert m["blue"] == 1 and m["score_diff"] == 1 and not m["over"] and eng._kickoff_pending


def test_start_scores_favor_close_games():
    mut, info = RandomMatchStart(seed=0), {}
    diffs, clocks = [], []
    for _ in range(4000):
        mut.apply(None, info)
        diffs.append(info["match"]["score_diff"])
        clocks.append(info["match"]["time_remaining"])
    diffs = np.array(diffs)
    assert set(diffs) == set(START_SCORE_WEIGHTS)
    assert np.mean(np.abs(diffs) <= 1) > 0.75 and np.mean(np.abs(diffs) == 3) < 0.1
    assert 0 < min(clocks) and max(clocks) <= 300 and np.mean(np.array(clocks) < 60) > 0.1


def _done_flags(env, steps=1):
    noop = {a: np.array([8]) for a in env.agents}
    for _ in range(steps):
        _, _, term, trunc = env.step(noop)
    return all(term.values()), all(trunc.values())


def test_match_end_terminates_and_cutoffs_truncate():
    """PPO bootstraps the value after a truncation but not after a termination; mixing them up would teach
    the bot that stalling the ball until the no-touch cutoff ends all future reward."""
    env = make_env(coordination=False, seed=0)
    env.reset()
    env.shared_info["match"] = new_match(blue=1, time_remaining=DT / 2)  # ball rests at kickoff: clock is paused
    s = copy.deepcopy(env.state)
    s.ball.linear_velocity = np.array([0, 0, -10], dtype=np.float32)
    s.ball.position = np.array([1000, 0, 93], dtype=np.float32)
    env.transition_engine.set_state(s, env.shared_info)
    assert _done_flags(env) == (True, False)
    assert env.shared_info["match"]["winner"] == "blue"

    no_touch, timeout = env.truncation_cond.conditions
    for cond in (no_touch, timeout):
        env.reset()
        env.shared_info["match"] = new_match(time_remaining=200)
        if cond is no_touch:
            cond.last_touch_tick -= 30 * 120
        else:
            cond.initial_tick -= 600 * 120
        assert _done_flags(env) == (False, True), type(cond).__name__


def test_goal_in_env_resets_to_kickoff_without_ending_episode():
    env = make_env(coordination=False, seed=0)
    env.reset()
    env.shared_info["match"] = new_match(time_remaining=200)
    s = copy.deepcopy(env.state)
    s.ball.position = np.array([0, 4800, 200], dtype=np.float32)
    s.ball.linear_velocity = np.array([0, 2000, 0], dtype=np.float32)
    env.transition_engine.set_state(s, env.shared_info)
    noop = {a: np.array([8]) for a in env.agents}
    for _ in range(30):
        _, _, term, trunc = env.step(noop)
        if env.shared_info["match"]["goal"]:
            break
    assert env.shared_info["match"]["goal"] == "blue" and env.shared_info["match"]["blue"] == 1
    assert not any(term.values())
    env.step(noop)
    ball = env.state.ball.position
    assert abs(ball[0]) < 1 and abs(ball[1]) < 1 and not env.state.goal_scored
