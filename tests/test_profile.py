import copy
from types import SimpleNamespace as NS

import numpy as np

from playstyle import profile_v1
from playstyle.env import make_env
from playstyle.match import new_match
from playstyle.profile_v1 import FEATURES, indicators

DT = 8 / 120
F = {name: i for i, name in enumerate(FEATURES)}


def car(team, y, speed=1000.0, demoed=False, touches=0, victim=None):
    return NS(team_num=team, is_demoed=demoed, ball_touches=touches, bump_victim_id=victim,
              physics=NS(position=np.array([0.0, y, 17.0]), linear_velocity=np.array([speed, 0.0, 0.0])))


def state(cars, ball_y=0.0):
    return NS(cars=cars, ball=NS(position=np.array([0.0, ball_y, 93.0])))


def base(**over):
    cars = {"b0": car(0, -1000), "b1": car(0, -3000), "o0": car(1, 1000), "o1": car(1, 3000)}
    cars.update(over)
    return cars


TTB = {"b0": 0.5, "b1": 1.5, "o0": 0.8, "o1": 2.0}


def test_frozen_constants():
    np.testing.assert_array_equal(profile_v1.WARMUP, [0.725, 0.500, 0.425, 0.500, 0.013, 0.500])
    assert (profile_v1.EMA_SECONDS, profile_v1.DEMO_SCALE, profile_v1.VERSION) == (25.0, 2.0, "profile-v1")


def test_position_speed_and_challenge():
    x = indicators(state(base()), TTB, [], DT)
    assert x["b0"][F["behind_ball"]] == 1 and x["b1"][F["last_man"]] == 1 and x["b0"][F["last_man"]] == 0
    assert x["o0"][F["behind_ball"]] == 1  # orange's own goal is at +y
    assert x["b0"][F["speed"]] == np.float64(1000 / 2300)
    assert x["b0"][F["challenge"]] == 1 and x["b1"][F["challenge"]] == 0


def test_challenge_skips_ties_and_both_unreachable():
    for b0, b1 in ((1.0, 1.0), (np.inf, np.inf)):
        x = indicators(state(base()), {**TTB, "b0": b0, "b1": b1}, [], DT)
        assert np.isnan(x["b0"][F["challenge"]]) and np.isnan(x["b1"][F["challenge"]])


def test_demoed_car_is_not_first_and_not_last_man():
    x = indicators(state(base(b0=car(0, -1000, demoed=True))), {**TTB, "b0": np.inf}, [], DT)
    assert x["b0"][F["challenge"]] == 0 and x["b1"][F["challenge"]] == 1
    assert x["b0"][F["last_man"]] == 0 and x["b1"][F["last_man"]] == 1
    assert np.isnan(x["b0"][F["behind_ball"]]) and np.isnan(x["b0"][F["speed"]])


def test_touch_share_and_demo_credit():
    x = indicators(state(base(b1=car(0, -3000, touches=1))), TTB, [], DT)
    assert x["b1"][F["touch_share"]] == 1 and x["b0"][F["touch_share"]] == 0
    assert np.isnan(x["o0"][F["touch_share"]])
    cars = base(b0=car(0, -1000, victim="o0"), o0=car(1, 1000, demoed=True))
    x = indicators(state(cars), TTB, ["o0"], DT)
    assert x["b0"][F["demo_rate"]] == 60 / DT / profile_v1.DEMO_SCALE and x["b1"][F["demo_rate"]] == 0


def test_profile_carries_over_goal_and_resets_at_episode_start():
    rng = np.random.default_rng(0)
    env = make_env(coordination=False, seed=0)
    env.reset()
    for a, v in env.shared_info["profile"].items():
        np.testing.assert_array_equal(v, profile_v1.WARMUP)
    for _ in range(300):
        env.step({a: np.array([rng.integers(90)]) for a in env.agents})
    env.shared_info["match"] = new_match(time_remaining=200)
    s = copy.deepcopy(env.state)
    s.ball.position = np.array([0, 4800, 200], dtype=np.float32)
    s.ball.linear_velocity = np.array([0, 2000, 0], dtype=np.float32)
    env.transition_engine.set_state(s, env.shared_info)
    before = {a: v.copy() for a, v in env.shared_info["profile"].items()}
    noop = {a: np.array([8]) for a in env.agents}
    for _ in range(40):
        env.step(noop)
        if env.shared_info["match"]["goal"]:
            break
    env.step(noop)  # the kickoff reset
    assert env.shared_info["kickoff"]
    for a, v in env.shared_info["profile"].items():
        assert not np.allclose(before[a], profile_v1.WARMUP)
        np.testing.assert_allclose(v, before[a], atol=0.15, err_msg=a)  # a few EMA steps, not a reset
    env.reset()
    for v in env.shared_info["profile"].values():
        np.testing.assert_array_equal(v, profile_v1.WARMUP)
