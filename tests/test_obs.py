import numpy as np
import pytest

from playstyle.env import make_env
from playstyle.obs import CAR_DIM, OBS_DIM, TTB_CAP, EntityObs
from playstyle import ttb_v1


def _env_after(steps=30, seed=0):
    rng = np.random.default_rng(seed)
    env = make_env(coordination=False, seed=seed)
    env.reset()
    for _ in range(steps):
        env.step({a: np.array([rng.integers(90)]) for a in env.agents})
    return env


def _mirror_state(env):
    """Swap teams and rotate the field 180 degrees: blue's world becomes orange's mirrored view."""
    from copy import deepcopy
    s = deepcopy(env.state)
    swap = {"blue-0": "orange-0", "blue-1": "orange-1", "orange-0": "blue-0", "orange-1": "blue-1"}
    cars = {}
    for agent, car in env.state.cars.items():
        c = deepcopy(car)
        c.team_num = 1 - car.team_num
        c.physics = deepcopy(car.inverted_physics)
        c._inverted_physics = None
        cars[swap[agent]] = c
    s.cars = cars
    s.ball = deepcopy(env.state.inverted_ball)
    s._inverted_ball = None
    s.boost_pad_timers = env.state.inverted_boost_pad_timers.copy()
    s._inverted_boost_pad_timers = None
    info = dict(env.shared_info)
    info["obs_ttb"] = {swap[a]: t for a, t in env.shared_info["obs_ttb"].items()}
    info["obs_ball_slices"] = env.shared_info["obs_ball_slices"] * np.array([-1, -1, 1])
    info["prev_action"] = {swap[a]: p for a, p in env.shared_info["prev_action"].items()}
    info["profile"] = {swap[a]: v for a, v in env.shared_info["profile"].items()}
    m = env.shared_info["match"]
    info["match"] = {**m, "blue": m["orange"], "orange": m["blue"], "score_diff": -m["score_diff"]}
    return s, info, swap


def test_shape_and_finite():
    env = _env_after()
    obs = env.obs_builder.build_obs(env.agents, env.state, env.shared_info)
    for o in obs.values():
        assert o.shape == (OBS_DIM,) and np.all(np.isfinite(o))


def test_orange_sees_the_mirror_of_what_blue_sees():
    env = _env_after()
    assert env.shared_info["match"]["score_diff"] != 0  # seed 0 starts uneven, so the score flip is covered
    builder = EntityObs()
    real = builder.build_obs(env.agents, env.state, env.shared_info)
    s, info, swap = _mirror_state(env)
    mirrored = builder.build_obs(list(s.cars), s, info)
    for agent in env.agents:
        np.testing.assert_allclose(mirrored[swap[agent]], real[agent], atol=1e-4, err_msg=agent)


def test_teammate_is_relative_to_self():
    env = _env_after()
    o = env.obs_builder.build_obs(["blue-0"], env.state, env.shared_info)["blue-0"]
    me, mate = env.state.cars["blue-0"].physics, env.state.cars["blue-1"].physics
    np.testing.assert_allclose(o[CAR_DIM:CAR_DIM + 3], (mate.position - me.position) / 2300, atol=1e-5)


def test_ball_in_car_frame():
    env = make_env(coordination=False)
    env.reset()
    o = env.obs_builder.build_obs(env.agents, env.state, env.shared_info)
    off = 4 * CAR_DIM + 15
    for agent, car in env.state.cars.items():
        local = o[agent][off:off + 3] * 2300
        d = env.state.ball.position - car.physics.position
        np.testing.assert_allclose(np.linalg.norm(local), np.linalg.norm(d), rtol=1e-4)
        assert local[0] > 0.9 * np.linalg.norm(d[:2])  # kickoff cars face the ball


def test_teammate_profile_is_last():
    env = _env_after()
    env.shared_info["profile"]["blue-1"][:] = np.arange(6) / 10
    o = env.obs_builder.build_obs(["blue-0"], env.state, env.shared_info)["blue-0"]
    np.testing.assert_allclose(o[-6:], np.arange(6) / 10, atol=1e-7)


def test_unreachable_ttb_is_capped_with_flag_off():
    env = _env_after()
    info = dict(env.shared_info)
    info["obs_ttb"] = {**info["obs_ttb"], "blue-0": np.inf}
    o = env.obs_builder.build_obs(["blue-0"], env.state, info)["blue-0"]
    assert o[CAR_DIM - 2] == pytest.approx(TTB_CAP / ttb_v1.HORIZON) and o[CAR_DIM - 1] == 0


def test_previous_action_recorded():
    env = make_env(coordination=False)
    env.reset()
    assert env.shared_info["prev_action"] == {}
    obs, *_ = env.step({a: np.array([5]) for a in env.agents})
    expected = env.action_parser.parser.parser._lookup_table[5]
    np.testing.assert_allclose(obs["blue-0"][-14:-6], expected)  # profile (6) comes last
