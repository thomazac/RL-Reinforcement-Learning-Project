import numpy as np
import pytest

from playstyle.env import build_env
from playstyle.rewards import CAR_MAX_SPEED, PHASE_1A, ShapedReward, components


@pytest.fixture
def env():
    env = build_env()
    env.reset()
    return env


def place(car, pos, vel=(0, 0, 0), forward=(1, 0, 0)):
    car.physics.position = np.array(pos, dtype=np.float32)
    car.physics.linear_velocity = np.array(vel, dtype=np.float32)
    f = np.array(forward, dtype=np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)
    car.physics.rotation_mtx = np.stack([f, np.cross(up, f), up], axis=1)


def test_components(env):
    state = env.state
    state.ball.position = np.array([1000, 0, 93], dtype=np.float32)
    car = state.cars["blue-0"]
    place(car, (0, 0, 93), vel=(1150, 0, 0), forward=(1, 0, 0))
    car.ball_touches, car.on_ground, car.demo_respawn_timer = 1, True, 0
    assert np.allclose(components(car, state.ball), [1, 1150 / CAR_MAX_SPEED, 1, 0], atol=1e-3)

    place(car, (0, 0, 93), vel=(-500, 0, 0), forward=(-1, 0, 0))  # driving away, nose away
    car.ball_touches, car.on_ground = 0, False
    assert np.allclose(components(car, state.ball), [0, 0, -1, 1], atol=1e-3)

    car.demo_respawn_timer = 2.0
    assert not components(car, state.ball).any()


def test_reward_vector_and_team_spirit(env):
    state = env.state
    agents = list(state.cars)
    raw = {a: components(state.cars[a], state.ball) for a in agents}
    w = np.array(list(PHASE_1A.values()))
    own = {a: raw[a] @ w for a in agents}
    team = {t: np.mean([own[a] for a in agents if state.cars[a].team_num == t]) for t in (0, 1)}
    for tau in (0.0, 0.3):
        out = ShapedReward(team_spirit=tau).get_rewards(agents, state, {}, {}, {})
        for a in agents:
            t = state.cars[a].team_num
            assert out[a].dtype == np.float32 and out[a].shape == (1 + len(PHASE_1A),)
            assert np.allclose(out[a][1:], raw[a])
            assert out[a][0] == pytest.approx((1 - tau) * own[a] + tau * team[t] - team[1 - t], abs=1e-4)


def test_build_env_trains_on_phase_1a_reward(env):
    assert isinstance(env.reward_fn, ShapedReward) and env.reward_fn.weights == PHASE_1A
    assert env.reward_fn.team_spirit == 0.0
    assert env.shared_info_provider.use_coordination is False
    _, rewards, _, _ = env.step({a: np.array([0]) for a in env.agents})
    assert all(r.shape == (1 + len(PHASE_1A),) for r in rewards.values())
