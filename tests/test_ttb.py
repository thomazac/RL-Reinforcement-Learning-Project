import numpy as np
import pytest

from playstyle.ball_prediction import SLICE_TIMES, BallPredictor
from playstyle.ttb import estimate_ttb

K = len(SLICE_TIMES)


def still_ball(x=0.0, y=0.0, z=93.15):
    return np.tile([x, y, z], (K, 1))


def ttb_for(pos, fwd=(0, 1, 0), vel=(0, 0, 0), boost=33, demoed=False, slices=None):
    slices = still_ball() if slices is None else slices
    return estimate_ttb([pos], [vel], [fwd], [boost], [demoed], slices)[0]


def test_closer_car_is_faster():
    assert ttb_for((0, -1000, 17)) < ttb_for((0, -2500, 17))


def test_facing_away_is_slower():
    assert ttb_for((0, -1500, 17), fwd=(0, 1, 0)) < ttb_for((0, -1500, 17), fwd=(0, -1, 0))


def test_moving_toward_ball_is_faster():
    assert ttb_for((0, -2000, 17), vel=(0, 1500, 0)) < ttb_for((0, -2000, 17))


def test_touching_is_zero_and_demoed_is_inf():
    assert ttb_for((0, -100, 17)) == 0.0
    assert ttb_for((0, -100, 17), demoed=True) == np.inf


def test_high_ball_needs_boost():
    high = still_ball(0, 0, 1200)
    assert ttb_for((0, -300, 17), boost=0, slices=high) == np.inf
    assert np.isfinite(ttb_for((0, -300, 17), boost=100, slices=high))


def test_out_of_range_is_inf():
    assert ttb_for((0, -5000, 17), boost=0, slices=still_ball(0, 5000)) == np.inf


def test_kickoff_spawn_order():
    # Standard blue spawns: diagonal, off-center, back center; all face the ball.
    spawns = np.array([[-2048, -2560, 17], [-256, -3840, 17], [0, -4608, 17]], dtype=float)
    fwd = -spawns / np.linalg.norm(spawns, axis=1, keepdims=True)
    fwd[:, 2] = 0
    t = estimate_ttb(spawns, np.zeros((3, 3)), fwd, [33.3] * 3, [False] * 3, still_ball())
    assert t[0] < t[1] < t[2]
    assert 1.8 < t[0] < 3.0, t


def test_ball_prediction_falls_and_starts_at_state():
    p = BallPredictor().predict((1000, 2000, 1000), (0, 0, 0), (0, 0, 0))
    assert p.shape == (K, 3)
    np.testing.assert_allclose(p[0], (1000, 2000, 1000), atol=1)
    # Free fall under -650 uu/s^2 for 0.5 s drops ~81 uu.
    assert p[5, 2] == pytest.approx(1000 - 0.5 * 650 * 0.25, abs=10)
    assert p[:, 2].min() > 80  # never sinks through the floor


def test_cached_prediction_matches_fresh_during_play():
    from playstyle.env import make_env
    from playstyle.scripted import RoleBot

    np.random.seed(1)
    env, bot, fresh = make_env(reward_ttb="live"), RoleBot(seed=1), BallPredictor()
    env.reset()
    for _ in range(600):
        _, _, term, trunc = env.step(bot.act(env.state, env.shared_info))
        if any(term.values()) or any(trunc.values()):
            env.reset()
        b = env.state.ball
        expected = fresh.predict(b.position, b.linear_velocity, b.angular_velocity)
        np.testing.assert_allclose(env.shared_info["ball_slices"], expected, atol=2.0)
    assert env.shared_info_provider.predictor.hits > 300


def test_ttb_from_live_env_state():
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence
    from playstyle.ttb import ttb_from_state

    engine = RocketSimEngine()
    state = engine.create_base_state()
    MutatorSequence(FixedTeamSizeMutator(2, 2), KickoffMutator()).apply(state, {})
    state = engine.set_state(state, {})
    ttb, slices = ttb_from_state(state, BallPredictor())
    assert set(ttb) == set(state.cars)
    assert any(np.isfinite(v) for v in ttb.values())
    for agent, t in ttb.items():
        if agent.startswith("blue"):
            mirror = next(o for o in ttb if o.startswith("orange")
                          and np.allclose(state.cars[o].physics.position[:2], -state.cars[agent].physics.position[:2]))
            assert t == pytest.approx(ttb[mirror])
    assert max(c.boost_amount for c in state.cars.values()) > 1  # boost is on a 0-100 scale
