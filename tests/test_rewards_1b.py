import numpy as np
import pytest

from playstyle.ball_prediction import SLICE_TIMES
from playstyle.env import build_env
from playstyle.match import new_match
from playstyle.rewards import (BALL_MAX_SPEED, CEILING, PHASE_1B, PHASE_1B_TERMS, TOUCH_COOLDOWN_TICKS,
                               Phase1bReward, rotation_spot)

NO_EVENTS = {"double_commit": None, "excused": None, "revoked": None, "nobody_challenging": None,
             "nobody_home": False}


def path(start, vel=(0, 0, 0)):
    """Straight-line predicted ball positions at SLICE_TIMES."""
    return np.asarray(start, dtype=np.float64) + np.outer(SLICE_TIMES, np.asarray(vel, dtype=np.float64))


class Rig:
    """A 2v2 state held still: all cars on the ground with 50 boost, ball at rest at center, nothing predicted."""

    def __init__(self, weights=None, team_spirit=0.0, opponent_scale=1.0):
        env = build_env(phase="1b")
        env.reset()
        self.state, self.agents = env.state, list(env.agents)
        spots = {"blue-0": (-2000, -3000), "blue-1": (2000, -3000), "orange-0": (-2000, 3000), "orange-1": (2000, 3000)}
        for a, car in self.state.cars.items():  # fixed spots: the env's own start may put cars anywhere
            car.ball_touches, car.on_ground, car.boost_amount, car.demo_respawn_timer = 0, True, 50.0, 0.0
            car.physics.position = np.array([*spots[a], 17], dtype=np.float32)
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
        self.ball(pos=(0, 0, 93), vel=(0, 0, 0))
        self.state.goal_scored = False
        self.info = {"ball_slices": path((0, 0, 93)), "match": new_match(),
                     "coordination": {0: dict(NO_EVENTS), 1: dict(NO_EVENTS)}}
        self.fn = Phase1bReward(weights, team_spirit, opponent_scale)
        self.fn.reset(self.agents, self.state, self.info)

    def ball(self, pos=None, vel=None):
        if pos is not None:
            self.state.ball.position = np.array(pos, dtype=np.float32)
        if vel is not None:
            self.state.ball.linear_velocity = np.array(vel, dtype=np.float32)

    def touch(self, *agents):
        for a in self.agents:
            self.state.cars[a].ball_touches = int(a in agents)

    def step(self, ticks=8):
        self.state.tick_count += ticks
        out = self.fn.get_rewards(self.agents, self.state, {}, {}, self.info)
        self.final = {a: float(out[a][0]) for a in self.agents}
        return {a: dict(zip(self.fn.names, out[a][1:].astype(np.float64))) for a in self.agents}


@pytest.fixture
def rig():
    return Rig()


def test_touch_pays_once_per_distinct_touch_by_hit_strength(rig):
    rig.touch("blue-0")
    rig.ball(vel=(0, 3000, 0))
    assert rig.step()["blue-0"]["touch"] == pytest.approx(3000 / BALL_MAX_SPEED)
    rig.ball(vel=(0, 3500, 0))  # contact continues (a push): no second payment
    assert rig.step()["blue-0"]["touch"] == 0
    rig.touch()
    rig.step()
    rig.touch("blue-0")  # new contact, but inside the cooldown
    rig.ball(vel=(0, 1000, 0))
    assert TOUCH_COOLDOWN_TICKS > 16
    assert rig.step()["blue-0"]["touch"] == 0
    rig.touch()
    rig.step(ticks=TOUCH_COOLDOWN_TICKS)
    rig.touch("blue-0")
    rig.ball(vel=(0, 2200, 0))  # 1200 uu/s change from 1000
    assert rig.step()["blue-0"]["touch"] == pytest.approx(1200 / BALL_MAX_SPEED)


def test_air_touch_scales_with_ball_height_only_when_airborne(rig):
    rig.state.cars["blue-0"].on_ground = False
    rig.ball(pos=(0, 0, CEILING / 2), vel=(0, 1000, 0))
    rig.touch("blue-0")
    r = rig.step()["blue-0"]
    assert r["air_touch"] == pytest.approx(0.5) and r["touch"] > 0
    ground = Rig()
    ground.ball(pos=(0, 0, CEILING / 2), vel=(0, 1000, 0))
    ground.touch("blue-0")
    assert ground.step()["blue-0"]["air_touch"] == 0


def test_save_needs_a_ball_headed_into_the_own_net_that_the_touch_stops(rig):
    rig.ball(pos=(0, -4500, 300))
    rig.info["ball_slices"] = path((0, -4500, 300), (0, -2000, 0))  # into blue's net in ~0.3 s
    rig.step()
    rig.touch("blue-0")
    rig.info["ball_slices"] = path((0, -4500, 300), (0, 2000, 0))
    r = rig.step()
    assert r["blue-0"]["save"] == 1 and r["blue-1"]["save"] == 0
    still_in = Rig()
    still_in.ball(pos=(0, -4500, 300))
    still_in.info["ball_slices"] = path((0, -4500, 300), (0, -2000, 0))
    still_in.step()
    still_in.touch("blue-0")
    still_in.info["ball_slices"] = path((0, -4500, 300), (0, -1500, 0))  # touched but still going in
    assert still_in.step()["blue-0"]["save"] == 0


def test_clear_needs_the_touch_to_send_the_ball_out_of_the_defensive_third(rig):
    rig.ball(pos=(0, -3000, 93))
    rig.info["ball_slices"] = path((0, -3000, 93))
    rig.step()
    rig.touch("blue-0")
    rig.ball(vel=(0, 2000, 0))
    rig.info["ball_slices"] = path((0, -3000, 93), (0, 2000, 0))  # crosses y = -1707 in ~0.65 s
    assert rig.step()["blue-0"]["clear"] == 1
    upfield = Rig()  # the same touch in midfield is not a clear
    upfield.touch("blue-0")
    upfield.ball(vel=(0, 2000, 0))
    upfield.info["ball_slices"] = path((0, 0, 93), (0, 2000, 0))
    assert upfield.step()["blue-0"]["clear"] == 0


def test_shot_is_a_touch_that_puts_the_ball_on_course_for_the_opponent_net(rig):
    rig.ball(pos=(0, 4000, 300))
    rig.info["ball_slices"] = path((0, 4000, 300))
    rig.step()
    rig.touch("blue-0")
    rig.ball(vel=(0, 3000, 0))
    rig.info["ball_slices"] = path((0, 4000, 300), (0, 3000, 0))
    r = rig.step()
    assert r["blue-0"]["shot"] == 1 and r["orange-0"]["shot"] == 0


def test_boost_pays_sqrt_gains_but_not_refills(rig):
    car = rig.state.cars["blue-0"]
    car.boost_amount = 25.0
    rig.step()
    car.boost_amount = 100.0
    assert rig.step()["blue-0"]["boost"] == pytest.approx(1.0 - 0.5)
    car.boost_amount = 10.0
    assert rig.step()["blue-0"]["boost"] == 0
    car.demo_respawn_timer = 2.0
    rig.step()
    car.demo_respawn_timer, car.boost_amount = 0.0, 33.0  # respawn refill
    assert rig.step()["blue-0"]["boost"] == 0
    rig.ball(pos=(0, 5300, 300))
    rig.state.goal_scored = True
    rig.step()
    rig.state.goal_scored = False
    car.boost_amount = 100.0  # kickoff reset after the goal
    assert rig.step()["blue-0"]["boost"] == 0


def test_speed_to_ball_pays_only_the_first_man(rig):
    for a in ("blue-0", "blue-1"):
        car = rig.state.cars[a]
        to_ball = rig.state.ball.position - car.physics.position
        car.physics.linear_velocity = (1000 * to_ball / np.linalg.norm(to_ball)).astype(np.float32)
    rig.info["roles"] = {"blue-0": "first", "blue-1": "second", "orange-0": "first", "orange-1": "second"}
    r = rig.step()
    assert r["blue-0"]["speed_to_ball"] == pytest.approx(1000 / 2300, rel=1e-3)
    assert r["blue-1"]["speed_to_ball"] == 0


def test_rotation_spot_sits_off_the_far_post_and_deepens_with_the_ball():
    ball_right_deep = rotation_spot(np.array([2000.0, -4000.0, 93]), team=0)  # ball on blue's right, deep
    ball_right_high = rotation_spot(np.array([2000.0, 4000.0, 93]), team=0)  # same side, in orange's half
    assert ball_right_deep[0] < 0  # ball deep on the right: off the far (left) post
    assert abs(ball_right_high[0]) < 2000  # ball upfield: toward the middle, short of the ball's side
    assert ball_right_deep[1] < ball_right_high[1] < 0  # deeper when the ball is deeper, always in blue's half
    orange = rotation_spot(np.array([2000.0, 4000.0, 93]), team=1)  # ball deep in orange's half
    assert orange[1] > 0 and orange[0] < 0


def test_rotation_pays_only_the_second_man(rig):
    rig.info["roles"] = {"blue-0": "first", "blue-1": "second", "orange-0": "first", "orange-1": "second"}
    second = rig.state.cars["blue-1"]
    spot = rotation_spot(rig.state.ball.position.astype(float), 0)
    second.physics.position = np.array([spot[0] + 2000, spot[1], 17], dtype=np.float32)
    second.physics.linear_velocity = np.array([-1150, 0, 0], dtype=np.float32)  # straight at the spot, half speed
    first = rig.state.cars["blue-0"]
    first.physics.position = np.array([spot[0] + 2000, spot[1] + 500, 17], dtype=np.float32)
    first.physics.linear_velocity = np.array([-1150, 0, 0], dtype=np.float32)
    r = rig.step()
    assert r["blue-1"]["rotation"] == pytest.approx(0.5, abs=1e-3) and r["blue-0"]["rotation"] == 0
    second.physics.position = np.array([spot[0] + 100, spot[1], 17], dtype=np.float32)  # arrived
    second.physics.linear_velocity = np.zeros(3, dtype=np.float32)
    assert rig.step()["blue-1"]["rotation"] == 1.0


def test_both_near_flags_a_team_with_both_cars_at_the_ball(rig):
    for a, pos in (("blue-0", (500, 0, 17)), ("blue-1", (-600, 300, 17)), ("orange-0", (700, 0, 17))):
        rig.state.cars[a].physics.position = np.array(pos, dtype=np.float32)
    r = rig.step()
    assert r["blue-0"]["both_near"] == r["blue-1"]["both_near"] == 1
    assert r["orange-0"]["both_near"] == r["orange-1"]["both_near"] == 0


def test_practice_credit_only_in_practice_episodes():
    shot = path((0, -3000, 300), (0, -2500, 0))  # into blue's net at x = 0 in ~0.85 s
    for practice in (True, False):
        rig = Rig()
        rig.info.update(ball_slices=shot, practice_level=0.3)
        if practice:
            rig.info["start"] = "defend_shot"
        rig.ball(pos=(0, -3000, 300), vel=(0, -2500, 0))
        rig.fn.reset(rig.agents, rig.state, rig.info)
        keeper = rig.state.cars["blue-0"]
        keeper.physics.position = np.array([1000, -5000, 17], dtype=np.float32)
        keeper.physics.linear_velocity = np.array([-2300, 0, 0], dtype=np.float32)  # straight at the crossing point
        r = rig.step()["blue-0"]
        if practice:
            assert r["practice_shot"] == 1 and r["practice_level"] == pytest.approx(1.3)
            assert r["path_approach"] == pytest.approx(1.0, abs=0.05)
        else:
            assert r["practice_shot"] == r["path_approach"] == r["practice_level"] == 0
        rig.touch("blue-0")  # a touch that takes it off course
        rig.info["ball_slices"] = path((0, -4000, 300), (0, 2000, 0))
        r = rig.step()["blue-0"]
        assert r["practice_shot"] == 0  # first step only
        assert (r["shot_touch"], r["practice_defended"]) == ((1, 1) if practice else (0, 0))


def test_coordination_events_charge_the_right_cars(rig):
    rig.info["coordination"][0] = {**NO_EVENTS, "double_commit": "blue-1", "revoked": {"blue-1": 3},
                                   "nobody_challenging": "blue-0", "nobody_home": True}
    r = rig.step()
    assert r["blue-1"]["double_commit"] == 4 and r["blue-0"]["double_commit"] == 0
    assert r["blue-0"]["nobody_challenging"] == 1 and r["blue-1"]["nobody_challenging"] == 0
    assert r["blue-0"]["nobody_home"] == r["blue-1"]["nobody_home"] == 1
    assert all(r[a]["nobody_home"] == 0 for a in ("orange-0", "orange-1"))


def test_goal_and_concede(rig):
    rig.ball(pos=(0, 5300, 300))  # in orange's net: blue scored
    rig.state.goal_scored = True
    r = rig.step()
    assert r["blue-0"]["goal"] == r["blue-1"]["goal"] == 1 and r["blue-0"]["concede"] == 0
    assert r["orange-0"]["concede"] == 1 and r["orange-0"]["goal"] == 0


def test_ball_to_goal_is_exactly_zero_sum(rig):
    rig.ball(pos=(1500, -2000, 400), vel=(300, 2500, 200))
    r = rig.step()
    blue, orange = r["blue-0"]["ball_to_goal"], r["orange-0"]["ball_to_goal"]
    assert blue > 0.2 and blue == pytest.approx(-orange, abs=1e-12)
    assert r["blue-1"]["ball_to_goal"] == blue


def test_win_and_loss_on_the_final_step(rig):
    rig.info["match"] = {**new_match(), "over": True, "winner": "orange"}
    r = rig.step()
    assert r["orange-0"]["win"] == r["orange-1"]["win"] == 1
    assert r["blue-0"]["win"] == r["blue-1"]["win"] == -1


def test_zero_sum_terms_skip_the_opponent_subtraction():
    """A goal is worth the goal weight to the scorers and the concede weight to the other team, not their sum."""
    rig = Rig(team_spirit=0.3)
    rig.ball(pos=(0, 5300, 300))
    rig.state.goal_scored = True
    rig.step()
    assert rig.final["blue-0"] == pytest.approx(PHASE_1B["goal"])
    assert rig.final["orange-0"] == pytest.approx(PHASE_1B["concede"])


def test_one_sided_terms_blend_and_subtract():
    tau, scale = 0.3, 1.0
    rig = Rig(team_spirit=tau, opponent_scale=scale)
    rig.touch("blue-0")
    rig.ball(vel=(0, 3000, 0))  # the ball moves toward orange's net: ball_to_goal is non-zero too
    r = rig.step()
    w = PHASE_1B
    own = {a: sum(w[n] * r[a][n] for n in w if PHASE_1B_TERMS[n] == "one-sided") for a in rig.agents}
    shared = {a: sum(w[n] * r[a][n] for n in w if PHASE_1B_TERMS[n] == "zero-sum") for a in rig.agents}
    blue = (own["blue-0"] + own["blue-1"]) / 2
    orange = (own["orange-0"] + own["orange-1"]) / 2
    assert own["blue-0"] > 0
    assert rig.final["blue-0"] == pytest.approx((1 - tau) * own["blue-0"] + tau * blue - scale * orange
                                                + shared["blue-0"], abs=1e-5)
    assert rig.final["orange-0"] == pytest.approx((1 - tau) * own["orange-0"] + tau * orange - scale * blue
                                                  + shared["orange-0"], abs=1e-5)


def test_build_env_phase_1b():
    env = build_env(phase="1b", team_spirit=0.2)
    env.reset()
    assert isinstance(env.reward_fn, Phase1bReward) and env.reward_fn.team_spirit == 0.2
    assert env.shared_info_provider.use_coordination
    for _ in range(30):
        _, rewards, _, _ = env.step({a: np.array([0]) for a in env.agents})
    assert all(r.shape == (1 + len(PHASE_1B),) and np.isfinite(r).all() for r in rewards.values())
