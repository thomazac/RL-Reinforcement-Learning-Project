import numpy as np
import pytest

from playstyle.ball_prediction import SLICE_TIMES
from playstyle.env import build_env
from playstyle.match import new_match
from playstyle.rewards import rotation_spot
from playstyle.rewards_nx import (CORE_TERMS_NX, PHASE_NX, PHASE_NX_TERMS, NextoStyleReward, air_touch_value,
                                  ball_potential, first_potential, rotation_potential, wall_distance, win_probability)
from playstyle.roles import FIRST, SECOND

NO_EVENTS = {"double_commit": None, "excused": None, "revoked": None, "nobody_challenging": None,
             "nobody_home": False}
ROLES = {"blue-0": FIRST, "blue-1": SECOND, "orange-0": FIRST, "orange-1": SECOND}
G = 0.99  # NextoStyleReward's default gamma, which the rig uses
BALL = (0, 0, 93)


def discounted(rewards):
    """sum of gamma^k * r_k over consecutive steps"""
    return sum(G ** k * r for k, r in enumerate(rewards))


def path(start, vel=(0, 0, 0)):
    return np.asarray(start, dtype=np.float64) + np.outer(SLICE_TIMES, np.asarray(vel, dtype=np.float64))


class Rig:
    """A 2v2 state held still: all cars on the ground with 50 boost, ball at rest at center, nothing predicted,
    fixed roles, a tied match with 150 s left."""

    def __init__(self, weights=None, team_spirit=0.0, opponent_scale=1.0):
        env = build_env(phase="nx")
        env.reset()
        self.state, self.agents = env.state, list(env.agents)
        spots = {"blue-0": (-2000, -3000), "blue-1": (2000, -3000), "orange-0": (-2000, 3000), "orange-1": (2000, 3000)}
        for a, car in self.state.cars.items():
            car.ball_touches, car.on_ground, car.boost_amount, car.demo_respawn_timer = 0, True, 50.0, 0.0
            car.bump_victim_id = None
            car.physics.position = np.array([*spots[a], 17], dtype=np.float32)
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
        self.ball(pos=(0, 0, 93), vel=(0, 0, 0))
        self.state.goal_scored = False
        self.info = {"ball_slices": path((0, 0, 93)), "match": new_match(time_remaining=150.0),
                     "coordination": {0: dict(NO_EVENTS), 1: dict(NO_EVENTS)}, "roles": dict(ROLES)}
        self.fn = NextoStyleReward(weights, team_spirit, opponent_scale)
        self.fn.reset(self.agents, self.state, self.info)

    def ball(self, pos=None, vel=None):
        if pos is not None:
            self.state.ball.position = np.array(pos, dtype=np.float32)
        if vel is not None:
            self.state.ball.linear_velocity = np.array(vel, dtype=np.float32)

    def move(self, agent, pos):
        self.state.cars[agent].physics.position = np.array(pos, dtype=np.float32)

    def step(self, ticks=8, terminated=False):
        self.state.tick_count += ticks
        done = {a: terminated for a in self.agents}
        out = self.fn.get_rewards(self.agents, self.state, done, {a: False for a in self.agents}, self.info)
        self.final = {a: float(out[a][0]) for a in self.agents}
        return {a: dict(zip(self.fn.names, out[a][1:].astype(np.float64))) for a in self.agents}


@pytest.fixture
def rig():
    return Rig()


# --- pure functions ---------------------------------------------------------------------------------------------

def test_win_probability_is_symmetric_monotone_and_settles_at_the_end():
    assert win_probability(0, 150) == pytest.approx(0.5)
    for t in (10, 100, 300):
        for d in range(-3, 4):
            assert win_probability(d, t) + win_probability(-d, t) == pytest.approx(1.0)
        assert win_probability(1, t) > win_probability(0, t) > win_probability(-1, t)
    assert win_probability(1, 10) > win_probability(1, 300)  # a lead is worth more with less time left
    assert win_probability(1, 0) == 1.0 and win_probability(-2, 0) == 0.0
    assert win_probability(0, 0) == 0.5 and win_probability(0, 0, overtime=True) == 0.5
    assert win_probability(1, 0, overtime=True, over=True) == 1.0


def test_potentials_point_the_right_way():
    blue_attack, orange_attack = (0, 4500, 93), (0, -4500, 93)
    assert ball_potential(blue_attack, 0) > 0.5 and ball_potential(blue_attack, 1) < -0.5
    assert ball_potential((0, 0, 93), 0) == pytest.approx(0.0, abs=1e-9)
    assert ball_potential(orange_attack, 0) == pytest.approx(-ball_potential(blue_attack, 0))
    # Closer to the ball and lined up behind it toward the opponent goal is better for the first man.
    assert first_potential((0, -500, 17), (0, 0, 93), 0) > first_potential((0, -3000, 17), (0, 0, 93), 0)
    assert first_potential((0, -500, 17), (0, 0, 93), 0) > first_potential((0, 500, 17), (0, 0, 93), 0)
    spot = rotation_spot(np.array([0.0, 0.0, 93.0]), 0)
    assert rotation_potential((*spot, 17), (0, 0, 93), 0) == pytest.approx(1.0)
    assert rotation_potential((0, 3000, 17), (0, 0, 93), 0) < 0.2


def test_wall_distance_covers_side_back_and_corner_walls():
    assert wall_distance(0, 0) == pytest.approx(4096)
    assert wall_distance(4096, 0) == 0 and wall_distance(0, -5120) == 0
    assert wall_distance(4000, -1000) == pytest.approx(96)
    assert wall_distance(3500, 4500) == pytest.approx(64 / np.sqrt(2))  # 45-degree corner: x + y = 8064


def test_air_touch_value_grows_with_height_and_halves_at_walls():
    ground = air_touch_value((0, 0, 17), (0, 0, 93))
    low, high = air_touch_value((0, 0, 400), (0, 0, 500)), air_touch_value((0, 0, 1500), (0, 0, 1600))
    assert ground < 0.01 < low < high <= 1.0
    at_wall = air_touch_value((4096, 0, 1500), (4050, 0, 1600))  # same height, against the side wall
    centre_factor = (1 + (1 - np.exp(-4096 / 2300))) / 2
    assert at_wall == pytest.approx(high * 0.5 / centre_factor)


# --- the reward --------------------------------------------------------------------------------------------------

def test_terms_weights_and_core_list_are_consistent():
    assert PHASE_NX["boost_spend"] == 0  # logged only (module docstring)
    assert list(PHASE_NX) == list(PHASE_NX_TERMS)
    assert set(CORE_TERMS_NX) <= set(PHASE_NX)
    for kept in ("save", "demo", "rotation_potential", "ball_potential", "win_prob", "goal", "concede",
                 "double_commit", "nobody_challenging", "nobody_home"):
        assert kept not in CORE_TERMS_NX
    # The training metrics (playstyle.training.phase_1b_metrics) read these names.
    for name in ("goal", "shot", "save", "clear", "air_touch", "touch", "double_commit", "nobody_challenging",
                 "nobody_home", "both_near", "practice_shot", "practice_defended", "practice_level"):
        assert name in PHASE_NX


def test_standing_still_pays_only_the_discount(rig):
    """gamma * phi - phi = -(1 - gamma) * phi: slightly below 0, so sitting in a good spot is never paid."""
    r = rig.step()
    assert r["blue-0"]["first_potential"] == pytest.approx((G - 1) * first_potential((-2000, -3000, 17), BALL, 0))
    assert r["blue-1"]["rotation_potential"] == pytest.approx((G - 1) * rotation_potential((2000, -3000, 17), BALL, 0))
    for a in rig.agents:
        assert r[a]["ball_potential"] == pytest.approx(0.0, abs=1e-9)  # ball at center: potential 0
        assert r[a]["win_prob"] == pytest.approx(0.0, abs=1e-9)  # tied: potential 0
        assert r[a]["first_potential"] <= 0 and r[a]["rotation_potential"] <= 0


def test_first_man_is_paid_for_closing_on_the_ball_not_for_staying(rig):
    near = (-1000, -1500, 17)
    rig.move("blue-0", near)
    assert rig.step()["blue-0"]["first_potential"] > 0
    assert rig.step()["blue-0"]["first_potential"] == pytest.approx((G - 1) * first_potential(near, BALL, 0))
    rig.move("blue-1", (1000, -1500, 17))  # the second man gets no first_potential
    assert rig.step()["blue-1"]["first_potential"] == 0


def test_second_man_is_paid_for_reaching_his_spot_not_for_staying(rig):
    spot = rotation_spot(np.array([0.0, 0.0, 93.0]), 0)
    rig.move("blue-1", (*spot, 17))
    assert rig.step()["blue-1"]["rotation_potential"] > 0
    staying = rig.step()["blue-1"]["rotation_potential"]
    assert staying == pytest.approx(G - 1) and staying < 0  # potential 1 at the spot: -(1 - gamma)


def test_role_potentials_telescope_through_role_switches(rig):
    """Approach as first man, hand off, drive back as second man, then take the first role again. Discounted, each
    term sums to gamma^N * last potential - first potential, so no cycle of roles and moves can farm it."""
    start = first_potential((-2000, -3000, 17), BALL, 0)  # blue-0 starts as first man
    seen = {"first_potential": [], "rotation_potential": []}
    plan = [((-1000, -1500, 17), FIRST), ((-300, -600, 17), FIRST), ((-300, -600, 17), SECOND),
            ((-800, -2500, 17), SECOND), ((-1500, -3500, 17), SECOND), ((-1500, -3500, 17), FIRST),
            ((-1000, -1500, 17), FIRST)]
    for pos, role in plan:
        rig.move("blue-0", pos)
        rig.info["roles"] = {**ROLES, "blue-0": role, "blue-1": SECOND if role == FIRST else FIRST}
        r = rig.step()["blue-0"]
        for n in seen:
            seen[n].append(r[n])
    n = len(plan)
    end = first_potential((-1000, -1500, 17), BALL, 0)
    assert discounted(seen["first_potential"]) == pytest.approx(G ** n * end - start, abs=1e-6)
    assert discounted(seen["rotation_potential"]) == pytest.approx(0.0, abs=1e-6)  # second man at neither end


def test_a_role_switch_pays_the_difference_of_the_two_potentials(rig):
    rig.info["roles"] = {"blue-0": SECOND, "blue-1": FIRST, "orange-0": FIRST, "orange-1": SECOND}
    r = rig.step()["blue-0"]
    pos = (-2000, -3000, 17)
    assert r["first_potential"] == pytest.approx(-first_potential(pos, BALL, 0))
    assert r["rotation_potential"] == pytest.approx(G * rotation_potential(pos, BALL, 0))


def test_shaping_potentials_end_at_zero_when_the_match_ends(rig):
    """On the final step the ball and role potentials count as 0, so over an episode they sum to nothing;
    win_prob keeps its final value, the result."""
    rig.info["match"].update(over=True, winner="blue", score_diff=1, blue=1)
    r = rig.step(terminated=True)
    assert r["blue-0"]["first_potential"] == pytest.approx(-first_potential((-2000, -3000, 17), BALL, 0))
    assert r["blue-1"]["rotation_potential"] == pytest.approx(-rotation_potential((2000, -3000, 17), BALL, 0))
    assert r["blue-0"]["win_prob"] == pytest.approx(G * 0.5) and r["orange-0"]["win_prob"] == pytest.approx(-G * 0.5)


def test_ball_potential_is_zero_sum_and_telescopes(rig):
    seen = {a: [] for a in rig.agents}
    for y in (1000, 2500, 4000, 2000, 0):
        rig.ball(pos=(0, y, 93))
        r = rig.step()
        assert r["blue-0"]["ball_potential"] == pytest.approx(-r["orange-0"]["ball_potential"])
        for a in rig.agents:
            seen[a].append(r[a]["ball_potential"])
    assert all(discounted(v) == pytest.approx(0.0, abs=1e-6) for v in seen.values())  # center to center


def score_blue(rig):
    """Put the ball in orange's net (rlgym works out the scoring team from the ball's half) and count the goal."""
    rig.ball(pos=(0, 5300, 300))
    rig.state.goal_scored = True
    rig.info["match"].update(blue=1, score_diff=1)


def test_goal_pays_goal_and_win_probability(rig):
    score_blue(rig)
    r = rig.step()
    gain = G * (win_probability(1, 150) - 0.5)  # from an even game (potential 0)
    assert r["blue-0"]["goal"] == 1 and r["orange-0"]["concede"] == 1
    assert r["blue-0"]["win_prob"] == pytest.approx(gain) and r["orange-0"]["win_prob"] == pytest.approx(-gain)


def test_zero_sum_terms_skip_the_opponent_subtraction():
    """final = (1 - tau) own + tau team mean - scale opponent mean + zero-sum terms, with goal, concede, ball_potential
    and win_prob in the zero-sum part only. The first men's first_potential moves too, since the ball moved."""
    tau, scale = 0.3, 1.0
    rig = Rig(team_spirit=tau, opponent_scale=scale)
    score_blue(rig)
    r = rig.step()
    w = PHASE_NX
    own = {a: sum(w[n] * r[a][n] for n in w if PHASE_NX_TERMS[n] == "one-sided") for a in rig.agents}
    shared = {a: sum(w[n] * r[a][n] for n in w if PHASE_NX_TERMS[n] == "zero-sum") for a in rig.agents}
    blue, orange = (own["blue-0"] + own["blue-1"]) / 2, (own["orange-0"] + own["orange-1"]) / 2
    for a, mine, theirs in (("blue-0", blue, orange), ("orange-0", orange, blue)):
        assert rig.final[a] == pytest.approx((1 - tau) * own[a] + tau * mine - scale * theirs + shared[a], abs=1e-5)
    gain = G * (win_probability(1, 150) - 0.5)
    assert shared["blue-0"] == pytest.approx(w["goal"] + w["win_prob"] * gain + w["ball_potential"] * r["blue-0"]["ball_potential"])
    assert shared["orange-0"] == pytest.approx(w["concede"] - w["win_prob"] * gain - w["ball_potential"] * r["blue-0"]["ball_potential"])


def test_field_reset_after_a_goal_pays_no_ball_or_car_potential(rig):
    score_blue(rig)
    rig.step()
    rig.state.goal_scored = False
    rig.ball(pos=(0, 0, 93))  # kickoff reset: ball and cars teleport
    rig.move("blue-0", (0, -4000, 17))
    r = rig.step()
    assert r["blue-0"]["ball_potential"] == 0 and r["blue-0"]["first_potential"] == 0


def test_demo_credits_the_attacker_and_the_subtraction_charges_the_victims_team(rig):
    rig.state.cars["blue-0"].bump_victim_id = "orange-0"
    rig.state.cars["orange-0"].demo_respawn_timer = 3.0
    r = rig.step()
    assert r["blue-0"]["demo"] == 1 and sum(r[a]["demo"] for a in rig.agents) == 1
    same = Rig()  # the same step without the demo: every car still pays its small potential discount
    same.step()
    # The victim's teammate loses the opponent mean of the demo (opponent_scale 1): half the weight.
    assert rig.final["orange-1"] - same.final["orange-1"] == pytest.approx(-PHASE_NX["demo"] / 2, abs=1e-5)


def test_boost_spend_charges_low_boosting_only(rig):
    rig.state.cars["blue-0"].boost_amount = 30.0
    r = rig.step()
    assert r["blue-0"]["boost_spend"] < 0 and r["blue-0"]["boost"] == 0
    car = rig.state.cars["blue-1"]
    car.physics.position = np.array([2000, -3000, 1500], dtype=np.float32)
    rig.step()
    car.boost_amount = 30.0
    assert rig.step()["blue-1"]["boost_spend"] == 0  # above goal height


def test_air_touch_needs_the_car_airborne(rig):
    rig.state.cars["blue-0"].ball_touches = 1
    rig.ball(vel=(0, 2000, 0))
    assert rig.step()["blue-0"]["air_touch"] == 0
    rig.state.cars["blue-0"].ball_touches = 0
    rig.step(ticks=60)
    car = rig.state.cars["blue-0"]
    car.ball_touches, car.on_ground = 1, False
    car.physics.position = np.array([0, -150, 900], dtype=np.float32)
    rig.ball(pos=(0, 0, 1000), vel=(0, 3000, 0))
    assert rig.step()["blue-0"]["air_touch"] > 0.05


def test_jump_touches_pay_no_air_touch():
    from playstyle.states import AERIAL_MIN_Z
    rig = Rig()
    car = rig.state.cars["blue-0"]
    car.ball_touches, car.on_ground = 1, False
    car.physics.position = np.array([0, -150, 250], dtype=np.float32)
    rig.ball(pos=(0, 0, AERIAL_MIN_Z - 20), vel=(0, 2000, 0))
    r = rig.step()["blue-0"]
    assert r["touch"] > 0 and r["air_touch"] == 0


def test_aerial_drills_log_shots_levels_and_hits_for_the_practicing_team():
    rig = Rig()
    rig.ball(pos=(-1900, -2500, 900))  # nearest blue-0 at (-2000, -3000): blue practices
    rig.info.update(start="aerial", aerial_level=0.3)
    rig.fn.reset(rig.agents, rig.state, rig.info)
    first = rig.step()
    assert first["blue-0"]["aerial_shot"] == 1 and first["blue-1"]["aerial_shot"] == 1
    assert first["orange-0"]["aerial_shot"] == 0 and first["blue-0"]["aerial_level"] == pytest.approx(1.3)
    assert -0.01 < first["blue-0"]["aerial_approach"] <= 0  # standing still: only the discount
    assert all(first[a]["aerial_approach"] == 0 for a in rig.agents if a != "blue-0")  # only the drilled car
    car = rig.state.cars["blue-0"]
    car.physics.position = np.array([-1900, -2500, 17], dtype=np.float32)  # drove under the ball
    under = rig.step()["blue-0"]["aerial_approach"]
    car.physics.position = np.array([-1900, -2500, 600], dtype=np.float32)  # jumped up toward it
    up = rig.step()["blue-0"]["aerial_approach"]
    assert up > under > 0 and up > 0.3
    car.ball_touches, car.on_ground = 1, False
    car.physics.position = np.array([-1900, -2650, 820], dtype=np.float32)
    r = rig.step()
    assert r["blue-0"]["aerial_shot"] == 0 and r["blue-0"]["aerial_hit"] == 1 and r["blue-0"]["air_touch"] > 0.3
    assert sum(r[a]["aerial_hit"] for a in rig.agents) == 1


def test_build_env_phase_nx():
    env = build_env(phase="nx", team_spirit=0.4)
    env.reset()
    assert isinstance(env.reward_fn, NextoStyleReward) and env.reward_fn.team_spirit == 0.4
    assert env.shared_info_provider.use_coordination
    for _ in range(30):
        _, rewards, _, _ = env.step({a: np.array([0]) for a in env.agents})
    assert all(r.shape == (1 + len(PHASE_NX),) and np.isfinite(r).all() for r in rewards.values())
