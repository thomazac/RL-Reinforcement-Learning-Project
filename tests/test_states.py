import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from extract_replay_states import PRE_GOAL, rotations, usable_frames
from playstyle.env import REPLAY_STATES, TICK_SKIP, build_env, make_env
from playstyle.match import TICK_RATE
from playstyle.states import (BALL_RADIUS, FIELD_X, START_WEIGHTS, MixedStart, ReplayStates, apply_state,
                              coord_counter, coord_equidistant, defend_shot, ground_car, mirror, random_state,
                              state_is_valid,
                              yaw_matrix)

needs_replays = pytest.mark.skipif(not REPLAY_STATES.exists(), reason="run scripts/extract_replay_states.py first")
BALL = {"pos": np.array([0.0, 0.0, BALL_RADIUS]), "vel": np.zeros(3), "ang_vel": np.zeros(3)}
SPREAD = [ground_car((x, -3000), 0.0, 0.0, 50) for x in (-2000, 2000)] + \
         [ground_car((x, 3000), np.pi, 0.0, 50) for x in (-2000, 2000)]


def test_replay_rotation_convention():
    """R = Rx Ry Rz of the replay's rotation x, y, z: yaw alone turns the nose in the ground plane; roll alone
    (rotation about the nose) turns the roof toward -y."""
    r = rotations(np.array([[0.0, 0.0, np.pi / 2], [np.pi / 2, 0.0, 0.0]]))
    assert np.allclose(r[0][:, 0], [0, 1, 0], atol=1e-9)  # forward
    assert np.allclose(r[1][:, 2], [0, -1, 0], atol=1e-9)  # up
    assert np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-9) and np.allclose(np.linalg.det(r), 1)


def test_valid_state_geometry():
    assert state_is_valid(BALL, SPREAD)
    cases = {
        "car in the side wall": ground_car((FIELD_X - 30, 0), np.pi / 2, 0, 50),
        "car flat on the wall curve": ground_car((FIELD_X - 150, 0), np.pi / 2, 0, 50),
        "car in a corner": ground_car((3700, 4300), 0.0, 0, 50),
        "car on the ball": ground_car((100, 0), 0.0, 0, 50),
        "car on a car": ground_car((2000, -2900), 0.0, 0, 50),  # SPREAD[1] is at (2000, -3000)
    }
    for name, car in cases.items():
        assert not state_is_valid(BALL, [car] + SPREAD[1:]), name
    near_wall = ground_car((FIELD_X - 300, 0), np.pi / 2, 0, 50)  # on the floor beside the wall, clear of the curve
    assert state_is_valid(BALL, [near_wall] + SPREAD[1:])
    on_wall = {"pos": np.array([FIELD_X - 17, 0, 800]), "rot": np.column_stack([[0, 1, 0], [0, 0, 1], [-1, 0, 0]]),
               "vel": np.zeros(3), "ang_vel": np.zeros(3), "boost": 50, "airborne": False}
    assert state_is_valid(BALL, [on_wall] + SPREAD[1:])  # driving along the side wall
    for pos in ((0, 5200, 300), (0, 0, 50), (FIELD_X - 80, 0, 500)):
        assert not state_is_valid({**BALL, "pos": np.array(pos, dtype=float)}, SPREAD), pos


def test_mirror_swaps_teams_and_turns_the_field():
    ball = {"pos": np.array([100.0, -2000, 300]), "vel": np.array([10.0, 20, 30]), "ang_vel": np.array([1.0, 2, 3])}
    cars = [ground_car((i * 500.0, -1000), 0.3, 800, 10 * i) for i in range(4)]
    b, c = mirror(ball, cars)
    assert np.allclose(b["pos"], [-100, 2000, 300]) and np.allclose(b["vel"], [-10, -20, 30])
    assert np.allclose(c[0]["pos"][:2], -cars[2]["pos"][:2]) and c[0]["boost"] == cars[2]["boost"]
    assert np.allclose(c[0]["rot"] @ c[0]["rot"].T, np.eye(3)) and np.isclose(np.linalg.det(c[0]["rot"]), 1)
    assert np.allclose(c[0]["rot"][:, 0], -cars[2]["rot"][:, 0] * [1, 1, -1])  # nose turned 180 degrees


def test_usable_frames_skip_countdowns_pre_goal_and_post_goal():
    t = np.arange(0, 30, 0.1)
    ball = np.zeros((len(t), 3))
    vel = np.zeros((len(t), 3))
    vel[t >= 3] = (0, 1000, 0)  # countdown until 3 s
    ball[t >= 3, 1] = (t[t >= 3] - 3) * 400
    goal = np.searchsorted(t, 16)
    ball[goal:goal + 30, 1] = 5300  # in the net from 16 s; next kickoff at 20 s
    ball[(t >= 20) & (t < 22)] = 0
    vel[(t >= 20) & (t < 22)] = 0
    ball[t >= 22, 1] = (t[t >= 22] - 22) * 100  # back in play after the second kickoff
    keep = usable_frames(t, ball, vel, np.ones((len(t), 4, 3)))
    assert not keep[t < 3].any()
    assert keep[(t > 3.05) & (t < 16 - PRE_GOAL - 0.05)].all()
    assert not keep[(t >= 16 - PRE_GOAL) & (t < 22)].any()
    assert keep[t > 22.05].all()


@needs_replays
def test_replay_states_are_valid_physical_and_sampled_evenly_across_games():
    states = ReplayStates(REPLAY_STATES)
    rng = np.random.default_rng(0)
    for _ in range(500):
        ball, cars = states.sample(rng)
        assert state_is_valid(ball, cars)
        for c in cars:
            assert 0 <= c["boost"] <= 100 and np.linalg.norm(c["ang_vel"]) < 6
    fast = [(c["rot"][:, 0], c["vel"]) for _ in range(300) for c in states.sample(rng)[1]
            if not c["airborne"] and c["pos"][2] < 25 and np.linalg.norm(c["vel"]) > 1200]
    alignment = [abs(f @ v) / np.linalg.norm(v) for f, v in fast]
    assert np.median(alignment) > 0.95  # fast ground cars point along their velocity
    games = Counter(states.game[states.by_game[rng.integers(len(states.by_game))][0]] for _ in range(30_000))
    expected = 30_000 / len(states.by_game)
    assert all(abs(n - expected) < 0.25 * expected for n in games.values())


def test_generated_states_are_mostly_valid_and_set_up_their_situations():
    rng = np.random.default_rng(1)
    for gen in (random_state, coord_equidistant, coord_counter):
        valid = sum(state_is_valid(*gen(rng)) for _ in range(300))
        assert valid > 100, gen.__name__  # MixedStart redraws invalid ones; a third is plenty
    for _ in range(50):
        ball, cars = coord_equidistant(rng)
        d = [np.linalg.norm(c["pos"][:2] - ball["pos"][:2]) for c in cars]
        team = (0, 1) if abs(d[0] - d[1]) < 1e-6 else (2, 3)
        assert abs(d[team[0]] - d[team[1]]) < 1e-6
        assert min(d[i] for i in range(4) if i not in team) > d[team[0]] + 900
        ball, cars = coord_counter(rng)
        sign = -1 if ball["vel"][1] < 0 else 1  # the ball heads for the defending team's net
        defenders = (0, 1) if sign < 0 else (2, 3)
        assert all(cars[i]["pos"][1] * sign < -1000 for i in defenders)  # both caught upfield


def test_defend_shot_states_are_shots_on_target_with_a_keeper_back():
    from playstyle.ball_prediction import BallPredictor
    from playstyle.rewards import SHOT_HORIZON, enters_net
    rng, predictor = np.random.default_rng(2), BallPredictor()
    valid = on_target = 0
    for _ in range(300):
        ball, cars = defend_shot(rng)
        if not state_is_valid(ball, cars):
            continue
        valid += 1
        defending = 0 if ball["vel"][1] < 0 else 1
        keeper = cars[0] if defending == 0 else cars[2]
        goal = np.array([0.0, (-1 if defending == 0 else 1) * 5120.0])
        assert np.linalg.norm(keeper["pos"][:2] - goal) < 1500
        on_target += enters_net(predictor.predict(ball["pos"], ball["vel"], ball["ang_vel"]), defending, SHOT_HORIZON)
    assert valid > 100
    assert on_target / valid > 0.8


def test_practice_episodes_end_when_the_play_resolves():
    from types import SimpleNamespace as NS

    from playstyle.states import PRACTICE_SECONDS, PracticeEnd

    def game(tick, vel_y=-2000, goal=False, touches=(0, 0, 0, 0)):
        cars = {a: NS(team_num=t, ball_touches=n) for a, t, n in zip(("b0", "b1", "o0", "o1"), (0, 0, 1, 1), touches)}
        return NS(tick_count=tick, goal_scored=goal, scoring_team=1 if goal else None, cars=cars,
                  ball=NS(linear_velocity=np.array([0.0, vel_y, 0.0])))

    into_blue_net = {"ball_slices": np.tile([0.0, -5300.0, 300.0], (31, 1)), "start": "defend_shot"}
    away = {"ball_slices": np.tile([0.0, -2000.0, 300.0], (31, 1)), "start": "defend_shot"}
    agents = ["b0", "b1", "o0", "o1"]
    end = PracticeEnd()

    def ends(g, info, start=into_blue_net):
        end.reset(agents, game(0), start)  # a fresh practice episode for each case
        result = end.is_done(agents, g, info)
        assert len(set(result.values())) == 1
        return result["b0"]

    assert not ends(game(8), into_blue_net)
    assert ends(game(int(PRACTICE_SECONDS * 120)), into_blue_net)
    assert ends(game(8, goal=True), into_blue_net)
    assert ends(game(8, touches=(1, 0, 0, 0)), away)  # blue touched it off course
    assert not ends(game(8, touches=(1, 0, 0, 0)), into_blue_net)  # touched, still going in
    assert not ends(game(8, touches=(0, 0, 1, 0)), away)  # only the attacker touched it
    assert not ends(game(10_000, goal=True), into_blue_net, start={**into_blue_net, "start": "replay"})  # whole match


def test_start_weights_set_the_practice_share_and_keep_match_proportions():
    from playstyle.states import start_weights
    w = start_weights(0.3)
    assert w["defend_shot"] == 0.3 and sum(w.values()) == pytest.approx(1.0)
    assert w["replay"] / w["kickoff"] == pytest.approx(START_WEIGHTS["replay"] / START_WEIGHTS["kickoff"])


def test_shot_curriculum_steps_up_on_success_only():
    from playstyle.states import ShotCurriculum
    c = ShotCurriculum()
    for i in range(40):
        c.record(i % 2 == 0)  # 50% defended
    assert c.level == pytest.approx(0.1) and c.results == []
    for _ in range(100):
        c.record(False)
    assert c.level == pytest.approx(0.1) and len(c.results) == 40  # window kept, no step down
    c.level = 1.0
    for _ in range(40):
        c.record(True)
    assert c.level == 1.0


def test_easy_shots_are_slower_lower_and_at_the_keeper():
    from playstyle.ball_prediction import BallPredictor
    from playstyle.rewards import crossing_point
    rng, predictor = np.random.default_rng(3), BallPredictor()

    def stats(level):
        speeds, misses = [], []
        for _ in range(200):
            ball, cars = defend_shot(rng, level)
            defending = 0 if ball["vel"][1] < 0 else 1
            keeper = cars[0] if defending == 0 else cars[2]
            point = crossing_point(predictor.predict(ball["pos"], ball["vel"], ball["ang_vel"]), defending)
            speeds.append(np.linalg.norm(ball["vel"]))
            if point is not None:
                misses.append(abs(point[0] - keeper["pos"][0]))
        return np.median(speeds), np.median(misses)

    easy, hard = stats(0.0), stats(1.0)
    assert easy[0] < 0.8 * hard[0]  # slower (longer flights need more upward speed, so not by the flight-time ratio)
    assert easy[1] < 250 < hard[1]  # aimed at the keeper rather than anywhere in the goal mouth


def test_practice_end_feeds_the_curriculum():
    from types import SimpleNamespace as NS

    from playstyle.states import PracticeEnd, ShotCurriculum

    def game(tick, goal=False, ball_y=0.0, touches=(0, 0, 0, 0)):
        cars = {a: NS(team_num=t, ball_touches=n) for a, t, n in zip(("b0", "b1", "o0", "o1"), (0, 0, 1, 1), touches)}
        return NS(tick_count=tick, goal_scored=goal, scoring_team=1 if goal else None, cars=cars,
                  ball=NS(linear_velocity=np.array([0.0, -2000.0, 0.0]), position=np.array([0.0, ball_y, 300.0])))

    agents = ["b0", "b1", "o0", "o1"]
    shot = {"ball_slices": np.tile([0.0, -5300.0, 300.0], (31, 1)), "start": "defend_shot"}
    away = {**shot, "ball_slices": np.tile([0.0, -2000.0, 300.0], (31, 1))}
    curriculum = ShotCurriculum()
    end = PracticeEnd(curriculum)
    for info, g, expect in ((away, game(8, touches=(1, 0, 0, 0)), [True]),
                            (shot, game(8, goal=True), [True, False]),
                            (shot, game(10_000), [True, False])):  # timeout: not counted
        end.reset(agents, game(0), shot)
        assert all(end.is_done(agents, g, info).values())
        assert curriculum.results == expect


def test_apply_state_sets_conservative_jump_state():
    env = make_env(seed=0)
    env.reset()
    cars = [dict(c) for c in SPREAD]
    cars[0] = {**cars[0], "pos": np.array([-2000.0, -3000, 600]), "airborne": True}
    apply_state(env.state, BALL, cars)
    blue0, blue1 = env.state.cars["blue-0"], env.state.cars["blue-1"]
    assert blue0.has_flipped and blue0.has_double_jumped and blue0.has_jumped and not blue0.on_ground
    assert not (blue1.has_flipped or blue1.has_jumped) and blue1.on_ground


@needs_replays
def test_mixed_start_mix_and_first_step_physics():
    env = make_env(seed=0, mixed_starts=True)
    kinds = Counter()
    dt = TICK_SKIP / TICK_RATE
    pushed = 0
    for _ in range(400):
        env.reset()
        kinds[env.shared_info["start"]] += 1
        before = [(c.physics.position.copy(), c.physics.linear_velocity.copy()) for c in env.state.cars.values()]
        env.step({a: np.array([8]) for a in env.agents})  # no input
        after = [c.physics.position for c in env.state.cars.values()]
        pushed += max(np.linalg.norm(q - (p + v * dt)) for (p, v), q in zip(before, after)) > 40
    for kind, w in START_WEIGHTS.items():
        assert abs(kinds[kind] / 400 - w) < 0.07, (kind, kinds)
    assert pushed / 400 < 0.03  # cars pushed off their path: overlaps being resolved


def test_build_env_phase_1b_uses_mixed_starts_and_1a_does_not():
    assert isinstance(build_env(phase="1b").state_mutator.mutators[1], MixedStart)
    assert not isinstance(build_env().state_mutator.mutators[1], MixedStart)
