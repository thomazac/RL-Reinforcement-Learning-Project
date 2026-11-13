from types import SimpleNamespace as NS

import numpy as np

from playstyle.ball_prediction import SLICE_TIMES
from playstyle.coordination import NOBODY_CHALLENGING_TIME, CoordinationTracker
from playstyle.roles import FIRST, SECOND

DT = 8 / 120


def car(team, pos, vel=(0, 0, 0), touches=0):
    return NS(team_num=team, ball_touches=touches, is_demoed=False,
              physics=NS(position=np.array(pos, float), linear_velocity=np.array(vel, float)))


def state(cars, ball=(0, 0, 93)):
    return NS(cars=cars, ball=NS(position=np.array(ball, float)))


def still(ball=(0, 0, 93)):
    return np.tile(ball, (len(SLICE_TIMES), 1)).astype(float)


ROLES = {"b0": FIRST, "b1": SECOND, "o0": FIRST, "o1": SECOND}


def converging(b1_pos=(-500, -500, 17), b1_vel=(1000, 1000, 0)):
    return {"b0": car(0, (0, -600, 17), (0, 1500, 0)), "b1": car(0, b1_pos, b1_vel),
            "o0": car(1, (0, 3000, 17)), "o1": car(1, (0, 4000, 17))}


def update(tracker, s, ttb, slices, n=1, roles=ROLES):
    for _ in range(n):
        events = tracker.update(s, ttb, roles, slices, DT)
    return events


TTB_CLOSE = {"b0": 0.3, "b1": 0.4, "o0": 2.0, "o1": 2.5}


def test_double_commit_penalizes_second_man():
    ev = update(CoordinationTracker(), state(converging()), TTB_CLOSE, still())[0]
    assert ev["double_commit"] == "b1" and ev["excused"] is None


def test_no_double_commit_when_second_man_is_far_or_slow():
    assert update(CoordinationTracker(), state(converging()), {**TTB_CLOSE, "b1": 1.5}, still())[0]["double_commit"] is None
    slow = converging(b1_vel=(100, 100, 0))
    assert update(CoordinationTracker(), state(slow), TTB_CLOSE, still())[0]["double_commit"] is None


def test_double_commit_excused_when_ball_heads_into_own_net():
    slices = still()
    slices[5:] = (0, -5300, 93)
    ev = update(CoordinationTracker(), state(converging()), TTB_CLOSE, slices)[0]
    assert ev["double_commit"] is None and ev["excused"] == "save"


def test_double_commit_excused_for_demo_play():
    cars = converging()
    # b1 drives at an opponent standing well off the line to the ball.
    cars["o0"] = car(1, (1500, 0, 17))
    cars["b1"] = car(0, (-500, -500, 17), (1500, 375, 0))
    ev = update(CoordinationTracker(), state(cars), TTB_CLOSE, still(), n=20)[0]
    assert ev["double_commit"] is None and ev["excused"] == "demo"


def demo_excused_tracker(n, roles=ROLES):
    cars = converging()
    cars["o0"] = car(1, (1500, 0, 17))
    cars["b1"] = car(0, (-500, -500, 17), (1500, 375, 0))
    tracker = CoordinationTracker()
    update(tracker, state(cars), TTB_CLOSE, still(), n=n, roles=roles)
    return tracker, cars


def test_demo_excuse_revoked_and_charged_to_the_hunter_that_touches_ball():
    hunter_is_first = {**ROLES, "b0": SECOND, "b1": FIRST}
    for roles in (ROLES, hunter_is_first):
        tracker, cars = demo_excused_tracker(20, roles)
        cars["b1"] = car(0, (0, -100, 17), touches=1)
        ev = update(tracker, state(cars), {**TTB_CLOSE, "b1": 2.0}, still(), roles=roles)[0]
        assert ev["revoked"] == {"b1": 20 - 1}  # the excuse needs a step of aim history before it applies


def test_demo_excuse_stands_when_teammate_touches_or_window_passes():
    tracker, cars = demo_excused_tracker(20)
    cars["b0"] = car(0, (0, -100, 17), touches=1)
    cars["b1"] = car(0, (1400, 0, 17))
    far = {**TTB_CLOSE, "b0": 2.0, "b1": 2.0}
    assert update(tracker, state(cars), far, still())[0]["revoked"] is None
    cars["b0"].ball_touches = 0
    update(tracker, state(cars), far, still(), n=int(0.5 / DT) + 1)
    cars["b1"] = car(0, (0, -100, 17), touches=1)
    assert update(tracker, state(cars), far, still())[0]["revoked"] is None


def test_nobody_home_when_both_in_front_of_ball_in_own_third():
    cars = {"b0": car(0, (0, 0, 17)), "b1": car(0, (500, -1000, 17)), "o0": car(1, (0, -3500, 17)),
            "o1": car(1, (0, 3000, 17))}
    ttb = dict.fromkeys(cars, 2.0)
    assert update(CoordinationTracker(), state(cars, ball=(0, -3000, 93)), ttb, still((0, -3000, 93)))[0]["nobody_home"]
    cars["b1"] = car(0, (0, -4500, 17))
    assert not update(CoordinationTracker(), state(cars, ball=(0, -3000, 93)), ttb, still((0, -3000, 93)))[0]["nobody_home"]


def test_nobody_challenging_after_opponent_keeps_possession():
    ball = (0, -1000, 93)
    cars = {"b0": car(0, (0, -4800, 17)), "b1": car(0, (300, -4900, 17)),
            "o0": car(1, (0, -900, 17), touches=1), "o1": car(1, (0, 2000, 17))}
    tracker, ttb = CoordinationTracker(), dict.fromkeys(cars, 2.0)
    update(tracker, state(cars, ball), ttb, still(ball))
    cars["o0"].ball_touches = 0
    steps = int(NOBODY_CHALLENGING_TIME / DT)
    assert update(tracker, state(cars, ball), ttb, still(ball), n=steps - 2)[0]["nobody_challenging"] is None
    assert update(tracker, state(cars, ball), ttb, still(ball), n=4)[0]["nobody_challenging"] == "b0"
    cars["b0"] = car(0, (0, -4800, 17), (0, 1000, 0))  # now closing on the ball
    assert update(tracker, state(cars, ball), ttb, still(ball))[0]["nobody_challenging"] is None
