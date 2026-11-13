from types import SimpleNamespace as NS

import numpy as np

from playstyle.roles import FIRST, MIN_HOLD, SECOND, RoleTracker

DT = 8 / 120


def car(team, pos, vel=(0, 0, 0), touches=0, demoed=False, on_ground=True, has_jumped=False):
    return NS(team_num=team, ball_touches=touches, is_demoed=demoed, on_ground=on_ground, has_jumped=has_jumped,
              physics=NS(position=np.array(pos, float), linear_velocity=np.array(vel, float)))


def state(cars, ball=(0, 500, 93), ball_vel=(0, 100, 0)):
    return NS(cars=cars, ball=NS(position=np.array(ball, float), linear_velocity=np.array(ball_vel, float)))


def blue(a_pos=(0, -1000, 17), b_pos=(0, -3000, 17), **kw):
    return {"a": car(0, a_pos, **kw.get("a", {})), "b": car(0, b_pos, **kw.get("b", {}))}


def run(tracker, s, ttb, seconds):
    roles = None
    for _ in range(int(round(seconds / DT))):
        roles = tracker.update(s, ttb, DT)
    return roles


def test_lowest_ttb_starts_first():
    roles = RoleTracker().reset(state(blue()), {"a": 0.8, "b": 1.5})
    assert roles == {"a": FIRST, "b": SECOND}


def test_no_swap_within_margin_or_before_min_hold():
    t = RoleTracker()
    t.reset(state(blue()), {"a": 1.0, "b": 2.0})
    assert run(t, state(blue()), {"a": 1.0, "b": 0.9}, 2.0)["a"] == FIRST  # inside margin
    t = RoleTracker()
    t.reset(state(blue()), {"a": 1.0, "b": 2.0})
    assert t.update(state(blue()), {"a": 1.0, "b": 0.5}, DT)["a"] == FIRST  # hold not yet reached
    assert run(t, state(blue()), {"a": 1.0, "b": 0.5}, MIN_HOLD)["b"] == FIRST


def test_touch_releases_and_reassigns_by_ttb():
    t = RoleTracker()
    t.reset(state(blue()), {"a": 0.5, "b": 2.0})
    s = state(blue(a={"touches": 1}))
    assert t.update(s, {"a": 1.2, "b": 1.1}, DT)["b"] == FIRST


def test_committed_first_man_is_locked():
    t = RoleTracker()
    s = state(blue(a={"vel": (0, 1500, 0)}))
    t.reset(s, {"a": 0.2, "b": 2.0})
    run(t, s, {"a": 0.2, "b": 2.0}, 1.0)
    assert run(t, s, {"a": 0.6, "b": 0.1}, 1.0)["a"] == FIRST


def test_moving_away_releases():
    t = RoleTracker()
    t.reset(state(blue()), {"a": 0.5, "b": 0.8})
    s = state(blue(a={"vel": (0, -1000, 0)}))
    assert t.update(s, {"a": 0.7, "b": 0.8}, DT)["a"] == FIRST  # still the lowest TTB
    assert t.update(s, {"a": 0.9, "b": 0.8}, DT)["b"] == FIRST


def test_demoed_car_is_second():
    t = RoleTracker()
    t.reset(state(blue()), {"a": 0.5, "b": 1.0})
    s = state(blue(a={"demoed": True}))
    assert t.update(s, {"a": np.inf, "b": 1.0}, DT) == {"a": SECOND, "b": FIRST}


def test_nobody_reaches_deeper_car_is_second():
    roles = RoleTracker().reset(state(blue()), {"a": np.inf, "b": np.inf})
    assert roles == {"a": FIRST, "b": SECOND}


def test_kickoff_closest_goes_left_breaks_tie():
    ko = dict(ball=(0, 0, 93), ball_vel=(0, 0, 0))
    cars = {"diag": car(0, (-2048, -2560, 17)), "center": car(0, (0, -4608, 17))}
    assert RoleTracker().reset(state(cars, **ko), {"diag": 2.4, "center": 3.0})["diag"] == FIRST
    cars = {"right": car(0, (2048, -2560, 17)), "left": car(0, (-2048, -2560, 17)),
            "o_right": car(1, (-2048, 2560, 17)), "o_left": car(1, (2048, 2560, 17))}
    roles = RoleTracker().reset(state(cars, **ko), dict.fromkeys(cars, 2.4))
    assert roles["left"] == FIRST and roles["o_left"] == FIRST
