"""Hand-built game states for the observation v1 reference: situations scripted and random play never reach.

Each is an independent state (not a sequence). The states are written directly, not stepped through the
physics engine, so they are exactly reproducible; they cover what the observation builder and the profile
have to read, not what the engine would produce next.
  aerial      cars high in the air at any attitude, with and without a flip left
  ceiling     driving on the ceiling, dropping off it, the ball pinned against it
  upside_down upside down in the air, spinning at the angular velocity cap about each axis
  mid_flip    flipping, double-jumped, or flip expired: no flip and no jump left
  in_goal     cars and ball past the back line, at either end, and goal to goal
  corners     cars in opposite corners (largest relative distance), facing the far corner, floor and ceiling
  max_speed   ball at 6000 uu/s and 6 rad/s in each direction, cars supersonic against it and head-on
  match       overtime, the clock at zero, leads of 3 and beyond
  demos       both opponents, both teammates, the teammate alone, three and four cars demoed; new and old demos
  kickoff     kickoff layout with pads empty and full, no previous action, extreme previous actions
  touches     one, two and four cars touching on the same step
"""

import copy

import numpy as np
from rlgym.rocket_league.action_parsers import LookupTableAction

from playstyle import profile_v1, ttb_v1
from playstyle.env import make_env
from playstyle.match import new_match
from playstyle.obs import EntityObs
from playstyle.roles import is_kickoff

DT = 8 / 120
PI = np.pi
CAR_Z, CEILING, BALL_R = 17.0, 2044.0, 91.25


def _car(car, pos, yaw=0.0, pitch=0.0, roll=0.0, speed=0.0, vel=None, ang=(0, 0, 0), ground=True, boost=33.0):
    p = car.physics
    p.euler_angles = np.array([pitch, yaw, roll], dtype=np.float32)
    p.position = np.array(pos, dtype=np.float32)
    p.linear_velocity = np.array(p.forward * speed if vel is None else vel, dtype=np.float32)
    p.angular_velocity = np.array(ang, dtype=np.float32)
    car._inverted_physics = None
    car.on_ground = ground
    car.boost_amount = float(boost)
    car.has_jumped = car.has_flipped = car.has_double_jumped = car.is_jumping = car.is_holding_jump = False
    car.air_time_since_jump = car.flip_time = car.jump_time = 0.0
    car.demo_respawn_timer, car.ball_touches, car.bump_victim_id = 0.0, 0, None


def _ball(state, pos, vel=(0, 0, 0), ang=(0, 0, 0)):
    b = state.ball
    b.position, b.linear_velocity, b.angular_velocity = (np.array(v, dtype=np.float32) for v in (pos, vel, ang))
    state._inverted_ball = None


def _pads(state, timers):
    state.boost_pad_timers = np.asarray(timers, dtype=np.float32)
    state._inverted_boost_pad_timers = None


def _scenarios(base, rng):
    """Yields (name, state, overrides); overrides may set match, prev (previous actions) and old_demos."""
    ids = sorted(base.cars)
    b0, b1, o0, o1 = ids
    U = rng.uniform

    def direction():
        v = rng.normal(size=3)
        return v / np.linalg.norm(v)

    def midplay():
        s = copy.deepcopy(base)
        for car in s.cars.values():
            _car(car, (U(-3800, 3800), U(-4800, 4800), CAR_Z), yaw=U(-PI, PI), speed=U(0, 2300), boost=U(0, 100))
        _ball(s, (U(-3500, 3500), U(-4500, 4500), U(93, 1500)), vel=rng.normal(0, 1200, 3), ang=rng.normal(0, 2, 3))
        _pads(s, U(0, 10, 34) * (rng.random(34) < 0.4))
        s.tick_count, s.goal_scored = int(rng.integers(1000, 30000)), False
        return s, s.cars

    def airborne(car, z, **kw):
        _car(car, (U(-3500, 3500), U(-4500, 4500), z), yaw=U(-PI, PI), ground=False, boost=U(0, 40), **kw)
        car.has_jumped = True

    for n in (1, 1, 2, 2, 3, 3, 4, 4):
        s, cars = midplay()
        for a in rng.choice(ids, n, replace=False):
            airborne(cars[a], U(600, 1950), pitch=U(-PI / 2, PI / 2), roll=U(-PI, PI),
                     vel=direction() * U(500, 2300), ang=direction() * U(0, 5.5))
            cars[a].air_time_since_jump = U(0, 2.5)  # past 1.25 s the flip is gone
        _ball(s, (U(-3000, 3000), U(-4000, 4000), U(800, CEILING - BALL_R)), vel=direction() * U(0, 3000))
        yield "aerial", s, {}

    for k in range(4):
        s, cars = midplay()
        a = ids[k]
        if k < 2:  # driving on the ceiling
            _car(cars[a], (U(-3000, 3000), U(-4000, 4000), CEILING - CAR_Z), yaw=U(-PI, PI), roll=PI, speed=1500)
        else:  # just dropped off it: still has its jump and flip
            _car(cars[a], (U(-3000, 3000), U(-4000, 4000), 1900), yaw=U(-PI, PI), roll=PI, vel=(0, 0, -300),
                 ground=False)
        _ball(s, (U(-3000, 3000), U(-4000, 4000), CEILING - BALL_R), vel=(U(-500, 500), U(-500, 500), 0))
        yield "ceiling", s, {}

    for k in range(4):
        s, cars = midplay()
        axis = np.eye(3)[k % 3] * (1 if k < 3 else -1)
        airborne(cars[ids[k]], U(150, 900), roll=PI, pitch=U(-0.3, 0.3), vel=direction() * U(0, 1500), ang=axis * 5.5)
        cars[ids[k]].air_time_since_jump = U(0.2, 1.0)
        yield "upside_down", s, {}

    for k in range(6):
        s, cars = midplay()
        c = cars[ids[k % 4]]
        airborne(c, U(40, 300), pitch=U(-1, 1), speed=U(800, 2300))
        c.physics.angular_velocity = (c.physics.right * 5.5 * (1 if k % 2 else -1)).astype(np.float32)
        c.air_time_since_jump = U(0.1, 0.6)
        if k < 2:
            c.has_flipped, c.flip_time = True, U(0.05, 0.6)
        elif k < 4:
            c.has_double_jumped = True
        else:
            c.air_time_since_jump = U(1.3, 2.0)
        assert not c.has_flip
        yield "mid_flip", s, {}

    for k, side in enumerate((1, -1, 1, -1, 1, -1)):
        s, cars = midplay()
        for a in (ids if k >= 4 else [ids[k]]):  # the last two crowd all four cars into the goal
            _car(cars[a], (U(-800, 800), side * (5120 + U(100, 820)), CAR_Z), yaw=U(-PI, PI), speed=U(0, 1500),
                 boost=U(0, 100))
        if k in (2, 3):  # goal to goal: a car deep in the far goal, facing the ball
            _car(cars[ids[k - 2]], (0, -side * 5900, CAR_Z), yaw=side * PI / 2, boost=U(0, 100))
        _ball(s, (U(-700, 700), side * (5120 + U(100, 750)), U(93, 500)), vel=direction() * U(0, 2000))
        yield "in_goal", s, {}

    X, Y, HIGH = 2950.0, 4950.0, CEILING - 60  # the corner walls cut the field at |x| + |y| = 8064
    far = float(np.arctan2(2 * Y, 2 * X))
    for k in range(6):
        s, cars = midplay()
        z0, z1 = [(CAR_Z, CAR_Z), (HIGH, CAR_Z), (HIGH, HIGH), (CAR_Z, CAR_Z), (CAR_Z, CAR_Z), (CAR_Z, CAR_Z)][k]
        if k < 5:
            _car(cars[b0], (-X, -Y, z0), yaw=far, ground=z0 == CAR_Z)  # faces the far corner
            _car(cars[o0], (X, Y, z1), yaw=far + PI, ground=z1 == CAR_Z)
            _car(cars[b1], (X, -Y, CAR_Z), yaw=PI - far)
            _car(cars[o1], (-X, Y, CAR_Z), yaw=-far)
        else:  # everyone in one corner, the ball at the ceiling in the opposite one
            for i, a in enumerate(ids):
                _car(cars[a], (-X + 150 * i, -Y + 150 * (i % 2), CAR_Z), yaw=far)
        side = [1, 1, -1, 1, -1, 1][k]
        ball_z = [150, 150, CEILING - BALL_R, 150, 150, CEILING - BALL_R][k]
        _ball(s, (side * (X - 120), side * (Y - 120), ball_z))
        yield "corners", s, {}

    for k, v in enumerate([(1, 0, 0), (0, -1, 0), (0, 0, 1), (-1, 1, -1), (0, 0, 0), (0, -1, 0)]):
        s, cars = midplay()
        v = np.array(v, dtype=float)
        v = v / max(np.linalg.norm(v), 1)
        spin = v if k != 5 else np.array([-1.0, 0, 0])
        _ball(s, (U(-2000, 2000), U(-2000, 2000), U(300, 1500)), vel=v * 6000, ang=spin * 6)
        if k == 4:  # two cars head-on at top speed, spinning opposite ways
            _car(cars[b0], (0, -1500, CAR_Z), yaw=PI / 2, speed=2300, ang=(0, 0, 5.5), boost=100)
            _car(cars[o0], (0, 1500, CAR_Z), yaw=-PI / 2, speed=2300, ang=(0, 0, -5.5), boost=100)
        elif k == 5:  # every car supersonic against the ball
            for a in ids:
                _car(cars[a], (U(-3000, 3000), U(-4000, 0), CAR_Z), yaw=PI / 2, speed=2300, boost=100)
        else:  # one car at top speed straight against the ball
            c, flat = cars[ids[k]], v[2] == 0
            _car(c, (*c.physics.position[:2], CAR_Z if flat else 900), vel=-v * 2300, ground=flat, boost=100)
            c.has_jumped = not flat
        yield "max_speed", s, {}

    kick = copy.deepcopy(base)
    yield "match", copy.deepcopy(kick), {"match": dict(new_match(2, 2, 0.0), overtime=True)}
    for m, air in ((dict(new_match(0, 0, 0.0), overtime=True), False), (new_match(2, 1, 0.0), True),
                   (new_match(0, 2, 0.0), True), (new_match(3, 0, 150.0), False), (new_match(1, 4, 12.0), False),
                   (new_match(5, 0, 299.9), False), (new_match(0, 7, 0.5), False)):
        s, cars = midplay()
        if air:  # 0:00 and the ball hasn't landed yet
            _ball(s, (U(-2000, 2000), U(-3000, 3000), 800), vel=(U(-500, 500), U(-500, 500), 600))
        yield "match", s, {"match": m}

    def demo(cars, victim, by=None, timer=3.0):
        cars[victim].demo_respawn_timer = timer
        if by is not None:
            cars[by].bump_victim_id = victim

    for k in range(8):
        s, cars = midplay()
        over = {}
        if k == 0:  # both opponents (from blue's side), just now
            demo(cars, o0, b0), demo(cars, o1, b1, timer=1.2)
        elif k == 1:  # both blue cars
            demo(cars, b0, o0), demo(cars, b1, o1, timer=0.3)
        elif k == 2:  # blue-0's teammate, just now
            demo(cars, b1, o1)
        elif k == 3:  # orange-0's teammate, demoed earlier and still out
            demo(cars, o1, timer=0.4)
            over["old_demos"] = [o1]
        elif k == 4:
            demo(cars, b0, o0), demo(cars, b1, o0, timer=2.0), demo(cars, o0, b1, timer=1.0), demo(cars, o1, timer=0.1)
        elif k == 5:
            demo(cars, b0, o1), demo(cars, b1, timer=2.5), demo(cars, o0, b1, timer=1.5)
            over["old_demos"] = [b1]
        elif k == 6:  # contacts that earn nothing: a teammate "victim", and an opponent bumped but not demoed
            demo(cars, b1, b0)
            cars[o0].bump_victim_id = b0
        else:  # both opponents out since earlier, the other team sharing a touch
            demo(cars, o0, timer=2.0), demo(cars, o1, timer=1.0)
            over["old_demos"] = [o0, o1]
            cars[b0].ball_touches = cars[b1].ball_touches = 1
        yield "demos", s, over

    s = copy.deepcopy(kick)
    _pads(s, np.zeros(34))
    yield "kickoff", s, {"prev": {}}
    s = copy.deepcopy(kick)
    _pads(s, np.full(34, 10.0))
    for car in s.cars.values():
        car.boost_amount = 0.0
    yield "kickoff", s, {"prev": {a: np.array([-1, -1, -1, -1, -1, 0, 0, 0], dtype=np.float32) for a in ids}}
    s, cars = midplay()
    _pads(s, np.full(34, 10.0))
    for car in cars.values():
        car.boost_amount = 100.0
    yield "kickoff", s, {"prev": {a: np.ones(8, dtype=np.float32) for a in ids}}
    s, cars = midplay()
    _pads(s, np.arange(34) % 2 * 4.0)
    for i, car in enumerate(cars.values()):
        car.boost_amount = 100.0 * (i % 2)
    yield "kickoff", s, {}

    for touching in ((b0, b1), (o1,), ids, (b0, o0)):
        s, cars = midplay()
        for a in touching:
            cars[a].ball_touches = 2 if a == o1 else 1
        yield "touches", s, {}


def _profile_before(ids, rng, old_demos):
    """A plausible mid-match tracker state; demo rates include a burst so the soft cap is exercised."""
    values, touch = {}, {}
    for a in ids:
        den = rng.uniform(0, 0.03)
        num = den * rng.uniform(0, 1)
        v = rng.uniform(0, 1, 6)
        v[4] = rng.uniform(0, 0.5) if rng.random() < 0.8 else rng.uniform(1, 4)
        v[5] = (num + 0.5 * profile_v1.TOUCH_PRIOR) / (den + profile_v1.TOUCH_PRIOR)
        values[a], touch[a] = v, np.array([num, den])
    return {"values": values, "touch": touch, "demoed": {a: a in old_demos for a in ids}}


def hand_rows(seed=0):
    rng = np.random.default_rng(seed)
    env = make_env(seed=seed)
    env.reset()
    base = copy.deepcopy(env.state)
    ids = sorted(base.cars)
    table = LookupTableAction.make_lookup_table().astype(np.float32)
    builder, tracker, rows = EntityObs(), profile_v1.ProfileTracker(), []
    for name, state, over in _scenarios(base, rng):
        ttb, slices = ttb_v1.ttb_from_state(state, ttb_v1.BallPredictor())  # fresh predictor: no cache between states
        before = _profile_before(ids, rng, over.get("old_demos", ()))
        tracker.resume(copy.deepcopy(before))
        match = over.get("match") or new_match(int(rng.integers(0, 3)), int(rng.integers(0, 3)), rng.uniform(1, 300))
        info = {
            "obs_ttb": ttb, "obs_ball_slices": np.array(slices), "kickoff": is_kickoff(state), "match": match,
            "prev_action": over.get("prev", {a: table[rng.integers(len(table))] for a in ids}),
            "profile": copy.deepcopy(tracker.update(state, ttb, DT)),
        }
        rows.append(dict(name=name, state=state, start=False, info=info, profile_before=before, dt=DT,
                         obs=builder.build_obs(ids, state, info)))
    return rows
