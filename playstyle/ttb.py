"""Time-to-ball (TTB) estimate for every car against the predicted ball path."""

import numpy as np

from .ball_prediction import SLICE_TIMES, TICK_RATE, BallPredictor

# Fitted on GC1 2s replays with scripts/validate_ttb.py --calibrate.
CONTACT_DIST = 200.0  # ball radius plus the car's reach, xy plane
# Turn constants come from scripts/measure_turns.py, not replays: replay touchers were mostly
# already facing the ball, which pulls a replay fit far below what the game allows.
# The scripted driver never half-flips, so its times bound real ones from above; a half-flip
# helps from rest but not at speed, so the values lean toward the measured times at speed.
TURN_TIME_LOW = 0.30  # s/rad at rest; ground turn rate is speed x curvature, so a parked car turns slowly
TURN_TIME_HIGH = 0.50  # s/rad for a full U-turn at max speed
AIR_TURN_TIME = 0.18  # s/rad; the 5.5 rad/s angular velocity cap, reachable only in the air
GROUND_TURN_RATE = 2.0  # rad/s, roughly constant above ~500 uu/s, so turn radius ~ speed / 2
INSIDE_DECEL = 2300.0  # uu/s^2 shed while sliding when the target is inside the turning circle
GROUND_REACH_DZ = 110.0  # ball height above the car hittable without jumping
JUMP_REACH_DZ = 360.0  # ball height above the car hittable with a single jump
JUMP_TIME = 0.15
AERIAL_RISE_SPEED = 1200.0
MIN_AERIAL_BOOST = 0.0

MAX_SPEED = 2300.0
BOOST_ACCEL = 991.6667
BOOST_PER_SEC = 33.3333
_THROTTLE_CURVE = ([0.0, 1400.0, 1410.0], [1600.0, 160.0, 0.0])

_V_STEP, _B_STEP = 50.0, 4.0
_V_GRID = np.arange(0.0, MAX_SPEED + 1, _V_STEP)
_B_GRID = np.arange(0.0, 100.0 + 1, _B_STEP)
_N_TICKS = round(SLICE_TIMES[-1] * TICK_RATE)


def _build_distance_table() -> np.ndarray:
    """Straight-line distance after n ticks of full throttle and boost, indexed [speed, boost, tick]."""
    v, b = (g.astype(np.float64) for g in np.meshgrid(_V_GRID, _B_GRID, indexing="ij"))
    d = np.zeros_like(v)
    dt = 1.0 / TICK_RATE
    table = np.zeros((len(_V_GRID), len(_B_GRID), _N_TICKS + 1), dtype=np.float32)
    for i in range(1, _N_TICKS + 1):
        a = np.interp(v, *_THROTTLE_CURVE) + np.where(b > 0, BOOST_ACCEL, 0.0)
        v = np.minimum(v + a * dt, MAX_SPEED)
        b = np.maximum(b - BOOST_PER_SEC * dt, 0.0)
        d += v * dt
        table[..., i] = d
    return table


_DISTANCE = _build_distance_table()


def estimate_ttb(car_pos, car_vel, car_fwd, boost, demoed, ball_slices, slice_times=SLICE_TIMES,
                 on_ground=None) -> np.ndarray:
    """TTB in seconds for each car; inf if the ball is unreachable within the horizon.

    car_pos, car_vel, car_fwd: (n, 3). boost: (n,) in 0-100. demoed, on_ground: (n,) bool;
    on_ground defaults to car height below 60 uu (replays carry no wheel-contact flag).
    ball_slices: predicted ball positions at slice_times (k,), starting at t=0; either (k, 3)
    shared by all cars or (n, k, 3) per car.
    """
    car_pos, car_vel, car_fwd = (np.asarray(x, dtype=np.float64) for x in (car_pos, car_vel, car_fwd))
    grounded = (car_pos[:, 2] < 60 if on_ground is None else np.asarray(on_ground, dtype=bool))[:, None]
    boost = np.asarray(boost, dtype=np.float64)
    ball = np.asarray(ball_slices, dtype=np.float64)
    if ball.ndim == 2:
        ball = ball[None]

    to_ball = ball[..., :2] - car_pos[:, None, :2]  # (n, k, 2)
    dist = np.linalg.norm(to_ball, axis=-1)
    need = np.maximum(dist - CONTACT_DIST, 0.0)
    direction = to_ball / np.maximum(dist, 1e-6)[..., None]

    fwd = car_fwd[:, None, :2]
    fwd = fwd / np.maximum(np.linalg.norm(fwd, axis=-1, keepdims=True), 1e-6)
    angle = np.arccos(np.clip((direction * fwd).sum(-1), -1.0, 1.0))
    angle = np.where(need > 0, angle, 0.0)
    v_toward = (car_vel[:, None, :2] * direction).sum(-1)

    # Height is relative to the car, so a car already in the air or on the wall needs no rise.
    dz = ball[..., 2] - car_pos[:, None, 2]
    rise_time = np.maximum(dz - JUMP_REACH_DZ, 0.0) / AERIAL_RISE_SPEED
    jump_time = np.where(dz > GROUND_REACH_DZ, JUMP_TIME, 0.0)
    boost_needed = np.where(dz > JUMP_REACH_DZ, MIN_AERIAL_BOOST + rise_time * BOOST_PER_SEC, 0.0)

    # On the ground a car curving toward the target keeps part of its speed: all of it straight
    # ahead, none after a U-turn. In the air steering doesn't bend the velocity.
    speed = np.linalg.norm(car_vel[:, :2], axis=1, keepdims=True)
    v0 = np.clip(np.where(grounded, np.maximum(v_toward, speed * np.cos(angle / 2)), v_toward), 0.0, MAX_SPEED)
    speed_frac = np.minimum(speed / MAX_SPEED, 1.0)
    ground_turn = TURN_TIME_LOW + (TURN_TIME_HIGH - TURN_TIME_LOW) * speed_frac * angle / np.pi
    turn_time_per_rad = np.where(grounded, ground_turn, AIR_TURN_TIME)

    # A target inside the turning circle can't be reached by steering; the car must shed speed first.
    cross = fwd[..., 0] * direction[..., 1] - fwd[..., 1] * direction[..., 0]
    side = np.stack([-fwd[..., 1], fwd[..., 0]], axis=-1) * np.where(cross >= 0, 1.0, -1.0)[..., None]
    radius = speed / GROUND_TURN_RATE
    inside = grounded & (angle > 0.1) & (np.linalg.norm(to_ball - side * radius[..., None], axis=-1) < radius)
    slide_time = np.where(inside, speed / INSIDE_DECEL, 0.0)

    # Driving and rising overlap: both must be finished by the slice time.
    tau = slice_times[None, :] - turn_time_per_rad * angle - slide_time - jump_time
    vi = np.clip(np.rint(v0 / _V_STEP).astype(int), 0, len(_V_GRID) - 1)
    bi = np.clip(np.rint(boost / _B_STEP).astype(int), 0, len(_B_GRID) - 1)[:, None]
    ti = np.clip(np.rint(tau * TICK_RATE).astype(int), 0, _N_TICKS)
    reach = _DISTANCE[vi, bi, ti]

    # Negative times are scaled into distance so margin crosses zero exactly when the car is ready.
    horizontal = np.where(tau >= 0, reach - need, tau * 1000.0 - need)
    margin = np.minimum(horizontal, (tau - rise_time) * 1000.0)
    margin = np.where(boost[:, None] < boost_needed, -np.inf, margin)

    ok = margin >= 0
    first = ok.argmax(axis=1)
    rows = np.arange(len(car_pos))
    prev = np.maximum(first - 1, 0)
    m_prev, m_cur = margin[rows, prev], margin[rows, first]
    t_prev, t_cur = slice_times[prev], slice_times[first]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(np.isfinite(m_prev), -m_prev / (m_cur - m_prev), 1.0)
        ttb = np.where(first == 0, slice_times[0], t_prev + (t_cur - t_prev) * frac)
    return np.where(ok.any(axis=1) & ~np.asarray(demoed, dtype=bool), ttb, np.inf)


def ttb_from_state(state, predictor: BallPredictor):
    """TTB per agent id from an RLGym v2 GameState, plus the ball slices used."""
    ids = list(state.cars)
    cars = [state.cars[i] for i in ids]
    ball = state.ball
    slices = predictor.predict(ball.position, ball.linear_velocity, ball.angular_velocity, state.tick_count)
    ttb = estimate_ttb(
        np.array([c.physics.position for c in cars]),
        np.array([c.physics.linear_velocity for c in cars]),
        np.array([c.physics.forward for c in cars]),
        np.array([c.boost_amount for c in cars]),
        np.array([c.is_demoed for c in cars]),
        slices,
        on_ground=np.array([c.on_ground for c in cars]),
    )
    return dict(zip(ids, ttb)), slices
