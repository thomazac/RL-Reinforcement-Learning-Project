"""FROZEN time-to-ball estimator and ball prediction, version 1. Observation inputs only.

Policies trained on these inputs break if they change, so this module never changes. Improvements go in
playstyle/ttb.py and playstyle/ball_prediction.py (reward side) or into a new ttb_v2.py with a new version.
It imports nothing from this package, so edits elsewhere can't leak in; RocketSim is pinned in requirements.txt.
tests/test_ttb_v1.py checks it against saved outputs in tests/data/ttb_v1_reference.npz.
"""

import os

import numpy as np
import RocketSim as rs
import rlgym.rocket_league.sim.rocketsim_engine as _rlgym_engine

VERSION = "ttb-v1"

TICK_RATE = 120
SLICE_DT = 0.1
HORIZON = 3.0
SLICE_TIMES = np.arange(0, HORIZON + 1e-6, SLICE_DT)
CACHE_TICKS = 4
CACHE_SECONDS = 6.0
POS_TOLERANCE = 0.1
VEL_TOLERANCE = 0.5

CONTACT_DIST = 200.0
TURN_TIME_LOW = 0.25
TURN_TIME_HIGH = 0.40
AIR_TURN_TIME = 0.18
GROUND_TURN_RATE = 2.0
INSIDE_DECEL = 6000.0
GROUND_REACH_DZ = 110.0
JUMP_REACH_DZ = 360.0
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

_initialized = False


def _ensure_init():
    global _initialized
    if not _initialized:
        try:
            rs.init(os.path.join(os.path.dirname(_rlgym_engine.__file__), "collision_meshes"))
        except RuntimeError as e:
            if "Already inited" not in str(e):
                raise
        _initialized = True


class BallPredictor:
    """Ball positions at SLICE_TIMES through a ball-only arena, cached until the real ball leaves the path."""

    def __init__(self):
        _ensure_init()
        self._arena = rs.Arena(rs.GameMode.SOCCAR)
        self._state = rs.BallState()
        self._stride = round(SLICE_DT * TICK_RATE) // CACHE_TICKS
        self._n_out = (len(SLICE_TIMES) - 1) * self._stride + 1
        self._cache_pos = self._cache_vel = None
        self._cache_tick = 0

    def predict(self, position, velocity, angular_velocity, tick_count=None) -> np.ndarray:
        if tick_count is not None and self._cache_pos is not None:
            offset, rem = divmod(tick_count - self._cache_tick, CACHE_TICKS)
            if rem == 0 and 0 <= offset and offset + self._n_out <= len(self._cache_pos) \
                    and np.linalg.norm(self._cache_pos[offset] - position) < POS_TOLERANCE \
                    and np.linalg.norm(self._cache_vel[offset] - velocity) < VEL_TOLERANCE:
                return self._cache_pos[offset:offset + self._n_out:self._stride]
        s = self._state
        s.pos = rs.Vec(*map(float, position))
        if not np.any(velocity):
            velocity = (0.0, 0.0, -1e-3)
        s.vel = rs.Vec(*map(float, velocity))
        s.ang_vel = rs.Vec(*map(float, angular_velocity))
        self._arena.ball.set_state(s)
        n = round(CACHE_SECONDS * TICK_RATE / CACHE_TICKS) + 1
        pred = self._arena.get_ball_prediction(n, CACHE_TICKS)
        self._cache_pos = np.array([(b.pos.x, b.pos.y, b.pos.z) for b in pred], dtype=np.float32)
        self._cache_vel = np.array([(b.vel.x, b.vel.y, b.vel.z) for b in pred], dtype=np.float32)
        self._cache_tick = tick_count if tick_count is not None else 0
        return self._cache_pos[:self._n_out:self._stride]


def _build_distance_table() -> np.ndarray:
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


def estimate_ttb(car_pos, car_vel, car_fwd, boost, demoed, ball_slices, on_ground) -> np.ndarray:
    car_pos, car_vel, car_fwd = (np.asarray(x, dtype=np.float64) for x in (car_pos, car_vel, car_fwd))
    grounded = np.asarray(on_ground, dtype=bool)[:, None]
    boost = np.asarray(boost, dtype=np.float64)
    ball = np.asarray(ball_slices, dtype=np.float64)
    if ball.ndim == 2:
        ball = ball[None]

    to_ball = ball[..., :2] - car_pos[:, None, :2]
    dist = np.linalg.norm(to_ball, axis=-1)
    need = np.maximum(dist - CONTACT_DIST, 0.0)
    direction = to_ball / np.maximum(dist, 1e-6)[..., None]

    fwd = car_fwd[:, None, :2]
    fwd = fwd / np.maximum(np.linalg.norm(fwd, axis=-1, keepdims=True), 1e-6)
    angle = np.arccos(np.clip((direction * fwd).sum(-1), -1.0, 1.0))
    angle = np.where(need > 0, angle, 0.0)
    v_toward = (car_vel[:, None, :2] * direction).sum(-1)

    dz = ball[..., 2] - car_pos[:, None, 2]
    rise_time = np.maximum(dz - JUMP_REACH_DZ, 0.0) / AERIAL_RISE_SPEED
    jump_time = np.where(dz > GROUND_REACH_DZ, JUMP_TIME, 0.0)
    boost_needed = np.where(dz > JUMP_REACH_DZ, MIN_AERIAL_BOOST + rise_time * BOOST_PER_SEC, 0.0)

    speed = np.linalg.norm(car_vel[:, :2], axis=1, keepdims=True)
    v0 = np.clip(np.where(grounded, np.maximum(v_toward, speed * np.cos(angle / 2)), v_toward), 0.0, MAX_SPEED)
    speed_frac = np.minimum(speed / MAX_SPEED, 1.0)
    ground_turn = TURN_TIME_LOW + (TURN_TIME_HIGH - TURN_TIME_LOW) * speed_frac * angle / np.pi
    turn_time_per_rad = np.where(grounded, ground_turn, AIR_TURN_TIME)

    cross = fwd[..., 0] * direction[..., 1] - fwd[..., 1] * direction[..., 0]
    side = np.stack([-fwd[..., 1], fwd[..., 0]], axis=-1) * np.where(cross >= 0, 1.0, -1.0)[..., None]
    radius = speed / GROUND_TURN_RATE
    inside = grounded & (angle > 0.1) & (np.linalg.norm(to_ball - side * radius[..., None], axis=-1) < radius)
    slide_time = np.where(inside, speed / INSIDE_DECEL, 0.0)

    tau = SLICE_TIMES[None, :] - turn_time_per_rad * angle - slide_time - jump_time
    vi = np.clip(np.rint(v0 / _V_STEP).astype(int), 0, len(_V_GRID) - 1)
    bi = np.clip(np.rint(boost / _B_STEP).astype(int), 0, len(_B_GRID) - 1)[:, None]
    ti = np.clip(np.rint(tau * TICK_RATE).astype(int), 0, _N_TICKS)
    reach = _DISTANCE[vi, bi, ti]

    horizontal = np.where(tau >= 0, reach - need, tau * 1000.0 - need)
    margin = np.minimum(horizontal, (tau - rise_time) * 1000.0)
    margin = np.where(boost[:, None] < boost_needed, -np.inf, margin)

    ok = margin >= 0
    first = ok.argmax(axis=1)
    rows = np.arange(len(car_pos))
    prev = np.maximum(first - 1, 0)
    m_prev, m_cur = margin[rows, prev], margin[rows, first]
    t_prev, t_cur = SLICE_TIMES[prev], SLICE_TIMES[first]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(np.isfinite(m_prev), -m_prev / (m_cur - m_prev), 1.0)
        ttb = np.where(first == 0, SLICE_TIMES[0], t_prev + (t_cur - t_prev) * frac)
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
