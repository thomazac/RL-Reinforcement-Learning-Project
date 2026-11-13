import os

import numpy as np
import RocketSim as rs
import rlgym.rocket_league.sim.rocketsim_engine as _rlgym_engine

TICK_RATE = 120
SLICE_DT = 0.1
HORIZON = 3.0
SLICE_TIMES = np.arange(0, HORIZON + 1e-6, SLICE_DT)

# The cache is stored every 4 ticks, which divides both the 8-tick action step and the 12-tick slice spacing.
CACHE_TICKS = 4
CACHE_SECONDS = 6.0
# Untouched balls follow the cache exactly; looser tolerances let gentle car contact through.
POS_TOLERANCE = 0.1  # uu
VEL_TOLERANCE = 0.5  # uu/s

_initialized = False


def _ensure_init():
    global _initialized
    if not _initialized:
        try:
            rs.init(os.path.join(os.path.dirname(_rlgym_engine.__file__), "collision_meshes"))
        except RuntimeError as e:  # RocketSimEngine may have initialized it already
            if "Already inited" not in str(e):
                raise
        _initialized = True


class BallPredictor:
    """Ball-only arena used for prediction.

    RocketSim 2.2.1's standalone BallPredictor ignores its input state and always
    returns a ball at rest at center field, so prediction runs through an Arena.

    With a tick count, the path is cached and reused until the real ball leaves it.
    """

    def __init__(self):
        _ensure_init()
        self._arena = rs.Arena(rs.GameMode.SOCCAR)
        self._state = rs.BallState()
        self._stride = round(SLICE_DT * TICK_RATE) // CACHE_TICKS
        self._n_out = (len(SLICE_TIMES) - 1) * self._stride + 1
        self._cache_pos = self._cache_vel = None
        self._cache_tick = 0
        self.hits = self.misses = 0

    def predict(self, position, velocity, angular_velocity, tick_count=None) -> np.ndarray:
        """Ball positions at SLICE_TIMES, shape (len(SLICE_TIMES), 3)."""
        if tick_count is not None and self._cache_pos is not None:
            offset, rem = divmod(tick_count - self._cache_tick, CACHE_TICKS)
            if rem == 0 and 0 <= offset and offset + self._n_out <= len(self._cache_pos) \
                    and np.linalg.norm(self._cache_pos[offset] - position) < POS_TOLERANCE \
                    and np.linalg.norm(self._cache_vel[offset] - velocity) < VEL_TOLERANCE:
                self.hits += 1
                return self._cache_pos[offset:offset + self._n_out:self._stride]
        self.misses += 1

        s = self._state
        s.pos = rs.Vec(*map(float, position))
        # An exactly-zero velocity makes RocketSim skip simulation, so a ball set in midair would float.
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
