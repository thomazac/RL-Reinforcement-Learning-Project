import os

import numpy as np
import RocketSim as rs
import rlgym.rocket_league.sim.rocketsim_engine as _rlgym_engine

TICK_RATE = 120
SLICE_DT = 0.1
HORIZON = 3.0
SLICE_TIMES = np.arange(0, HORIZON + 1e-6, SLICE_DT)

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
    """

    def __init__(self):
        _ensure_init()
        self._arena = rs.Arena(rs.GameMode.SOCCAR)
        self._state = rs.BallState()
        self._interval = round(SLICE_DT * TICK_RATE)

    def predict(self, position, velocity, angular_velocity) -> np.ndarray:
        """Ball positions at SLICE_TIMES, shape (len(SLICE_TIMES), 3)."""
        s = self._state
        s.pos = rs.Vec(*map(float, position))
        # An exactly-zero velocity makes RocketSim skip simulation, so a ball set in midair would float.
        if not np.any(velocity):
            velocity = (0.0, 0.0, -1e-3)
        s.vel = rs.Vec(*map(float, velocity))
        s.ang_vel = rs.Vec(*map(float, angular_velocity))
        self._arena.ball.set_state(s)
        pred = self._arena.get_ball_prediction(len(SLICE_TIMES), self._interval)
        return np.array([(b.pos.x, b.pos.y, b.pos.z) for b in pred], dtype=np.float32)
