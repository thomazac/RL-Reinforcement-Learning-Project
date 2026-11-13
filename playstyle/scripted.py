"""Scripted 2v2 bot for exercising roles and the eval harness before any policy is trained."""

import numpy as np
from rlgym.rocket_league.action_parsers import LookupTableAction

from .ball_prediction import SLICE_DT
from .roles import FIRST

GOAL_Y = 5120.0
_TABLE = LookupTableAction.make_lookup_table()
# Ground rows are [throttle, steer, 0, steer, 0, 0, boost, handbrake].
_GROUND = {(r[0], r[1], r[6], r[7]): i for i, r in enumerate(_TABLE) if r[2] == r[4] == r[5] == 0}
GROUND_TURN_RATE = 2.0  # rad/s; roughly constant above ~500 uu/s, so turn radius ~ speed / 2


class RoleBot:
    """First man drives at the predicted intercept; second man holds a spot between ball and own net."""

    def __init__(self, noise=0.1, seed=None):
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def act(self, state, shared_info):
        roles, ttb, slices = shared_info["roles"], shared_info["ttb"], shared_info["ball_slices"]
        actions = {}
        for agent, car in state.cars.items():
            if self.rng.random() < self.noise:
                actions[agent] = np.array([self.rng.integers(len(_TABLE))])
                continue
            if roles[agent] == FIRST:
                k = len(slices) - 1 if not np.isfinite(ttb[agent]) else min(round(ttb[agent] / SLICE_DT), len(slices) - 1)
                target = slices[k]
            else:
                goal = np.array([0.0, -GOAL_Y if car.team_num == 0 else GOAL_Y, 0.0])
                target = goal + 0.35 * (state.ball.position - goal)
            actions[agent] = np.array([self._drive(car, target, boost=roles[agent] == FIRST)])
        return actions

    @staticmethod
    def _drive(car, target, boost):
        to = target[:2] - car.physics.position[:2]
        fwd = car.physics.forward[:2]
        angle = np.arctan2(fwd[0] * to[1] - fwd[1] * to[0], fwd @ to)
        steer = 0 if abs(angle) < 0.1 else (1 if angle > 0 else -1)
        # A target inside the turning circle cannot be reached by steering alone; slide instead of orbiting.
        radius = np.linalg.norm(car.physics.linear_velocity[:2]) / GROUND_TURN_RATE
        side = np.array([-fwd[1], fwd[0]]) * (1 if angle > 0 else -1)
        inside = steer != 0 and np.linalg.norm(to - side * radius) < radius
        if inside:
            return _GROUND[(0, steer, 0, 1)]
        use_boost = int(boost and abs(angle) < 0.3)
        handbrake = int(abs(angle) > 1.8)
        return _GROUND[(1, steer, use_boost, handbrake)]
