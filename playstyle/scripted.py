"""Scripted 2v2 bots for exercising the pipeline and the coordination layer before any policy is trained.

The normal bot picks its role from the same TTB and role logic the reward uses, so it can't be used to
check that logic or TTB accuracy. The other modes are deliberately bad teams that should each trigger
one coordination event:

    chaser   both drive at the ball          -> double-commit
    camper   both park in their own net      -> nobody challenging
    forward  both stay upfield               -> nobody home
    demo     one fixed car hunts the         -> must NOT count as double-commit
             opponent nearest the ball, the
             other always plays the ball
"""

from collections import defaultdict

import numpy as np
from rlgym.rocket_league.action_parsers import LookupTableAction

from .ball_prediction import SLICE_DT
from .roles import FIRST

MODES = ("normal", "chaser", "camper", "forward", "demo")
GOAL_Y = 5120.0
WALL_X, WALL_Y = 3800.0, 4800.0
_TABLE = LookupTableAction.make_lookup_table()
# Ground rows are [throttle, steer, 0, steer, 0, 0, boost, handbrake].
_GROUND = {(r[0], r[1], r[6], r[7]): i for i, r in enumerate(_TABLE) if r[2] == r[4] == r[5] == 0}
GROUND_TURN_RATE = 2.0  # rad/s; roughly constant above ~500 uu/s, so turn radius ~ speed / 2
ARRIVED = 250.0  # uu; parked targets closer than this are held by idling
STUCK_SPEED, STUCK_TIME, UNSTICK_TIME = 250.0, 1.0, 0.6


class RoleBot:
    def __init__(self, noise=0.1, seed=None, team_modes=None, dt=8 / 120):
        self.noise, self.dt = noise, dt
        self.team_modes = team_modes or {}
        self.rng = np.random.default_rng(seed)
        self.stuck_time = defaultdict(float)
        self.unstick_time = defaultdict(float)

    def act(self, state, shared_info):
        actions = {}
        for agent, car in state.cars.items():
            target, boost, park = self._target(agent, car, state, shared_info)
            if self._stuck(agent, car, target, park):
                actions[agent] = np.array([_GROUND[(-1, 0, 0, 0)]])
            elif self.rng.random() < self.noise:
                actions[agent] = np.array([self.rng.integers(len(_TABLE))])
            else:
                actions[agent] = np.array([self._drive(car, target, boost, park)])
        return actions

    def _target(self, agent, car, state, info):
        """Where to drive, whether to boost, and whether to park on arrival."""
        mode = self.team_modes.get(car.team_num, "normal")
        ball = state.ball.position
        own = -1.0 if car.team_num == 0 else 1.0
        first = info["roles"][agent] == FIRST
        goal = np.array([0.0, own * GOAL_Y, 0.0])

        if mode == "camper":
            return np.array([0.0, own * (GOAL_Y - 300), 17.0]), False, True
        if mode == "forward" and not (ball[1] * own < 0 and first):
            x = 1200.0 if agent.endswith("0") else -1200.0
            return np.array([x, -own * 2000.0, 17.0]), False, True
        if mode == "demo":
            # Independent of roles, so the role system can't make the hunter look like a ball chaser.
            opps = [c for c in state.cars.values() if c.team_num != car.team_num and not c.is_demoed]
            if agent.endswith("1") and opps:
                victim = min(opps, key=lambda c: np.linalg.norm(c.physics.position - ball))
                return victim.physics.position, True, False
            return self._intercept(agent, info), True, False
        if mode == "chaser" or first:
            return self._intercept(agent, info), True, False
        return goal + 0.35 * (ball - goal), False, False

    @staticmethod
    def _intercept(agent, info):
        ttb, slices = info["ttb"][agent], info["ball_slices"]
        k = len(slices) - 1 if not np.isfinite(ttb) else min(round(ttb / SLICE_DT), len(slices) - 1)
        return slices[k]

    def _stuck(self, agent, car, target, park):
        """Low speed near a wall for over a second while trying to go somewhere: reverse briefly."""
        if self.unstick_time[agent] > 0:
            self.unstick_time[agent] -= self.dt
            return True
        pos = car.physics.position
        near_wall = abs(pos[0]) > WALL_X or abs(pos[1]) > WALL_Y or (car.on_ground and pos[2] > 150)
        trying = not park or np.linalg.norm(target[:2] - pos[:2]) > ARRIVED
        slow = np.linalg.norm(car.physics.linear_velocity) < STUCK_SPEED
        self.stuck_time[agent] = self.stuck_time[agent] + self.dt if near_wall and trying and slow else 0.0
        if self.stuck_time[agent] > STUCK_TIME:
            self.stuck_time[agent], self.unstick_time[agent] = 0.0, UNSTICK_TIME
            return True
        return False

    @staticmethod
    def _drive(car, target, boost, park=False):
        to = target[:2] - car.physics.position[:2]
        if park and np.linalg.norm(to) < ARRIVED:
            return _GROUND[(0, 0, 0, 0)]
        fwd = car.physics.forward[:2]
        angle = np.arctan2(fwd[0] * to[1] - fwd[1] * to[0], fwd @ to)
        steer = 0 if abs(angle) < 0.1 else (1 if angle > 0 else -1)
        # A target inside the turning circle cannot be reached by steering alone; slide instead of orbiting.
        radius = np.linalg.norm(car.physics.linear_velocity[:2]) / GROUND_TURN_RATE
        side = np.array([-fwd[1], fwd[0]]) * (1 if angle > 0 else -1)
        if steer != 0 and np.linalg.norm(to - side * radius) < radius:
            return _GROUND[(0, steer, 0, 1)]
        use_boost = int(boost and abs(angle) < 0.3)
        handbrake = int(abs(angle) > 1.8)
        return _GROUND[(1, steer, use_boost, handbrake)]
