"""Coordination-layer events: double-commit, nobody challenging, nobody home.

Computed once per tick next to TTB and roles so the reward and the overlay see the same events.
"""

import numpy as np

from .ball_prediction import SLICE_TIMES
from .roles import FIRST, SECOND, is_kickoff

# Loosened from 0.6 s / 500 uu/s after 900M Phase 1b steps left crowding well above human (1.60 cars within 1,000 uu
# of the ball against 1.04; all four 7.8% of the time against 0.3%) while double-commits stayed rare (0.8%).
DOUBLE_COMMIT_TTB = 0.8  # both teammates below this TTB...
CLOSING_SPEED = 300.0  # ...and both closing on the ball faster than this (uu/s)
OWN_NET_HORIZON = 1.0  # double-commit is allowed if the ball is predicted into the own net this soon (s)
# Demo play excuses a double-commit when a teammate is aimed at an opponent rather than the ball:
# either it has recently aimed at opponents more than at its ball intercept, or it is aimed tightly
# at an opponent it will reach before the ball (ball and opponent in line, so it bumps first).
DEMO_AIM_COS = 0.9
DEMO_AIM_MARGIN = np.radians(5)  # average (intercept angle - opponent angle) must exceed this
AIM_MEMORY = 0.5  # s, time constant of that average
IN_LINE_COS = 0.97  # ~14 degrees
# The demo excuse is judged on outcome: if a car whose aim earned it touches the ball this soon after the
# last excused step, every excused step is charged to that car. Charging the second man instead would
# teach him not to go for the ball while his teammate hunts nearby.
DEMO_EXCUSE_WINDOW = 0.5
POSSESSION_DIST = 300.0  # ball within this of the last toucher counts as controlled possession
NOBODY_CHALLENGING_TIME = 1.5  # s of opponent possession before "nobody challenging" fires
CHALLENGE_SPEED = 300.0  # a teammate closing faster than this counts as challenging
# Shadowing also counts as pressure: a teammate goal-side of the ball, near the line from the ball to its own net,
# within this distance band of the ball, and retreating at least about as fast as the ball comes toward the net.
# Without it the penalty charges a defender for holding position and pushes him into early challenges.
SHADOW_DISTANCE = (300.0, 3000.0)
SHADOW_LATERAL = 900.0  # uu from the ball-to-goal line
SHADOW_PACE_SLACK = 400.0  # uu/s the defender may retreat slower than the ball approaches
GOAL_Y, GOAL_HALF_W, GOAL_H = 5120.0, 893.0, 642.0
OWN_THIRD_Y = GOAL_Y / 3


def _unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-6)


def _closing(car, target) -> float:
    return float(car.physics.linear_velocity @ _unit(target - car.physics.position))


def _own_goal_sign(team) -> float:
    return -1.0 if team == 0 else 1.0


def _shadowing(car, ball, ball_vel, team) -> bool:
    """Goal-side of the ball, near its line to the own net, in the distance band, keeping pace with it."""
    if car.is_demoed:
        return False
    goal = np.array([0.0, _own_goal_sign(team) * GOAL_Y, 0.0])
    pos = car.physics.position
    to_goal = goal[:2] - ball[:2]
    length = float(np.linalg.norm(to_goal))
    direction = to_goal / max(length, 1e-6)
    along = float((pos[:2] - ball[:2]) @ direction)  # how far toward the own net from the ball
    lateral = abs(float(np.cross(direction, pos[:2] - ball[:2])))
    distance = float(np.linalg.norm(pos - ball))
    if not (0 < along < length and lateral < SHADOW_LATERAL and SHADOW_DISTANCE[0] <= distance <= SHADOW_DISTANCE[1]):
        return False
    approach = float(np.asarray(ball_vel)[:2] @ direction)  # ball speed toward the own net
    retreat = float(car.physics.linear_velocity[:2] @ direction)
    return approach <= 0 or retreat >= approach - SHADOW_PACE_SLACK


class CoordinationTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_toucher, self.possession_time = None, 0.0
        self.aim_preference = {}  # agent -> running average of how much more it aims at an opponent than the ball
        self.pending = {}  # team -> {"hunters": set, "steps": excused steps, "left": s}

    def update(self, state, ttb, roles, slices, dt) -> dict:
        """Per team: {"double_commit": penalized agent or None, "excused": reason or None,
        "revoked": {agent: excused steps now penalized} or None,
        "nobody_challenging": penalized agent or None, "nobody_home": bool}."""
        ball = state.ball.position
        if is_kickoff(state):  # a possession never runs across a goal and the kickoff wait
            self.last_toucher, self.possession_time = None, 0.0
        touchers = [a for a, c in state.cars.items() if c.ball_touches > 0]
        if touchers:
            toucher = min(touchers, key=lambda a: np.linalg.norm(state.cars[a].physics.position - ball))
            if toucher != self.last_toucher:
                self.possession_time = 0.0
            self.last_toucher = toucher
        holder = state.cars.get(self.last_toucher)
        controlled = holder is not None and not holder.is_demoed \
            and np.linalg.norm(holder.physics.position - ball) < POSSESSION_DIST
        self.possession_time = self.possession_time + dt if controlled else 0.0
        aim = self._aim_all(state, ttb, slices)
        alpha = min(dt / AIM_MEMORY, 1.0)
        for a, (_, preference, _) in aim.items():
            prev = self.aim_preference.get(a, 0.0)
            self.aim_preference[a] = prev + alpha * (preference - prev)

        events = {}
        for team in (0, 1):
            ids = [a for a, c in state.cars.items() if c.team_num == team]
            if not ids:
                continue
            ev = {"double_commit": None, "excused": None, "revoked": None,
                  "nobody_challenging": None, "nobody_home": False}
            first = next((a for a in ids if roles[a] == FIRST), None)
            second = next((a for a in ids if roles[a] == SECOND), None)

            if first is not None and second is not None:
                both = all(ttb[a] < DOUBLE_COMMIT_TTB and _closing(state.cars[a], ball) > CLOSING_SPEED
                           for a in (first, second))
                if both:
                    ev["excused"], hunters = self._double_commit_excuse(
                        team, slices, {a: (aim[a][0], self.aim_preference[a], aim[a][2]) for a in (first, second)})
                    if ev["excused"] is None:
                        ev["double_commit"] = second
                    elif ev["excused"] == "demo":
                        p = self.pending.setdefault(team, {"hunters": set(), "steps": 0})
                        p["hunters"].update(hunters)
                        p["steps"] += 1
                        p["left"] = DEMO_EXCUSE_WINDOW + dt
            p = self.pending.get(team)
            if p is not None:
                p["left"] -= dt
                broke = [a for a in touchers if a in p["hunters"]]
                if broke:
                    ev["revoked"] = {broke[0]: p["steps"]}
                    del self.pending[team]
                elif p["left"] <= 0:
                    del self.pending[team]

            if controlled and holder.team_num != team and self.possession_time > NOBODY_CHALLENGING_TIME \
                    and first is not None \
                    and not any(_closing(state.cars[a], ball) > CHALLENGE_SPEED
                                or _shadowing(state.cars[a], ball, state.ball.linear_velocity, team) for a in ids):
                ev["nobody_challenging"] = first

            sign = _own_goal_sign(team)
            if ball[1] * sign > OWN_THIRD_Y:
                alive = [state.cars[a] for a in ids if not state.cars[a].is_demoed]
                ev["nobody_home"] = bool(alive) and all(c.physics.position[1] * sign < ball[1] * sign for c in alive)
            events[team] = ev
        return events

    @staticmethod
    def _aim_all(state, ttb, slices):
        """Per agent, from its velocity: (cosine to the best-aimed opponent, ball-intercept angle minus that
        opponent's angle, whether that opponent is nearer than the intercept). A chaser aims at the
        intercept rather than the ball's current position, so that is what the opponent is compared to."""
        ids = list(state.cars)
        cars = [state.cars[a] for a in ids]
        # Ground plane only: in 3D, a ball 76 uu above the car reads as several degrees off a car driving straight at it.
        pos = np.array([c.physics.position[:2] for c in cars], dtype=np.float64)
        vel = np.array([c.physics.linear_velocity[:2] for c in cars], dtype=np.float64)
        team = np.array([c.team_num for c in cars])
        demoed = np.array([c.is_demoed for c in cars])
        speed = np.linalg.norm(vel, axis=1)
        heading = vel / np.maximum(speed, 1e-6)[:, None]

        t = np.array([ttb[a] for a in ids], dtype=np.float64)
        k = np.where(np.isfinite(t), np.minimum(np.searchsorted(SLICE_TIMES, np.nan_to_num(t, posinf=0)), len(slices) - 1), 0)
        to_intercept = slices[k, :2] - pos
        intercept_dist = np.linalg.norm(to_intercept, axis=1)
        ball_angle = np.arccos(np.clip((heading * to_intercept).sum(1) / np.maximum(intercept_dist, 1e-6), -1, 1))

        offset = pos[None, :, :] - pos[:, None, :]  # [i, j] = car j relative to car i
        dist = np.linalg.norm(offset, axis=2)
        cos = np.einsum("ik,ijk->ij", heading, offset) / np.maximum(dist, 1e-6)
        is_opp = (team[:, None] != team[None, :]) & ~demoed[None, :]
        cos = np.where(is_opp, cos, -np.inf)
        best = cos.argmax(axis=1)
        rows = np.arange(len(ids))
        to_opp = cos[rows, best]
        valid = ~demoed & (speed >= 100) & is_opp.any(axis=1)
        preference = ball_angle - np.arccos(np.clip(to_opp, -1, 1))
        nearer = dist[rows, best] < intercept_dist
        return {a: (float(to_opp[i]), float(preference[i]), bool(nearer[i])) if valid[i] else (-1.0, 0.0, False)
                for i, a in enumerate(ids)}

    @staticmethod
    def _double_commit_excuse(team, slices, aims):
        """aims: agent -> (cosine to best-aimed opponent, aim preference, opponent nearer than intercept) for
        both teammates. Returns (reason or None, agents on a demo path). Either one on a demo path means they
        are not both going for the ball, whichever role TTB gave it; the outcome check revokes it otherwise."""
        sign = _own_goal_sign(team)
        soon = slices[SLICE_TIMES <= OWN_NET_HORIZON]
        if np.any((soon[:, 1] * sign > GOAL_Y) & (np.abs(soon[:, 0]) < GOAL_HALF_W) & (soon[:, 2] < GOAL_H)):
            return "save", []
        hunters = [a for a, (to_opp, preference, nearer) in aims.items()
                   if (to_opp > DEMO_AIM_COS and preference > DEMO_AIM_MARGIN) or (to_opp > IN_LINE_COS and nearer)]
        return ("demo" if hunters else None), hunters
