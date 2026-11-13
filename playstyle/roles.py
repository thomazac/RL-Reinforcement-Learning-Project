"""First/second-man role assignment from TTB, with hysteresis."""

from dataclasses import dataclass

import numpy as np

MARGIN = 0.2  # challenger must be this much faster (s) to take first man
MIN_HOLD = 0.4  # seconds the first man keeps the role before a swap is allowed
COMMIT_TTB = 0.3  # first man with TTB below this, and closing, is locked in
AWAY_SPEED = 300.0  # first man moving away from the ball faster than this releases the role
BACK_WALL_Y = 5120.0

FIRST, SECOND = "first", "second"


@dataclass
class TeamRoles:
    first: object = None
    hold: float = 0.0
    locked: bool = False


def is_kickoff(state) -> bool:
    b = state.ball
    return float(np.linalg.norm(b.position[:2])) < 1 and float(np.linalg.norm(b.linear_velocity)) < 1


def _depth(car) -> float:
    """Distance from the car to its own back wall; smaller is deeper."""
    y = car.physics.position[1]
    return y + BACK_WALL_Y if car.team_num == 0 else BACK_WALL_Y - y


def _toward_ball(car, ball_pos) -> float:
    d = ball_pos - car.physics.position
    return float(car.physics.linear_velocity @ d / max(np.linalg.norm(d), 1e-6))


class RoleTracker:
    def __init__(self):
        self.teams: dict[int, TeamRoles] = {}

    def reset(self, state, ttb):
        self.teams = {}
        return self.update(state, ttb, 0.0)

    def update(self, state, ttb, dt) -> dict:
        roles = {}
        for team in (0, 1):
            ids = [i for i, c in state.cars.items() if c.team_num == team]
            if not ids:
                continue
            tr = self.teams.setdefault(team, TeamRoles())
            first = self._update_team(tr, ids, state, ttb, dt)
            roles.update({i: FIRST if i == first else SECOND for i in ids})
        return roles

    def _update_team(self, tr, ids, state, ttb, dt):
        cars = state.cars
        ball = state.ball.position
        tr.hold += dt
        if len(ids) == 1:
            return self._set(tr, ids[0])

        if is_kickoff(state):
            tr.locked = True
            return self._set(tr, self._kickoff_taker(ids, state))

        f = tr.first if tr.first in ids else None
        if f is None or cars[f].ball_touches > 0 or cars[f].is_demoed or _toward_ball(cars[f], ball) < -AWAY_SPEED:
            tr.locked = False
            return self._set(tr, self._best(ids, state, ttb))

        if not tr.locked:
            c = cars[f]
            jumping_in = not c.on_ground and c.has_jumped and _toward_ball(c, ball) > 0
            tr.locked = jumping_in or (ttb[f] < COMMIT_TTB and _toward_ball(c, ball) > 0)
        if tr.locked or tr.hold < MIN_HOLD:
            return f

        challenger = next(i for i in ids if i != f)
        if cars[challenger].is_demoed:
            return f
        if ttb[challenger] < ttb[f] - MARGIN:
            return self._set(tr, challenger)
        if np.isinf(ttb[f]) and np.isinf(ttb[challenger]):
            return self._set(tr, self._best(ids, state, ttb))
        return f

    @staticmethod
    def _set(tr, first):
        if first != tr.first:
            tr.first, tr.hold = first, 0.0
        return first

    @staticmethod
    def _best(ids, state, ttb):
        """Lowest TTB; if nobody can reach the ball, the deeper car stays second."""
        alive = [i for i in ids if not state.cars[i].is_demoed] or ids
        reachable = [i for i in alive if np.isfinite(ttb[i])]
        if reachable:
            return min(reachable, key=lambda i: ttb[i])
        return max(alive, key=lambda i: _depth(state.cars[i]))

    @staticmethod
    def _kickoff_taker(ids, state):
        """Closest car goes; on a tie, the car on its team's left."""
        def key(i):
            c = state.cars[i]
            dist = round(float(np.linalg.norm(c.physics.position[:2])))
            left = c.physics.position[0] if c.team_num == 0 else -c.physics.position[0]
            return dist, left
        return min(ids, key=key)
