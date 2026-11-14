"""FROZEN teammate profile, version 1. Observation inputs only; never edit, add profile_v2.py instead.

Each car gets running averages (~25 s) of how it plays; every agent's observation carries its teammate's.
Averages start from WARMUP at the start of an episode only. They carry over goals and kickoffs, because
reading the teammate across the whole match is the point.

Per step, each car contributes an indicator per feature, or NaN to leave that average untouched:
  behind_ball   goal-side of the ball (NaN while demoed)
  last_man      deeper than its teammate; 1 if the teammate is demoed, 0 if it is (NaN if both are)
  speed         speed / 2300 (NaN while demoed)
  challenge     lower frozen TTB than its teammate; 0 if demoed and the teammate isn't;
                NaN on exact ties or when both are unreachable (no hysteresis, flicker averages out)
  demo_rate     demos made this step, as demos per minute / DEMO_SCALE
  touch_share   1 if it touched the ball and its teammate didn't, 0 the reverse, 0.5 both; NaN if neither
WARMUP comes from scripted normal-vs-normal games (scripts/profile_warmup.py). Pair-symmetric features
(last_man, challenge, touch_share) average exactly 0.5 over any population.
"""

import numpy as np

VERSION = "profile-v1"
FEATURES = ("behind_ball", "last_man", "speed", "challenge", "demo_rate", "touch_share")
EMA_SECONDS = 25.0
MAX_SPEED = 2300.0
DEMO_SCALE = 2.0  # demos per minute that reads as 1.0
WARMUP = np.array([0.725, 0.500, 0.425, 0.500, 0.013, 0.500])  # 10 sim-minutes, seed 0


def _teammate(agent, state):
    team = state.cars[agent].team_num
    return next((a for a, c in state.cars.items() if a != agent and c.team_num == team), None)


def indicators(state, ttb, newly_demoed, dt):
    """agent -> indicator per feature (NaN = no information this step)."""
    ball_y = state.ball.position[1]
    out = {}
    for agent, car in state.cars.items():
        x = np.full(len(FEATURES), np.nan)
        mate_id = _teammate(agent, state)
        mate = state.cars.get(mate_id)
        sign = 1.0 if car.team_num == 0 else -1.0  # own goal at y = -5120 * sign
        y = car.physics.position[1] * sign
        if not car.is_demoed:
            x[0] = float(y < ball_y * sign)
            x[2] = min(float(np.linalg.norm(car.physics.linear_velocity)) / MAX_SPEED, 1.0)
        if mate is not None:
            if car.is_demoed != mate.is_demoed:
                x[1] = float(mate.is_demoed)
                x[3] = float(mate.is_demoed)
            elif not car.is_demoed:
                x[1] = float(y < mate.physics.position[1] * sign)
                t, tm = ttb[agent], ttb[mate_id]
                if t != tm and (np.isfinite(t) or np.isfinite(tm)):
                    x[3] = float(t < tm)
            touched, mate_touched = car.ball_touches > 0, mate.ball_touches > 0
            if touched or mate_touched:
                x[5] = 0.5 if touched and mate_touched else float(touched)
        demos = sum(1 for v in newly_demoed if car.bump_victim_id == v and state.cars[v].team_num != car.team_num)
        x[4] = demos * 60.0 / dt / DEMO_SCALE if dt > 0 else 0.0
        out[agent] = x
    return out


class ProfileTracker:
    def reset(self, state):
        self.values = {a: WARMUP.copy() for a in state.cars}
        self._demoed = {a: c.is_demoed for a, c in state.cars.items()}
        return self.values

    def update(self, state, ttb, dt):
        newly = [a for a, c in state.cars.items() if c.is_demoed and not self._demoed.get(a, False)]
        self._demoed = {a: c.is_demoed for a, c in state.cars.items()}
        alpha = 1.0 - np.exp(-dt / EMA_SECONDS)
        for agent, x in indicators(state, ttb, newly, dt).items():
            v = self.values.setdefault(agent, WARMUP.copy())
            seen = ~np.isnan(x)
            v[seen] += alpha * (x[seen] - v[seen])
        return self.values
