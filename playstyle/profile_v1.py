"""FROZEN teammate profile, version 1. Observation inputs only; never edit, add profile_v2.py instead.

Each car gets running averages (~25 s; 90 s for demo rate) of how it plays; every agent's observation carries its teammate's.
Averages start from WARMUP at the start of an episode only. They carry over goals and kickoffs, because
reading the teammate across the whole match is the point.

Per step, each car contributes an indicator per feature, or NaN to leave that average untouched:
  behind_ball   goal-side of the ball (NaN while demoed)
  last_man      deeper than its teammate; 1 if the teammate is demoed, 0 if it is (NaN if both are)
  speed         speed / 2300 (NaN while demoed)
  challenge     lower frozen TTB than its teammate; 0 if demoed and the teammate isn't;
                NaN on exact ties or when both are unreachable (no hysteresis, flicker averages out)
  demo_rate     demos made this step, as demos per minute / DEMO_SCALE
  touch_share   share of the team's touches over the window: average of (1 if it touched and its teammate
                didn't, 0.5 both, 0 otherwise) divided by average of (team touched), so rare touches still
                count fully; starts as if one window of TEAM_TOUCH_RATE play split evenly
WARMUP comes from scripted normal-vs-normal games (scripts/profile_warmup.py), except demo_rate: scripted bots
never demo on purpose, so it uses the GC1 2s replay mean (0.13 demos/min; p90 0.34, max 0.77 over 120
player-games). Pair-symmetric features (last_man, challenge, touch_share) average exactly 0.5 over any population.
Demos are rare, so demo_rate averages over a longer window; one demo adds 60 / (90 * 0.75) ~ 0.9 and decays.
The observation reads demo_rate through 2 * tanh(x / 2) (observed()): bursts stay below 2, normal values barely move.
"""

import numpy as np

VERSION = "profile-v1"
FEATURES = ("behind_ball", "last_man", "speed", "challenge", "demo_rate", "touch_share")
EMA_SECONDS = np.array([25.0, 25.0, 25.0, 25.0, 90.0, 25.0])
MAX_SPEED = 2300.0
DEMO_SCALE = 0.75  # demos per minute that reads as 1.0: a 2x-baseline Demo Specialist reads ~0.5-1
WARMUP = np.array([0.725, 0.500, 0.425, 0.500, 0.173, 0.500])  # scripted 10 sim-minutes, seed 0; demo from replays


DEMO_SOFT_CAP = 2.0
TEAM_TOUCH_RATE = 0.0123  # share of steps with a team touch in scripted play (0.18/s); scripts/profile_warmup.py


def observed(values):
    """Profile as the observation reads it: the running averages, with demo_rate soft-capped."""
    out = np.array(values, dtype=np.float64)
    i = FEATURES.index("demo_rate")
    out[i] = DEMO_SOFT_CAP * np.tanh(out[i] / DEMO_SOFT_CAP)
    return out


def _teammate(agent, state):
    team = state.cars[agent].team_num
    return next((a for a, c in state.cars.items() if a != agent and c.team_num == team), None)


def indicators(state, ttb, newly_demoed, dt):
    """(agent -> indicator per feature, NaN = no information this step; agent -> whether its team touched).
    The touch_share slot holds the car's share of this step's team touch (0 when the team didn't touch)."""
    ball_y = state.ball.position[1]
    out, team_touch = {}, {}
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
        touched = car.ball_touches > 0
        mate_touched = mate is not None and mate.ball_touches > 0
        team_touch[agent] = float(touched or mate_touched)
        x[5] = (0.5 if mate_touched else 1.0) if touched else 0.0
        demos = sum(1 for v in newly_demoed if car.bump_victim_id == v and state.cars[v].team_num != car.team_num)
        x[4] = demos * 60.0 / dt / DEMO_SCALE if dt > 0 else 0.0
        out[agent] = x
    return out, team_touch


class ProfileTracker:
    def reset(self, state):
        self.values = {a: WARMUP.copy() for a in state.cars}
        # touch_share = average(own share of team touches) / average(team touched), both over the same window
        self._touch = {a: np.array([WARMUP[5] * TEAM_TOUCH_RATE, TEAM_TOUCH_RATE]) for a in state.cars}
        self._demoed = {a: c.is_demoed for a, c in state.cars.items()}
        return self.values

    def snapshot(self):
        """Full tracker state, enough to resume exactly."""
        return {"values": {a: v.copy() for a, v in self.values.items()},
                "touch": {a: t.copy() for a, t in self._touch.items()}}

    def resume(self, state, snapshot):
        """Continue from a snapshot (e.g. a recorded mid-episode state) instead of the warm-up."""
        self.values = {a: np.array(v, dtype=np.float64) for a, v in snapshot["values"].items()}
        self._touch = {a: np.array(t, dtype=np.float64) for a, t in snapshot["touch"].items()}
        self._demoed = {a: c.is_demoed for a, c in state.cars.items()}
        return self.values

    def update(self, state, ttb, dt):
        newly = [a for a, c in state.cars.items() if c.is_demoed and not self._demoed.get(a, False)]
        self._demoed = {a: c.is_demoed for a, c in state.cars.items()}
        alpha = 1.0 - np.exp(-dt / EMA_SECONDS)
        x_all, team_touch = indicators(state, ttb, newly, dt)
        for agent, x in x_all.items():
            v = self.values.setdefault(agent, WARMUP.copy())
            seen = ~np.isnan(x)
            seen[5] = False
            v[seen] += alpha[seen] * (x[seen] - v[seen])
            t = self._touch.setdefault(agent, np.array([WARMUP[5] * TEAM_TOUCH_RATE, TEAM_TOUCH_RATE]))
            t += alpha[5] * (np.array([x[5], team_touch[agent]]) - t)
            v[5] = t[0] / t[1]
        return self.values
