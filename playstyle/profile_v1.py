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
                count fully. Both averages carry a small constant, TOUCH_PRIOR_SECONDS of evenly split touches
                at TEAM_TOUCH_RATE, so the share drifts back to 0.5 when a team stops touching and one touch
                after a long gap reads ~0.79, not 1. Starts as if one window of TEAM_TOUCH_RATE play split evenly
WARMUP comes from scripted normal-vs-normal games (scripts/profile_warmup.py), except demo_rate: scripted bots
never demo on purpose, so it uses the GC1 2s replay mean (0.13 demos/min; p90 0.34, max 0.77 over 120
player-games). Pair-symmetric features (last_man, challenge, touch_share) average exactly 0.5 over any population.
Demos are rare, so demo_rate averages over a longer window; one demo adds 60 / (90 * 0.75) ~ 0.9 and decays.
The observation reads demo_rate through 2 * tanh(x / 2) (observed()): bursts stay below 2, normal values barely move.
"""

import math

import numpy as np

VERSION = "profile-v1"
FEATURES = ("behind_ball", "last_man", "speed", "challenge", "demo_rate", "touch_share")
EMA_SECONDS = np.array([25.0, 25.0, 25.0, 25.0, 90.0, 25.0])
MAX_SPEED = 2300.0
DEMO_SCALE = 0.75  # demos per minute that reads as 1.0: a 2x-baseline Demo Specialist reads ~0.5-1
WARMUP = np.array([0.725, 0.500, 0.425, 0.500, 0.173, 0.500])  # scripted 10 sim-minutes, seed 0; demo from replays


DEMO_SOFT_CAP = 2.0
TEAM_TOUCH_RATE = 0.0123  # share of steps with a team touch in scripted play (0.18/s); scripts/profile_warmup.py
TOUCH_PRIOR_SECONDS = 4.0
TOUCH_PRIOR = TEAM_TOUCH_RATE * TOUCH_PRIOR_SECONDS / EMA_SECONDS[5]  # added to the team average, half to the car's


def observed(values):
    """Profile as the observation reads it: the running averages, with demo_rate soft-capped. Any leading shape."""
    out = np.array(values, dtype=np.float64)
    i = FEATURES.index("demo_rate")
    out[..., i] = DEMO_SOFT_CAP * np.tanh(out[..., i] / DEMO_SOFT_CAP)
    return out


def _layout(cars):
    """Per car, in the given order: team, sign (own goal at y = -5120 * sign), teammate index or None."""
    team = [c.team_num for c in cars]
    mate = [next((j for j in range(len(cars)) if j != i and team[j] == team[i]), None) for i in range(len(cars))]
    return team, [1.0 if t == 0 else -1.0 for t in team], mate


def _indicators(ids, cars, ball_y, ttb, demoed, newly, layout, dt):
    """One row per car, in the tracker's column order: the indicators for the first five features (NaN = no
    information this step), an unused slot, the car's share of this step's team touch (0 when the team didn't
    touch), and whether its team touched.
    Plain floats: with four cars this is several times faster than array operations."""
    team, sign, mate = layout
    ball_y = float(ball_y)
    y = [float(c.physics.position[1]) * s for c, s in zip(cars, sign)]
    touched = [c.ball_touches > 0 for c in cars]
    victim = {a: i for i, a in enumerate(ids) if newly[i]} if any(newly) else None
    rows = []
    for i, car in enumerate(cars):
        behind = last = speed = challenge = _NAN
        m, mate_touched, demos = mate[i], False, 0.0
        if not demoed[i]:
            behind = 1.0 if y[i] < ball_y * sign[i] else 0.0
            vx, vy, vz = car.physics.linear_velocity.tolist()
            speed = min(math.sqrt(vx * vx + vy * vy + vz * vz) / MAX_SPEED, 1.0)
        if m is not None:
            mate_touched = touched[m]
            if demoed[i] != demoed[m]:
                last = challenge = 1.0 if demoed[m] else 0.0
            elif not demoed[i]:
                last = 1.0 if y[i] < y[m] else 0.0
                t, tm = ttb[i], ttb[m]
                if t != tm and (t < _INF or tm < _INF):
                    challenge = 1.0 if t < tm else 0.0
        if victim is not None:  # a car was demoed this step: credit whoever last hit it, if an opponent
            j = victim.get(car.bump_victim_id)
            if j is not None and team[j] != team[i]:
                demos = 1.0
        rows.append((behind, last, speed, challenge, demos * 60.0 / dt / DEMO_SCALE if dt > 0 else 0.0,
                     0.0, (0.5 if mate_touched else 1.0) if touched[i] else 0.0,
                     1.0 if touched[i] or mate_touched else 0.0))
    return rows


def indicators(state, ttb, newly_demoed, dt):
    """(agent -> indicator per feature, agent -> whether its team touched); see _indicators."""
    ids = list(state.cars)
    cars = [state.cars[a] for a in ids]
    rows = np.array(_indicators(ids, cars, state.ball.position[1], [float(ttb[a]) for a in ids],
                                [bool(c.is_demoed) for c in cars], [a in newly_demoed for a in ids], _layout(cars), dt))
    return dict(zip(ids, rows[:, [0, 1, 2, 3, 4, 6]])), dict(zip(ids, rows[:, 7]))


class ProfileTracker:
    """Running averages for every car in one array; self.values maps agent -> its row (a live view).

    Columns of the array: the six features, then the two averages touch_share is the ratio of
    (own share of team touches, team touched). touch_share itself is never averaged, only derived."""

    def _start(self, ids, values, touch, demoed):
        self._ids, self._layout, self._alpha = list(ids), None, {}
        self._state = np.concatenate([values, touch], axis=1)
        self._demoed = list(demoed)
        self.values = dict(zip(self._ids, self._state[:, :6]))
        return self.values

    def reset(self, state):
        n = len(state.cars)
        return self._start(state.cars, np.tile(WARMUP, (n, 1)),
                           np.tile([WARMUP[5] * TEAM_TOUCH_RATE, TEAM_TOUCH_RATE], (n, 1)),
                           [bool(c.is_demoed) for c in state.cars.values()])

    def snapshot(self):
        """Full tracker state, enough to resume exactly."""
        return {"values": {a: s[:6].copy() for a, s in zip(self._ids, self._state)},
                "touch": {a: s[6:].copy() for a, s in zip(self._ids, self._state)},
                "demoed": dict(zip(self._ids, self._demoed))}

    def resume(self, snapshot):
        """Continue from a snapshot (e.g. a recorded mid-episode state) instead of the warm-up."""
        ids = list(snapshot["values"])
        return self._start(ids, np.array([snapshot["values"][a] for a in ids], dtype=np.float64),
                           np.array([snapshot["touch"][a] for a in ids], dtype=np.float64),
                           [bool(snapshot["demoed"][a]) for a in ids])

    def update(self, state, ttb, dt):
        ids = self._ids
        cars = [state.cars[a] for a in ids]
        if self._layout is None:
            self._layout = _layout(cars)
        alpha = self._alpha.get(dt)
        if alpha is None:
            a = 1.0 - np.exp(-dt / EMA_SECONDS)
            alpha = self._alpha[dt] = np.array([*a[:5], 0.0, a[5], a[5]])
        demoed = [bool(c.is_demoed) for c in cars]
        newly = [now and not before for now, before in zip(demoed, self._demoed)]
        self._demoed = demoed
        x = np.array(_indicators(ids, cars, state.ball.position[1], [float(ttb[a]) for a in ids], demoed, newly,
                                 self._layout, dt))
        s = self._state
        step = x - s
        step *= alpha
        step[np.isnan(step)] = 0.0  # no information: that average stays where it is
        s += step
        s[:, 5] = (s[:, 6] + 0.5 * TOUCH_PRIOR) / (s[:, 7] + TOUCH_PRIOR)
        return self.values


_NAN, _INF = float("nan"), float("inf")
