"""Entity observation builder. Changing the layout means retraining; bump OBS_LAYOUT when it changes.

Per agent, mirrored for orange so one policy plays both sides (x and y flipped):
  cars       4x31  self, teammate, then opponents by agent id. Every block has the same fields with the same
                   meaning, so one encoder reads all four:
                   position, velocity, angular velocity, forward, up (15, field frame);
                   position and velocity relative to the observing agent (6, zero in the self block);
                   ball position in this car's own frame: forward, right, up (3);
                   boost; on ground; has flip; has jump; demoed (5); TTB (scaled) and TTB reachable (2)
  ball       24    position, velocity, angular velocity; position and velocity relative to self;
                   predicted position at +0.5, +1, +2 s
  global     52    34 boost pad timers; score difference; time remaining; overtime; kickoff; previous action (8);
                   teammate profile (6, see profile_v1; the network routes it to the teammate token)
TTB and the ball prediction come from the frozen estimator (shared_info["obs_ttb"], ["obs_ball_slices"]),
the profile from the frozen profile_v1 (shared_info["profile"]).
"""

from typing import Any, Dict, List

import numpy as np
from rlgym.api import ActionParser, ObsBuilder

from . import profile_v1, ttb_v1

OBS_LAYOUT = "entity-v1"
CAR_DIM, BALL_DIM, GLOBAL_DIM = 31, 24, 52
OBS_DIM = 4 * CAR_DIM + BALL_DIM + GLOBAL_DIM

# Scales keep every feature within about +-1.5 (checked by scripts/obs_ranges.py on the reference).
POS_COEF = 1 / 5120  # half the field length: absolute positions within +-1.2 even inside a goal
REL_POS_COEF = 1 / 8000  # relative positions and ball in car frame: corner to corner ~ +-1.4
VEL_COEF, REL_VEL_COEF = 1 / 2300, 1 / 4600  # car max speed; two cars at max speed head-on
BALL_VEL_COEF = 1 / 6000  # ball max speed; also the ball's velocity relative to a car
ANG_VEL_COEF, BALL_ANG_VEL_COEF = 1 / 5.5, 1 / 6.0  # car and ball angular velocity caps
PAD_TIMER_COEF, BOOST_COEF = 1 / 10, 1 / 100
TTB_CAP = ttb_v1.HORIZON + 0.5  # unreachable cars read as just past the horizon, with the reachable flag off
PREDICT_SLICES = [int(round(t / ttb_v1.SLICE_DT)) for t in (0.5, 1.0, 2.0)]
MATCH_SECONDS = 300.0
SCORE_CAP = 3.0
MIRROR = np.array([-1.0, -1.0, 1.0])


class RecordingActionParser(ActionParser):
    """Wraps an action parser and stores each agent's last parsed controls in shared_info["prev_action"]."""

    def __init__(self, parser):
        self.parser = parser

    def get_action_space(self, agent):
        return self.parser.get_action_space(agent)

    def reset(self, agents, initial_state, shared_info):
        self.parser.reset(agents, initial_state, shared_info)
        shared_info["prev_action"] = {}

    def parse_actions(self, actions, state, shared_info):
        parsed = self.parser.parse_actions(actions, state, shared_info)
        shared_info["prev_action"] = {a: np.asarray(p, dtype=np.float32).reshape(-1, 8)[-1] for a, p in parsed.items()}
        return parsed


class EntityObs(ObsBuilder):
    def get_obs_space(self, agent):
        return "real", OBS_DIM

    def __init__(self):
        self._key = None

    def reset(self, agents: List, initial_state, shared_info: Dict[str, Any]) -> None:
        pass

    def _set_layout(self, ids, team):
        """Everything that depends only on who is in the game: block order and mirror signs per observer."""
        n = len(ids)
        self._key, self._row = (ids, team), {a: i for i, a in enumerate(ids)}
        mates = [[j for j in range(n) if j != i and team[j] == team[i]] for i in range(n)]
        self._order = np.array([[i] + mates[i] + [j for j in range(n) if team[j] != team[i]] for i in range(n)])
        self._mate = [ids[m[0]] if m else None for m in mates]
        self._orange = np.array(team) == 1
        self._sign = np.where(self._orange, -1.0, 1.0)
        mirror = np.where(self._orange[:, None], MIRROR, 1.0)
        # Mirror signs for the absolute vectors of a car block, its relative vectors, and the ball block.
        self._mirror15, self._mirror6, self._mirror24 = (np.tile(mirror, k)[:, None, :] for k in (5, 2, 8))
        self._mirror24 = self._mirror24[:, 0]

    def build_obs(self, agents: List, state, shared_info: Dict[str, Any]) -> Dict[Any, np.ndarray]:
        """All observers at once: one [observer, feature] array, filled block by block."""
        ids = sorted(state.cars)
        cars = [state.cars[a] for a in ids]
        team = [c.team_num for c in cars]
        if self._key != (ids, team):
            self._set_layout(ids, team)
        n, order = len(ids), self._order
        out = np.empty((n, n * CAR_DIM + BALL_DIM + GLOBAL_DIM))  # OBS_DIM in 2v2
        blocks = out[:, :n * CAR_DIM].reshape(n, n, CAR_DIM)  # [observer, block, feature]

        # Blue-frame arrays, one row per car; orange observers get them mirrored (x and y flipped).
        physics = [c.physics for c in cars]
        pos = np.array([p.position for p in physics], dtype=np.float64)
        vel = np.array([p.linear_velocity for p in physics], dtype=np.float64)
        rot = np.array([p.rotation_mtx for p in physics], dtype=np.float64)  # columns: forward, right, up
        ball = state.ball
        b_pos, b_vel = np.asarray(ball.position, dtype=np.float64), np.asarray(ball.linear_velocity, dtype=np.float64)
        to_ball = b_pos - pos

        absolute = np.empty((n, 15))
        absolute[:, 0:3] = pos * POS_COEF
        absolute[:, 3:6] = vel * VEL_COEF
        absolute[:, 6:9] = np.array([p.angular_velocity for p in physics], dtype=np.float64) * ANG_VEL_COEF
        absolute[:, 9:12] = rot[:, :, 0]
        absolute[:, 12:15] = rot[:, :, 2]
        blocks[:, :, 0:15] = absolute[order] * self._mirror15
        blocks[:, :, 15:18] = (pos[order] - pos[:, None]) * REL_POS_COEF
        blocks[:, :, 18:21] = (vel[order] - vel[:, None]) * REL_VEL_COEF
        blocks[:, :, 15:21] *= self._mirror6

        # The same for every observer: ball in the car's own frame (car and ball rotate alike under mirroring),
        # the car's state flags and its TTB.
        own = np.empty((n, 10))
        own[:, 0:3] = np.matmul(to_ball[:, None, :], rot)[:, 0] * REL_POS_COEF
        ttb = shared_info["obs_ttb"]
        own[:, 3:] = [(c.boost_amount * BOOST_COEF, c.on_ground, c.has_flip, not c.has_jumped, c.is_demoed,
                       (t if t < TTB_CAP else TTB_CAP) / ttb_v1.HORIZON, t < _INF)
                      for c, t in zip(cars, (float(ttb[a]) for a in ids))]
        blocks[:, :, 21:31] = own[order]

        k = n * CAR_DIM
        ball_block = out[:, k:k + BALL_DIM]
        ball_block[:, 0:3] = b_pos * POS_COEF
        ball_block[:, 3:6] = b_vel * BALL_VEL_COEF
        ball_block[:, 6:9] = np.asarray(ball.angular_velocity, dtype=np.float64) * BALL_ANG_VEL_COEF
        ball_block[:, 9:12] = to_ball * REL_POS_COEF
        ball_block[:, 12:15] = (b_vel - vel) * BALL_VEL_COEF
        ball_block[:, 15:24] = np.asarray(shared_info["obs_ball_slices"], dtype=np.float64)[PREDICT_SLICES].ravel() \
            * POS_COEF
        ball_block *= self._mirror24

        k += BALL_DIM
        pads = np.asarray(state.boost_pad_timers, dtype=np.float64) * PAD_TIMER_COEF
        out[:, k:k + 34] = np.where(self._orange[:, None], pads[::-1], pads)
        match = shared_info.get("match", {})
        out[:, k + 34] = np.clip(float(match.get("score_diff", 0)) * self._sign, -SCORE_CAP, SCORE_CAP) / SCORE_CAP
        out[:, k + 35] = float(match.get("time_remaining", MATCH_SECONDS)) / MATCH_SECONDS
        out[:, k + 36] = float(match.get("overtime", False))
        out[:, k + 37] = float(shared_info.get("kickoff", False))
        prev = shared_info.get("prev_action", {})
        out[:, k + 38:k + 46] = [prev[a] for a in ids] if len(prev) == n else [prev.get(a, _NO_ACTION) for a in ids]
        profile = shared_info["profile"]
        out[:, k + 46:] = profile_v1.observed([profile_v1.WARMUP if m is None else profile[m] for m in self._mate])

        out = out.astype(np.float32)
        return {a: out[self._row[a]] for a in agents}


_INF = float("inf")
_NO_ACTION = np.zeros(8)
