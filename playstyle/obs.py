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

POS_COEF, VEL_COEF, ANG_VEL_COEF = 1 / 2300, 1 / 2300, 1 / 5.5
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

    def reset(self, agents: List, initial_state, shared_info: Dict[str, Any]) -> None:
        pass

    def build_obs(self, agents: List, state, shared_info: Dict[str, Any]) -> Dict[Any, np.ndarray]:
        ids = sorted(state.cars)
        cars = [state.cars[a] for a in ids]
        team = np.array([c.team_num for c in cars])
        ttb = np.array([shared_info["obs_ttb"][a] for a in ids])
        reachable = np.isfinite(ttb)
        ttb_scaled = np.minimum(np.where(reachable, ttb, TTB_CAP), TTB_CAP) / ttb_v1.HORIZON

        # Blue-frame arrays; orange agents get them mirrored.
        pos = np.array([c.physics.position for c in cars], dtype=np.float64)
        vel = np.array([c.physics.linear_velocity for c in cars], dtype=np.float64)
        ang = np.array([c.physics.angular_velocity for c in cars], dtype=np.float64)
        fwd = np.array([c.physics.forward for c in cars], dtype=np.float64)
        up = np.array([c.physics.up for c in cars], dtype=np.float64)
        rot = np.array([c.physics.rotation_mtx for c in cars], dtype=np.float64)  # columns: forward, right, up
        flags = np.array([[c.boost_amount * BOOST_COEF, c.on_ground, c.has_flip, not c.has_jumped, c.is_demoed]
                          for c in cars], dtype=np.float64)
        ttb_cols = np.stack([ttb_scaled, reachable], axis=1)
        ball = state.ball
        b_pos, b_vel, b_ang = (np.asarray(v, dtype=np.float64) for v in
                               (ball.position, ball.linear_velocity, ball.angular_velocity))
        predicted = np.asarray(shared_info["obs_ball_slices"], dtype=np.float64)[PREDICT_SLICES]
        pads = np.asarray(state.boost_pad_timers, dtype=np.float64)
        match = shared_info.get("match", {})
        prev = shared_info.get("prev_action", {})
        kickoff = float(shared_info.get("kickoff", False))
        profile = shared_info["profile"]

        obs = {}
        for agent in agents:
            i = ids.index(agent)
            orange = team[i] == 1
            m = MIRROR if orange else np.ones(3)
            others = [j for j in range(len(ids)) if j != i and team[j] == team[i]] + \
                     [j for j in range(len(ids)) if team[j] != team[i]]

            def car_block(j):
                p, v = pos[j] * m, vel[j] * m
                return np.concatenate([
                    p * POS_COEF, v * VEL_COEF, ang[j] * m * ANG_VEL_COEF, fwd[j] * m, up[j] * m,
                    (p - pos[i] * m) * POS_COEF, (v - vel[i] * m) * VEL_COEF,
                    rot[j].T @ (b_pos - pos[j]) * POS_COEF,  # unaffected by mirroring: car and ball rotate alike
                    flags[j], ttb_cols[j]])

            bp, bv = b_pos * m, b_vel * m
            score = float(match.get("score_diff", 0)) * (-1 if orange else 1)
            parts = [car_block(i)] + [car_block(j) for j in others] + [
                bp * POS_COEF, bv * VEL_COEF, b_ang * m * ANG_VEL_COEF,
                (bp - pos[i] * m) * POS_COEF, (bv - vel[i] * m) * VEL_COEF,
                (predicted * m).ravel() * POS_COEF,
                (pads[::-1] if orange else pads) * PAD_TIMER_COEF,
                [np.clip(score, -SCORE_CAP, SCORE_CAP) / SCORE_CAP,
                 float(match.get("time_remaining", MATCH_SECONDS)) / MATCH_SECONDS,
                 float(match.get("overtime", False)), kickoff],
                prev.get(agent, np.zeros(8)),
                profile_v1.observed(profile[ids[others[0]]] if others and team[others[0]] == team[i]
                                    else profile_v1.WARMUP),
            ]
            obs[agent] = np.concatenate(parts).astype(np.float32)
        return obs
