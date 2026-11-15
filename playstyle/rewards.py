"""Rewards.

Each agent's reward is a vector: the blended reward PPO trains on, then the raw value of every component. The
training side splits them (playstyle.training.ClippedGAETrajectoryProcessor) so per-component means can be
logged every iteration, which the design doc calls the main debugging tool.

Team spirit (design doc): R_final_i = (1 - tau) R_i + tau mean(R_team) - opponent_scale * mean(R_opp).
"""

import numpy as np
from rlgym.api import RewardFunction

CAR_MAX_SPEED = 2300.0

# Phase 1a: core layer only, focused on touching the ball. Weights follow the early-stage example in Zealan's
# RLGym-PPO guide; the component means in the run log show whether one term swamps the others.
PHASE_1A = {"touch": 50.0, "speed_to_ball": 5.0, "face_ball": 1.0, "in_air": 0.15}


def components(car, ball):
    """Raw Phase 1a components for one car: touched the ball this step (0/1), speed toward the ball as a
    fraction of max car speed (0 when moving away), cosine between the car's nose and the ball (-1..1), and
    airborne (0/1). All zero while demoed."""
    if car.is_demoed:
        return np.zeros(4)
    to_ball = ball.position - car.physics.position
    direction = to_ball / max(np.linalg.norm(to_ball), 1e-6)
    return np.array([float(car.ball_touches > 0),
                     max(0.0, float(np.dot(car.physics.linear_velocity, direction))) / CAR_MAX_SPEED,
                     float(np.dot(car.physics.forward, direction)),
                     float(not car.on_ground)])


class ShapedReward(RewardFunction):
    def __init__(self, weights=None, team_spirit=0.0, opponent_scale=1.0):
        self.weights = dict(PHASE_1A if weights is None else weights)
        assert list(self.weights) == list(PHASE_1A), "components are fixed; set a weight to 0 to switch one off"
        self.w = np.array(list(self.weights.values()))
        self.team_spirit, self.opponent_scale = team_spirit, opponent_scale

    @property
    def names(self):
        return list(self.weights)

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        raw = {a: components(state.cars[a], state.ball) for a in agents}
        own = {a: float(raw[a] @ self.w) for a in agents}
        team_mean = {t: np.mean([own[a] for a in agents if state.cars[a].team_num == t]) for t in (0, 1)}
        out = {}
        for a in agents:
            team = state.cars[a].team_num
            final = ((1 - self.team_spirit) * own[a] + self.team_spirit * team_mean[team]
                     - self.opponent_scale * team_mean[1 - team])
            out[a] = np.concatenate([[final], raw[a]]).astype(np.float32)
        return out
