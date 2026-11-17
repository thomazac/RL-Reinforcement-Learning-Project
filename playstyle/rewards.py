"""Rewards.

Each agent's reward is a vector: the blended reward PPO trains on, then the raw value of every component. The
training side splits them (playstyle.training.ClippedGAETrajectoryProcessor) so per-component means can be
logged every iteration, which the design doc calls the main debugging tool.

Team spirit (design doc): R_final_i = (1 - tau) R_i + tau mean(R_team) - opponent_scale * mean(R_opp).
"""

import numpy as np
from rlgym.api import RewardFunction

from .ball_prediction import SLICE_TIMES, TICK_RATE
from .roles import FIRST

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


# ---------------------------------------------------------------------------------------------------------------
# Phase 1b

BALL_MAX_SPEED = 6000.0
GOAL_Y, GOAL_HALF_W, GOAL_H = 5120.0, 893.0, 642.0
CEILING = 2044.0
DEFENSIVE_THIRD_Y = GOAL_Y / 3  # a team's defensive third: the ball at least this far into its own half
TOUCH_COOLDOWN_TICKS = round(0.25 * TICK_RATE)  # a car's touches closer together than this pay once
SAVE_HORIZON = 1.0  # s: a save stops a ball predicted into the own net this soon
CLEAR_HORIZON = 2.0  # s: a clear sends the ball out of the defensive third this soon
SHOT_HORIZON = float(SLICE_TIMES[-1])  # s: a shot puts the ball on course for the opponent net this soon

# Raw component -> category. One-sided terms are a car's own doing: they go through the team-spirit blend and the
# opponent subtraction. Zero-sum terms are the same for both teammates and the exact negative (or mirror) of the
# opponents' (goal / concede, ball velocity toward the opponent net minus toward the own, win / loss), so they are
# added once per team and left out of the subtraction, which would otherwise double them.
ONE_SIDED, ZERO_SUM = "one-sided", "zero-sum"
PHASE_1B_TERMS = {
    "touch": ONE_SIDED,  # hit strength: ball speed change / BALL_MAX_SPEED, once per distinct touch
    "air_touch": ONE_SIDED,  # on a paid touch with the car airborne: ball height / CEILING
    "save": ONE_SIDED,  # 1: touched a ball predicted into the own net within SAVE_HORIZON; now it is not
    "clear": ONE_SIDED,  # 1: paid touch in the defensive third that sends the ball out within CLEAR_HORIZON
    "shot": ONE_SIDED,  # 1: touch that puts the ball on course for the opponent net (logged; weight 0 to start)
    "boost": ONE_SIDED,  # gain in sqrt(boost / 100): a pickup pays more the emptier the tank
    "speed_to_ball": ONE_SIDED,  # as in Phase 1a, but only for the current first man (it would pay a second man to double-commit)
    "double_commit": ONE_SIDED,  # 1 per step charged to this car (playstyle/coordination.py), plus revoked steps
    "nobody_challenging": ONE_SIDED,  # 1 per step charged to this car (the first man)
    "nobody_home": ONE_SIDED,  # 1 per step for both teammates while their net is empty
    "goal": ZERO_SUM,  # 1 on the step the team scores
    "concede": ZERO_SUM,  # 1 on the step the team concedes
    "ball_to_goal": ZERO_SUM,  # (ball velocity toward the opponent net - toward the own net) / (2 BALL_MAX_SPEED)
    "win": ZERO_SUM,  # +1 / -1 on the step the match ends
}
# Sized with scripts/size_rewards.py on the Phase 1a 500M policy (per agent per 5 minutes of play, a goal = 10):
# shaping terms at most ~1/3 goal each, coordination ~1/10 goal each. touch is sized for the human rate and
# strength (47 touches at 1,309 uu/s -> 3.1), above today's farmed 84 weak touches (2.3). save and air_touch
# barely occur yet (0.04 and 4.6 per 5 minutes), so they are set per event (1.5 per save, 0.5 for a mid-height air
# touch) rather than from frequency, which would give 48 per save. speed_to_ball was measured with every car earning
# it; only the first man does now, so its weight is doubled to keep its total near 1.0.
# After 500M steps saves were still 0.1-0.2 per player per 5 minutes against 1.28 for GC1 humans (same definition),
# so save went from 1.5 to 3, still well under the concede penalty so letting a shot through never pays; most of the
# extra saves are meant to come from the defend_shot start states (playstyle/states.py), not the weight.
PHASE_1B = {"touch": 0.3, "air_touch": 1.0, "save": 3.0, "clear": 1.0, "shot": 0.0, "boost": 0.15,
            "speed_to_ball": 0.0014, "double_commit": -0.04, "nobody_challenging": -0.2, "nobody_home": -0.0017,
            "goal": 10.0, "concede": -8.0, "ball_to_goal": 0.0055, "win": 10.0}


def own_goal_sign(team) -> float:
    return -1.0 if team == 0 else 1.0


def enters_net(slices, defending_team, horizon) -> bool:
    """Predicted ball path (positions at SLICE_TIMES) crosses defending_team's goal line within horizon seconds."""
    s = slices[SLICE_TIMES <= horizon + 1e-6]
    sign = own_goal_sign(defending_team)
    return bool(np.any((s[:, 1] * sign > GOAL_Y) & (np.abs(s[:, 0]) < GOAL_HALF_W) & (s[:, 2] < GOAL_H)))


def leaves_third(slices, team, horizon) -> bool:
    """Predicted ball path leaves team's defensive third within horizon seconds."""
    s = slices[SLICE_TIMES <= horizon + 1e-6]
    return bool(np.any(s[:, 1] * own_goal_sign(team) < DEFENSIVE_THIRD_Y))


def in_third(position, team) -> bool:
    return float(position[1]) * own_goal_sign(team) >= DEFENSIVE_THIRD_Y


class Phase1bReward(RewardFunction):
    """Phase 1b: outcome terms plus core and coordination shaping (PHASE_1B_TERMS).

    final_i = (1 - tau) S_i + tau mean(S_team) - opponent_scale mean(S_opp) + Z_team(i)
    with S the weighted one-sided terms and Z the weighted zero-sum terms. Returns [final, raw components...] like
    ShapedReward. Needs the coordination events in shared_info (TTBRoleProvider with coordination on).

    Touch is paid once per distinct touch: on a car's first contact step after a step without contact, and only if
    its last paid touch is at least TOUCH_COOLDOWN_TICKS old. Pushing or dribbling the ball is one long contact and
    pays once, by the ball speed change on that first step. Steps that reset the field (the kickoff after a goal) and
    a car's first step after a demo pay no touch or boost, so respawn and kickoff boost refills are not rewarded."""

    def __init__(self, weights=None, team_spirit=0.0, opponent_scale=1.0):
        self.weights = dict(PHASE_1B if weights is None else weights)
        assert list(self.weights) == list(PHASE_1B_TERMS), "components are fixed; set a weight to 0 to switch one off"
        self.w = np.array(list(self.weights.values()))
        self.one_sided = np.array([PHASE_1B_TERMS[n] == ONE_SIDED for n in self.weights])
        self.team_spirit, self.opponent_scale = team_spirit, opponent_scale

    @property
    def names(self):
        return list(self.weights)

    def reset(self, agents, initial_state, shared_info):
        self.prev_ball_vel = np.array(initial_state.ball.linear_velocity, dtype=np.float64)
        self.prev_ball_pos = np.array(initial_state.ball.position, dtype=np.float64)
        self.prev_slices = np.array(shared_info["ball_slices"], dtype=np.float64)
        self.prev_boost = {a: float(initial_state.cars[a].boost_amount) for a in agents}
        self.prev_demoed = {a: bool(initial_state.cars[a].is_demoed) for a in agents}
        self.in_contact = {a: False for a in agents}
        self.last_paid = {a: -10**9 for a in agents}
        self.field_reset_next = False

    def _raw(self, agents, state, shared_info):
        idx = {n: i for i, n in enumerate(self.weights)}
        raw = {a: np.zeros(len(self.weights)) for a in agents}
        ball_pos = np.array(state.ball.position, dtype=np.float64)
        ball_vel = np.array(state.ball.linear_velocity, dtype=np.float64)
        slices = np.array(shared_info["ball_slices"], dtype=np.float64)
        hit = min(float(np.linalg.norm(ball_vel - self.prev_ball_vel)), BALL_MAX_SPEED) / BALL_MAX_SPEED
        field_reset = self.field_reset_next
        match = shared_info.get("match", {})
        coordination = shared_info.get("coordination", {})
        roles = shared_info.get("roles", {})

        for a in agents:
            car, r = state.cars[a], raw[a]
            team = car.team_num
            touching = car.ball_touches > 0
            if not car.is_demoed and roles.get(a) == FIRST:
                r[idx["speed_to_ball"]] = components(car, state.ball)[1]
            if touching and not field_reset:
                new_contact = not self.in_contact[a]
                if new_contact and state.tick_count - self.last_paid[a] >= TOUCH_COOLDOWN_TICKS:
                    self.last_paid[a] = state.tick_count
                    r[idx["touch"]] = hit
                    if not car.on_ground:
                        r[idx["air_touch"]] = min(float(ball_pos[2]) / CEILING, 1.0)
                    if in_third(self.prev_ball_pos, team) and not leaves_third(self.prev_slices, team, CLEAR_HORIZON) \
                            and leaves_third(slices, team, CLEAR_HORIZON):
                        r[idx["clear"]] = 1.0
                if enters_net(self.prev_slices, team, SAVE_HORIZON) and not enters_net(slices, team, SHOT_HORIZON):
                    r[idx["save"]] = 1.0
                if not enters_net(self.prev_slices, 1 - team, SHOT_HORIZON) and enters_net(slices, 1 - team, SHOT_HORIZON):
                    r[idx["shot"]] = 1.0
            self.in_contact[a] = touching

            boost = float(car.boost_amount)
            if not (field_reset or self.prev_demoed[a] or car.is_demoed):
                r[idx["boost"]] = max(0.0, np.sqrt(boost / 100) - np.sqrt(self.prev_boost[a] / 100))
            self.prev_boost[a], self.prev_demoed[a] = boost, bool(car.is_demoed)

            ev = coordination.get(team)
            if ev is not None:
                r[idx["double_commit"]] = float(ev["double_commit"] == a) + (ev["revoked"] or {}).get(a, 0)
                r[idx["nobody_challenging"]] = float(ev["nobody_challenging"] == a)
                r[idx["nobody_home"]] = float(ev["nobody_home"])

            if state.goal_scored:
                r[idx["goal"]] = float(state.scoring_team == team)
                r[idx["concede"]] = float(state.scoring_team != team)
            sign = own_goal_sign(team)
            toward_opp = _unit(np.array([0.0, -sign * GOAL_Y, GOAL_H / 2]) - ball_pos)
            toward_own = _unit(np.array([0.0, sign * GOAL_Y, GOAL_H / 2]) - ball_pos)
            r[idx["ball_to_goal"]] = float(ball_vel @ (toward_opp - toward_own)) / (2 * BALL_MAX_SPEED)
            if match.get("over") and match.get("winner") is not None:
                r[idx["win"]] = 1.0 if match["winner"] == ("blue" if team == 0 else "orange") else -1.0

        self.prev_ball_vel, self.prev_ball_pos, self.prev_slices = ball_vel, ball_pos, slices
        self.field_reset_next = bool(state.goal_scored)  # the engine resets to kickoff on the next step
        return raw

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        raw = self._raw(agents, state, shared_info)
        own = {a: float(raw[a][self.one_sided] @ self.w[self.one_sided]) for a in agents}
        shared = {a: float(raw[a][~self.one_sided] @ self.w[~self.one_sided]) for a in agents}
        team_mean = {t: np.mean([own[a] for a in agents if state.cars[a].team_num == t] or [0.0]) for t in (0, 1)}
        out = {}
        for a in agents:
            team = state.cars[a].team_num
            final = ((1 - self.team_spirit) * own[a] + self.team_spirit * team_mean[team]
                     - self.opponent_scale * team_mean[1 - team] + shared[a])
            out[a] = np.concatenate([[final], raw[a]]).astype(np.float32)
        return out


def _unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-6)
