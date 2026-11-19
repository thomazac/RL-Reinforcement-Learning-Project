"""Phase 1c reward, reshaped with ideas from Necto/Nexto (Rolv-Arild/Necto, training/reward.py).

What changes against Phase1bReward (playstyle/rewards.py), and why:

* Potential-based shaping for position. Necto pays for the change in how good a situation is, not for being in it,
  so standing in a good spot earns nothing per step and the shaping telescopes over an episode. Four terms work
  this way here:
    - ball_potential (zero-sum, replaces ball_to_goal): ball closeness to the opponent goal minus to the own goal.
    - win_prob (zero-sum, replaces win): win probability from score difference, clock and overtime. A goal in a tied
      game late is worth more than one at 4-0, and holding a lead pays a little as the clock runs.
    - first_potential (replaces speed_to_ball): the first man's closeness to the ball plus his ball-to-goal alignment.
    - rotation_potential (replaces the per-step rotation reward, which paid every step a second man sat in his spot):
      the second man's closeness to his rotation spot.
  Each role potential is a car's potential while it holds that role and 0 otherwise, and the reward is its change
  from the last step, role changes included. So every term telescopes: over an episode it sums to its last value
  minus its first, whatever the car does with roles in between, and no approach-hand-off-retreat cycle can farm it.
  A role switch pays the difference between the two potentials at that moment. The step that resets the field after
  a goal and a car's respawn after a demo restart the potentials without paying the jump.
* air_touch uses Necto's height shaping: a cube-root activation of the touch height (the mean of car and ball
  height), squared, so low aerials already pay; touches near a wall pay half. Still paid once per distinct touch,
  as in Phase 1b, so pushing or dribbling cannot farm it.
* demo: Necto rewards demolitions. Only the attacker's side is a term here; the opponent subtraction already
  charges the victim's team.
* boost_spend: Necto's charge for spending boost low to the ground. Logged at weight 0: measured, it cancelled most
  of the boost pickup reward and taxes ground boosting that is often right.

Kept from Phase 1b: distinct-touch hit strength, saves, clears, shots (logged), sqrt boost gain, the coordination
penalties, the practice-only partial credit and its logging, goal / concede, team spirit and the opponent
subtraction on one-sided terms only. Not borrowed: Necto pays the conceding team by distance from the ball, which
pulls defenders toward the ball (more crowding), and pays every contact step (dribble farming); flip resets wait for
the Air Dribbler.

Every potential pays gamma * phi(now) - phi(before), with the gamma the run trains at, not Necto's plain difference.
Under discounting the plain difference quietly adds (1 - gamma) * phi every step: a reward for sitting in a good spot,
which is what potentials are meant to avoid. The exact form cancels it, so a potential's weight changes how fast the
bots learn, not which play is best (Ng, Harada and Russell 1999). Standing still therefore pays slightly less than 0,
-(1 - gamma) * phi, which offsets the discount. When a match ends, the shaping potentials (ball, first, rotation) count
as 0, so they telescope to nothing; win_prob keeps its final value, which is the result. win_prob's potential is the
win probability minus 0.5, so an even game sits at 0 and the two teams' potentials are opposite: over a match the
winner gets about +10 and the loser -10 at weight 20, like Phase 1b's win term, discounted to the end of the match.
"""

import math

import numpy as np
from rlgym.api import RewardFunction

from .rewards import (BALL_MAX_SPEED, CAR_MAX_SPEED, CEILING, CLEAR_HORIZON, GOAL_H, GOAL_Y, NEAR_BALL, ONE_SIDED,
                      SAVE_HORIZON, SHOT_HORIZON, TOUCH_COOLDOWN_TICKS, ZERO_SUM, crossing_point, enters_net, in_third,
                      leaves_third, own_goal_sign, rotation_spot)
from .roles import FIRST, SECOND

FIELD_X = 4096.0
CORNER_CUT = 1152.0  # the 45-degree corner walls cut this much off each axis
GOAL_POINT_Y = GOAL_Y + 440.0  # midway between the goal line and the back of the net
GOAL_POINT_Z = 321.39
DRIBBLE_HEIGHT = 150.0  # touches around this height are about a car-top dribble
MAX_DRIVE_SPEED = 1410.0  # top speed without boost: the distance scale for closeness to the ball or a spot
GOAL_RATE = 7.8 / 2 / 300  # goals per team per second in these bots' games (phase1b, ~7.8 per 5 min); re-measure per stage
_K = 25  # goal counts summed in the win-probability model; 0.0105 * 300 s gives a mean of ~3

PHASE_NX_TERMS = {
    "touch": ONE_SIDED,  # hit strength: ball speed change / BALL_MAX_SPEED, once per distinct touch
    "air_touch": ONE_SIDED,  # on a paid touch with the car airborne: Necto height shaping, halved near walls
    "save": ONE_SIDED,
    "clear": ONE_SIDED,
    "shot": ONE_SIDED,  # logged only
    "boost": ONE_SIDED,  # gain in sqrt(boost / 100)
    "boost_spend": ONE_SIDED,  # drop in sqrt(boost / 100) below goal height, scaled by how low (negative); logged
    "demo": ONE_SIDED,  # 1 when a car this one last bumped is demolished this step
    "first_potential": ONE_SIDED,  # change in the first man's (closeness to ball + alignment) / 2
    "rotation_potential": ONE_SIDED,  # change in the second man's closeness to his rotation spot
    "double_commit": ONE_SIDED,
    "nobody_challenging": ONE_SIDED,
    "nobody_home": ONE_SIDED,
    "path_approach": ONE_SIDED,  # practice episodes only, as in Phase 1b
    "shot_touch": ONE_SIDED,  # practice episodes only, as in Phase 1b
    "practice_shot": ONE_SIDED,  # logged only
    "practice_defended": ONE_SIDED,  # logged only
    "practice_level": ONE_SIDED,  # logged only
    "both_near": ONE_SIDED,  # logged only
    "goal": ZERO_SUM,
    "concede": ZERO_SUM,
    "ball_potential": ZERO_SUM,  # change in closeness of the ball to the opponent goal minus to the own goal
    "win_prob": ZERO_SUM,  # change in the team's win probability minus 0.5
}
# Starting weights. Kept terms keep their Phase 1b weights. Sizes measured on the phase1c 300M policy (per car per 5
# minutes, |weighted| summed; a goal or concede is ~82 at weight 10):
#   ball_potential: 70 at 2.5 (Necto's effective weight), ~20x ball_to_goal's target of 3 -> 0.1 (~2.8).
#   win_prob: potential p - 0.5, so a match from an even start ends at +-0.5 -> 20 makes it +-10, like the old win.
#   boost_spend: -1.7 at 0.08 cancelled most of the boost pickup reward (+2.2) and taxes ground boosting that is often
#     right (challenges, supersonic play) -> 0, logged.
#   first_potential and rotation_potential (0.5): net ~0 and ~+1.1 (old terms 0.84 and 1.36), swing 19 and 13. Watch
#     their totals in a run.
#   demo 2 (+0.4 per 5 min); air_touch 1.0 (2.1, the old one 2.7).
PHASE_NX = {"touch": 0.3, "air_touch": 1.0, "save": 3.0, "clear": 1.0, "shot": 0.0, "boost": 0.15,
            "boost_spend": 0.0, "demo": 2.0, "first_potential": 0.5, "rotation_potential": 0.5,
            "double_commit": -0.04, "nobody_challenging": -0.2, "nobody_home": -0.0017,
            "path_approach": 0.015, "shot_touch": 1.0,
            "practice_shot": 0.0, "practice_defended": 0.0, "practice_level": 0.0, "both_near": 0.0,
            "goal": 10.0, "concede": -8.0, "ball_potential": 0.1, "win_prob": 20.0}
assert list(PHASE_NX) == list(PHASE_NX_TERMS)
# Annealed in Phase 1c with the core layer. The potentials for the ball and the win, saves, demos, rotation and the
# coordination layer are not: potentials telescope, and the rest is the outcome or the team layer.
CORE_TERMS_NX = ("touch", "air_touch", "boost", "boost_spend", "clear", "first_potential", "path_approach",
                 "shot_touch")


# ---------------------------------------------------------------------------------------------------------------
# Pure functions (no game objects), so they can be tested on their own.

def win_probability(score_diff, time_remaining, overtime=False, over=False, rate=GOAL_RATE):
    """Probability that the team with this score difference (own - opponent) wins.

    Each team scores as a Poisson process at `rate` goals per second over the time left; a tie at 0:00 goes to
    overtime, which is a coin flip. In overtime the next goal wins, so a tied overtime is 0.5. With the clock at
    0:00 and a lead, the match ends when the ball lands: the leader has won."""
    d = int(score_diff)
    if over or (time_remaining <= 0 and d != 0):
        return 1.0 if d > 0 else 0.0 if d < 0 else 0.5
    if overtime or time_remaining <= 0:
        return 0.5
    mu = rate * float(time_remaining)
    k = np.arange(_K + 1)
    pmf = np.exp(-mu + k * math.log(mu) - np.array([math.lgamma(i + 1) for i in k])) if mu > 0 else (k == 0) * 1.0
    diff = np.convolve(pmf, pmf[::-1])  # P(own goals - opponent goals = j) at index j + _K
    j = np.arange(-_K, _K + 1)
    return float(diff[j > -d].sum() + 0.5 * diff[j == -d].sum())


def goal_point(team, own):
    """Point inside a goal: the own team's goal if own, else the opponent's."""
    sign = own_goal_sign(team) * (1.0 if own else -1.0)
    return np.array([0.0, sign * GOAL_POINT_Y, GOAL_POINT_Z])


def ball_potential(ball_pos, team):
    """Ball closeness to the opponent goal minus to the own goal, each exp(-distance / CAR_MAX_SPEED): -1..1."""
    b = np.asarray(ball_pos, dtype=np.float64)
    return float(np.exp(-np.linalg.norm(goal_point(team, False) - b) / CAR_MAX_SPEED)
                 - np.exp(-np.linalg.norm(goal_point(team, True) - b) / CAR_MAX_SPEED))


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-6 and nb > 1e-6 else 0.0


def first_potential(car_pos, ball_pos, team):
    """Necto's player quality: closeness to the ball, exp(-distance / MAX_DRIVE_SPEED), plus alignment, how much the
    car-to-ball direction points at the opponent goal rather than the own one. Averaged: -0.5..1."""
    c, b = np.asarray(car_pos, dtype=np.float64), np.asarray(ball_pos, dtype=np.float64)
    to_ball = b - c
    closeness = float(np.exp(-np.linalg.norm(to_ball) / MAX_DRIVE_SPEED))
    alignment = 0.5 * (_cos(to_ball, goal_point(team, False) - c) - _cos(to_ball, goal_point(team, True) - c))
    return 0.5 * (closeness + alignment)


def rotation_potential(car_pos, ball_pos, team):
    """Closeness to the rotation spot (rewards.rotation_spot), exp(-distance / MAX_DRIVE_SPEED): 0..1."""
    spot = rotation_spot(np.asarray(ball_pos, dtype=np.float64), team)
    return float(np.exp(-np.linalg.norm(spot - np.asarray(car_pos, dtype=np.float64)[:2]) / MAX_DRIVE_SPEED))


def wall_distance(x, y):
    """Distance from (x, y) to the nearest side wall, back wall or 45-degree corner wall."""
    ax, ay = abs(float(x)), abs(float(y))
    corner = (FIELD_X + GOAL_Y - CORNER_CUT - ax - ay) / math.sqrt(2)  # to the line x + y = X + Y - cut
    return max(min(FIELD_X - ax, GOAL_Y - ay, corner), 0.0)


def _height_activation(z):
    return np.cbrt((float(z) - DRIBBLE_HEIGHT) / CEILING)


_H0, _H1 = _height_activation(0.0), _height_activation(CEILING)


def air_touch_value(car_pos, ball_pos):
    """Necto's touch height shaping: ((h(z) - h(0)) / (h(ceiling) - h(0)))^2 with h a cube root centred on dribble
    height and z the mean of car and ball height, times (1 + wall factor) / 2 so wall touches pay about half. 0..1."""
    z = 0.5 * (float(car_pos[2]) + float(ball_pos[2]))
    height = ((_height_activation(z) - _H0) / (_H1 - _H0)) ** 2
    wall = 1.0 - math.exp(-wall_distance(car_pos[0], car_pos[1]) / CAR_MAX_SPEED)
    return float(min(height, 1.0) * (1.0 + wall) / 2.0)


# ---------------------------------------------------------------------------------------------------------------

class NextoStyleReward(RewardFunction):
    """Phase 1c: Phase1bReward with Necto-style potentials, height shaping, demos and boost spending
    (module docstring). Same interface and output as Phase1bReward: [final, raw components...] per agent, with
    final_i = (1 - tau) S_i + tau mean(S_team) - opponent_scale mean(S_opp) + Z_team(i)."""

    def __init__(self, weights=None, team_spirit=0.0, opponent_scale=1.0, gamma=0.99, goal_rate=GOAL_RATE):
        self.weights = dict(PHASE_NX if weights is None else weights)
        assert list(self.weights) == list(PHASE_NX_TERMS), "components are fixed; set a weight to 0 to switch one off"
        self.w = np.array(list(self.weights.values()))
        self.one_sided = np.array([PHASE_NX_TERMS[n] == ONE_SIDED for n in self.weights])
        self.team_spirit, self.opponent_scale, self.goal_rate = team_spirit, opponent_scale, goal_rate
        self.gamma = gamma  # the run's discount: potentials pay gamma * phi(now) - phi(before)

    @property
    def names(self):
        return list(self.weights)

    def _win(self, match, team):
        """Win potential: the team's win probability minus 0.5 (0 in an even game, opposite for the two teams)."""
        p_blue = win_probability(match.get("score_diff", 0), match.get("time_remaining", 0.0),
                                 match.get("overtime", False), match.get("over", False), self.goal_rate)
        return (p_blue if team == 0 else 1.0 - p_blue) - 0.5

    @staticmethod
    def _role_phi(car, role, ball_pos):
        """(first_potential, rotation_potential) of a car: its potential for the role it holds, 0 for the other."""
        if car.is_demoed or role not in (FIRST, SECOND):
            return 0.0, 0.0
        pos = np.asarray(car.physics.position, dtype=np.float64)
        ball = np.asarray(ball_pos, dtype=np.float64)
        if role == FIRST:
            return first_potential(pos, ball, car.team_num), 0.0
        return 0.0, rotation_potential(pos, ball, car.team_num)

    def reset(self, agents, initial_state, shared_info):
        self.prev_ball_vel = np.array(initial_state.ball.linear_velocity, dtype=np.float64)
        self.prev_ball_pos = np.array(initial_state.ball.position, dtype=np.float64)
        self.prev_slices = np.array(shared_info["ball_slices"], dtype=np.float64)
        roles = shared_info.get("roles", {})
        self.prev_role_phi = {a: self._role_phi(initial_state.cars[a], roles.get(a), initial_state.ball.position)
                              for a in agents}
        self.prev_boost = {a: float(initial_state.cars[a].boost_amount) for a in agents}
        self.prev_demoed = {a: bool(initial_state.cars[a].is_demoed) for a in agents}
        match = shared_info.get("match", {})
        self.prev_win = {t: self._win(match, t) for t in (0, 1)}
        self.in_contact = {a: False for a in agents}
        self.last_paid = {a: -10**9 for a in agents}
        self.field_reset_next = False
        self.practice = shared_info.get("start") == "defend_shot"
        self.practice_first_step = self.practice

    def _raw(self, agents, state, shared_info, terminal=False):
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
        win = {t: self._win(match, t) for t in (0, 1)}
        demoed = {a: bool(state.cars[a].is_demoed) for a in agents}
        newly_demoed = {a for a in agents if demoed[a] and not self.prev_demoed[a]}
        role_phi = {}
        near = {a: not demoed[a]
                and float(np.linalg.norm(state.cars[a].physics.position - ball_pos)) < NEAR_BALL for a in agents}

        for a in agents:
            car, r = state.cars[a], raw[a]
            team = car.team_num
            pos = np.array(car.physics.position, dtype=np.float64)
            touching = car.ball_touches > 0
            moved = not (field_reset or demoed[a] or self.prev_demoed[a])  # no teleport between the two states

            # Potentials: gamma * phi(now) - phi(before); role changes included. Shaping potentials end at 0.
            g = self.gamma
            phi = self._role_phi(car, roles.get(a), ball_pos)
            end = 0.0 if terminal else 1.0
            if moved:
                r[idx["first_potential"]] = g * end * phi[0] - self.prev_role_phi[a][0]
                r[idx["rotation_potential"]] = g * end * phi[1] - self.prev_role_phi[a][1]
            role_phi[a] = phi
            if not field_reset:
                r[idx["ball_potential"]] = (g * end * ball_potential(ball_pos, team)
                                            - ball_potential(self.prev_ball_pos, team))
            r[idx["win_prob"]] = g * win[team] - self.prev_win[team]

            mates = [b for b in agents if state.cars[b].team_num == team]
            r[idx["both_near"]] = float(all(near[b] for b in mates))
            if self.practice and enters_net(self.prev_slices, team, SHOT_HORIZON):
                if self.practice_first_step:
                    r[idx["practice_shot"]] = 1.0
                r[idx["practice_level"]] = 1.0 + float(shared_info.get("practice_level", 0.0))
                point = crossing_point(slices, team)
                if point is not None and not demoed[a]:
                    to_point = point - pos[:2]
                    r[idx["path_approach"]] = max(0.0, float(car.physics.linear_velocity[:2] @ to_point)
                                                  / max(float(np.linalg.norm(to_point)), 1e-6)) / CAR_MAX_SPEED
                if touching and not self.in_contact[a] and not field_reset:
                    r[idx["shot_touch"]] = 1.0
                    if not enters_net(slices, team, SHOT_HORIZON):
                        r[idx["practice_defended"]] = 1.0
            if touching and not field_reset:
                if not self.in_contact[a] and state.tick_count - self.last_paid[a] >= TOUCH_COOLDOWN_TICKS:
                    self.last_paid[a] = state.tick_count
                    r[idx["touch"]] = hit
                    if not car.on_ground:
                        r[idx["air_touch"]] = air_touch_value(pos, ball_pos)
                    if in_third(self.prev_ball_pos, team) and not leaves_third(self.prev_slices, team, CLEAR_HORIZON) \
                            and leaves_third(slices, team, CLEAR_HORIZON):
                        r[idx["clear"]] = 1.0
                if enters_net(self.prev_slices, team, SAVE_HORIZON) and not enters_net(slices, team, SHOT_HORIZON):
                    r[idx["save"]] = 1.0
                if not enters_net(self.prev_slices, 1 - team, SHOT_HORIZON) and enters_net(slices, 1 - team, SHOT_HORIZON):
                    r[idx["shot"]] = 1.0
            self.in_contact[a] = touching

            boost = float(car.boost_amount)
            if moved:
                change = math.sqrt(max(boost, 0.0) / 100) - math.sqrt(max(self.prev_boost[a], 0.0) / 100)
                if change > 0:
                    r[idx["boost"]] = change
                elif change < 0 and pos[2] < GOAL_H:
                    r[idx["boost_spend"]] = change * (1.0 - pos[2] / GOAL_H)
            victim = getattr(car, "bump_victim_id", None)
            if victim in newly_demoed and state.cars[victim].team_num != team:
                r[idx["demo"]] = 1.0

            ev = coordination.get(team)
            if ev is not None:
                r[idx["double_commit"]] = float(ev["double_commit"] == a) + (ev["revoked"] or {}).get(a, 0)
                r[idx["nobody_challenging"]] = float(ev["nobody_challenging"] == a)
                r[idx["nobody_home"]] = float(ev["nobody_home"])

            if state.goal_scored:
                r[idx["goal"]] = float(state.scoring_team == team)
                r[idx["concede"]] = float(state.scoring_team != team)

        for a in agents:
            self.prev_role_phi[a] = role_phi[a]
            self.prev_boost[a], self.prev_demoed[a] = float(state.cars[a].boost_amount), demoed[a]
        self.prev_win = win
        self.prev_ball_vel, self.prev_ball_pos, self.prev_slices = ball_vel, ball_pos, slices
        self.field_reset_next = bool(state.goal_scored)  # the engine resets to kickoff on the next step
        self.practice_first_step = False
        return raw

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        raw = self._raw(agents, state, shared_info, terminal=any(is_terminated.values()) if is_terminated else False)
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
