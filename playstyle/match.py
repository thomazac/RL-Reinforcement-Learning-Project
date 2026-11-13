"""Matches that run across goals: score, clock and overtime in shared_info["match"].

Episodes don't end on a goal. The engine wrapper resets to kickoff on the step after a goal and keeps
playing; the episode ends when the match does (or on the truncation conditions). Each episode starts
from a sampled score difference and clock so the policy sees leads, deficits and late games from the start.

Rules, as in Rocket League: the clock runs only once the kickoff ball has moved. At 0:00 the match ends
when the ball next touches the ground, or goes to overtime if tied; in overtime the next goal wins.
"""

import copy

import numpy as np
from rlgym.api import DoneCondition, StateMutator, TransitionEngine
from rlgym.rocket_league.state_mutators import KickoffMutator

from .roles import is_kickoff

MATCH_SECONDS = 300.0
TICK_RATE = 120
BALL_ON_GROUND_Z = 100.0  # resting ball center is at 92.75
# Close games are the informative ones: a +3 lead with seconds left teaches little.
START_SCORE_WEIGHTS = {0: 0.35, 1: 0.22, -1: 0.22, 2: 0.07, -2: 0.07, 3: 0.035, -3: 0.035}


def new_match(blue=0, orange=0, time_remaining=MATCH_SECONDS):
    return {"blue": blue, "orange": orange, "score_diff": blue - orange, "time_remaining": float(time_remaining),
            "overtime": False, "over": False, "winner": None, "goal": None}


class RandomMatchStart(StateMutator):
    """Samples the starting score difference and clock into shared_info["match"]."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.diffs = np.array(list(START_SCORE_WEIGHTS))
        self.weights = np.array(list(START_SCORE_WEIGHTS.values()))

    def apply(self, state, shared_info):
        d = int(self.rng.choice(self.diffs, p=self.weights / self.weights.sum()))
        shared_info["match"] = new_match(max(d, 0), max(-d, 0), self.rng.uniform(1.0, MATCH_SECONDS))


class MatchEngine(TransitionEngine):
    """Wraps a transition engine: keeps score and clock, and resets to kickoff on the step after a goal."""

    def __init__(self, engine):
        self.engine = engine
        self.kickoff = KickoffMutator()
        self._kickoff_pending = False

    @property
    def agents(self):
        return self.engine.agents

    @property
    def max_num_agents(self):
        return self.engine.max_num_agents

    @property
    def state(self):
        return self.engine.state

    @property
    def config(self):
        return self.engine.config

    @config.setter
    def config(self, value):
        self.engine.config = value

    def create_base_state(self):
        return self.engine.create_base_state()

    def set_state(self, desired_state, shared_info):
        self._kickoff_pending = False
        shared_info.setdefault("match", new_match())
        return self.engine.set_state(desired_state, shared_info)

    def close(self):
        self.engine.close()

    def step(self, actions, shared_info):
        if self._kickoff_pending:
            self._kickoff_pending = False
            s = copy.deepcopy(self.engine.state)
            self.kickoff.apply(s, shared_info)
            s.boost_pad_timers = np.zeros_like(s.boost_pad_timers)
            for car in s.cars.values():
                car.demo_respawn_timer = 0.0
            self.engine.set_state(s, shared_info)
        prev_tick = self.engine.state.tick_count
        state = self.engine.step(actions, shared_info)
        self._update(shared_info["match"], state, (state.tick_count - prev_tick) / TICK_RATE)
        return state

    def _update(self, m, state, dt):
        m["goal"] = None
        if m["over"]:
            return
        if state.goal_scored:
            team = "blue" if state.scoring_team == 0 else "orange"
            m["goal"] = team
            m[team] += 1
            m["score_diff"] = m["blue"] - m["orange"]
            if m["overtime"] or (m["time_remaining"] <= 0 and m["score_diff"] != 0):
                self._finish(m)
            else:
                m["overtime"] = m["time_remaining"] <= 0
                self._kickoff_pending = True
            return
        if not m["overtime"] and not is_kickoff(state):
            m["time_remaining"] = max(m["time_remaining"] - dt, 0.0)
        if m["time_remaining"] <= 0 and not m["overtime"] and state.ball.position[2] < BALL_ON_GROUND_Z:
            if m["score_diff"] == 0:
                m["overtime"] = True
                self._kickoff_pending = True
            else:
                self._finish(m)

    @staticmethod
    def _finish(m):
        m["over"] = True
        m["winner"] = "blue" if m["score_diff"] > 0 else "orange"


class MatchOverCondition(DoneCondition):
    def reset(self, agents, initial_state, shared_info):
        pass

    def is_done(self, agents, state, shared_info):
        over = shared_info["match"]["over"]
        return {a: over for a in agents}
