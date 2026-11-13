"""Measure real turn-and-drive times to a still ball and fit the speed-dependent turn penalty.

Every trial starts facing away from the ball, so unlike the replay fit there is no selection
bias toward cars already facing it. The scripted driver never half-flips, so measured times
are an upper bound on what a good player or trained bot needs.
"""

import itertools

import numpy as np
from rlgym.api import RLGym, StateMutator
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence

import playstyle.ttb as T
from playstyle.ball_prediction import SLICE_TIMES
from playstyle.scripted import RoleBot

SPEEDS = (0, 500, 1000, 1500, 2000, 2300)
ANGLES_DEG = range(0, 181, 15)
DISTANCES = (1500, 3000)
BALL = np.array([0.0, 1500.0, 92.75])
MAX_TICKS = 720


class TurnTrial(StateMutator):
    def __init__(self):
        self.speed = self.angle = self.dist = 0.0

    def apply(self, state, shared_info):
        state.ball.position = BALL.astype(np.float32)
        state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        car = next(iter(state.cars.values()))
        yaw = np.pi / 2 + self.angle
        car.physics.position = np.array([0, BALL[1] - self.dist, 17], dtype=np.float32)
        car.physics.linear_velocity = np.array([np.cos(yaw), np.sin(yaw), 0], dtype=np.float32) * self.speed
        car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
        car.physics.euler_angles = np.array([0, yaw, 0], dtype=np.float32)
        car.boost_amount = 100.0


def run_trials():
    trial = TurnTrial()
    env = RLGym(state_mutator=MutatorSequence(FixedTeamSizeMutator(1, 0), trial), obs_builder=DefaultObs(zero_padding=None),
                action_parser=RepeatAction(LookupTableAction(), repeats=1), reward_fn=GoalReward(),
                termination_cond=GoalCondition(), transition_engine=RocketSimEngine())
    rows = []
    for speed, angle_deg, dist in itertools.product(SPEEDS, ANGLES_DEG, DISTANCES):
        trial.speed, trial.angle, trial.dist = speed, np.radians(angle_deg), dist
        env.reset()
        agent = env.agents[0]
        c = env.state.cars[agent]
        start = (c.physics.position.copy(), c.physics.linear_velocity.copy(), c.physics.forward.copy(), c.boost_amount)
        touch_tick = None
        for tick in range(1, MAX_TICKS + 1):
            env.step({agent: np.array([RoleBot._drive(env.state.cars[agent], BALL, boost=True)])})
            if env.state.cars[agent].ball_touches > 0:
                touch_tick = tick
                break
        rows.append((speed, angle_deg, dist, touch_tick / 120 if touch_tick else np.inf, *start))
    return rows


def predict(rows):
    pos, vel, fwd, boost = (np.array([r[i] for r in rows]) for i in (4, 5, 6, 7))
    slices = np.tile(BALL, (len(SLICE_TIMES), 1))
    return T.estimate_ttb(pos, vel, fwd, boost, np.zeros(len(rows), bool), slices)


def mae(rows, measured):
    pred = predict(rows)
    return np.mean(np.minimum(np.abs(pred - measured), 2.0))


def main():
    rows = run_trials()
    measured = np.array([r[3] for r in rows])
    print(f"{len(rows)} trials, {np.isinf(measured).sum()} without a touch in {MAX_TICKS / 120:.0f}s")

    current = (T.TURN_TIME_LOW, T.TURN_TIME_HIGH, T.INSIDE_DECEL)
    print(f"current constants {current}: MAE {mae(rows, measured):.3f}s")
    best = (np.inf, None)
    for params in itertools.product(np.arange(0.2, 0.42, 0.02), np.arange(0.3, 0.9, 0.05), (1200, 1600, 2000, 2300, 3000, 4000)):
        T.TURN_TIME_LOW, T.TURN_TIME_HIGH, T.INSIDE_DECEL = params
        score = mae(rows, measured)
        if score < best[0]:
            best = (score, tuple(round(float(p), 2) for p in params))
    print(f"best fit TURN_TIME_LOW, TURN_TIME_HIGH, INSIDE_DECEL = {best[1]}: MAE {best[0]:.3f}s")
    T.TURN_TIME_LOW, T.TURN_TIME_HIGH, T.INSIDE_DECEL = current
    after = predict(rows)
    T.TURN_TIME_LOW, T.TURN_TIME_HIGH, T.INSIDE_DECEL = 0.15, 0.15, 1e9
    before = predict(rows)
    T.TURN_TIME_LOW, T.TURN_TIME_HIGH, T.INSIDE_DECEL = current

    print(f"\n{'speed':>5} {'angle':>5} {'dist':>5} {'measured':>8} {'flat .15':>8} {'current':>8}")
    for r, m, b, a in zip(rows, measured, before, after):
        if r[2] == 3000 and r[1] % 45 == 0:
            print(f"{r[0]:5d} {r[1]:5d} {r[2]:5d} {m:8.2f} {b:8.2f} {a:8.2f}")
    for name, p in (("flat 0.15", before), ("current", after)):
        err = np.minimum(np.abs(p - measured), 2.0)
        print(f"{name} model: MAE {err.mean():.3f}s, facing away (>=90 deg) MAE "
              f"{err[np.array([r[1] >= 90 for r in rows])].mean():.3f}s, bias {np.mean(p[np.isfinite(p)] - measured[np.isfinite(p)]):+.3f}s")


if __name__ == "__main__":
    main()
