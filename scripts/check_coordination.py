"""Run each deliberately bad scripted team against the normal bot and count coordination events.

Each penalty should fire far more for its own variant than for the normal bot, and the demo
variant's second man should be excused rather than penalized for double-commit.
"""

import argparse
import random

import numpy as np

from playstyle.env import TICK_SKIP, make_env
from playstyle.scripted import MODES, RoleBot

DT = TICK_SKIP / 120
EVENTS = ("double_commit", "excused_demo", "excused_save", "nobody_challenging", "nobody_home")


def flags(ev):
    return {
        "double_commit": ev["double_commit"] is not None,
        "excused_demo": ev["excused"] == "demo",
        "excused_save": ev["excused"] == "save",
        "nobody_challenging": ev["nobody_challenging"] is not None,
        "nobody_home": ev["nobody_home"],
    }


def run(mode, seconds, seed):
    random.seed(seed)
    np.random.seed(seed)
    env, bot = make_env(), RoleBot(seed=seed, team_modes={0: mode, 1: "normal"})
    env.reset()
    active = {t: dict.fromkeys(EVENTS, 0) for t in (0, 1)}
    starts = {t: dict.fromkeys(EVENTS, 0) for t in (0, 1)}
    prev = {t: dict.fromkeys(EVENTS, False) for t in (0, 1)}
    goals = {0: 0, 1: 0}
    steps = int(seconds / DT)
    for _ in range(steps):
        _, _, term, trunc = env.step(bot.act(env.state, env.shared_info))
        for team, ev in env.shared_info["coordination"].items():
            for k, on in flags(ev).items():
                active[team][k] += on
                starts[team][k] += on and not prev[team][k]
                prev[team][k] = on
        if any(term.values()):
            goals[0 if env.state.ball.position[1] > 0 else 1] += 1
        if any(term.values()) or any(trunc.values()):
            env.reset()
    minutes = seconds / 60
    return {t: {k: (starts[t][k] / minutes, active[t][k] / steps * 100) for k in EVENTS} for t in (0, 1)}, goals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Blue plays the variant, orange the normal bot; {args.minutes:g} sim-minutes each.")
    print("Each cell: starts per minute / % of time active.\n")
    header = f"{'blue mode':9s} {'goals B-O':>9s} " + " ".join(f"{e:>20s}" for e in EVENTS)
    print(header)
    for mode in MODES:
        result, goals = run(mode, args.minutes * 60, args.seed)
        for team, label in ((0, mode), (1, "  (orange)")):
            cells = " ".join(f"{result[team][e][0]:9.1f} / {result[team][e][1]:5.1f}%" for e in EVENTS)
            score = f"{goals[0]}-{goals[1]}" if team == 0 else ""
            print(f"{label:9s} {score:>9s} {cells}")


if __name__ == "__main__":
    main()
