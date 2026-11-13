"""Run each deliberately bad scripted team against the normal bot and count coordination events.

Each penalty should fire far more for its own variant than for the normal bot. For the demo variant,
blue's double-commits (direct plus revoked demo excuses) are split by whether its fixed hunter, blue-1,
touched the ball during the episode or within DEMO_EXCUSE_WINDOW after it; the "did not" part should
sit near the normal bot's rate.
"""

import argparse
import random

import numpy as np

from playstyle.coordination import DEMO_EXCUSE_WINDOW
from playstyle.env import TICK_SKIP, make_env
from playstyle.scripted import MODES, RoleBot

DT = TICK_SKIP / 120
HUNTER = "blue-1"
EVENTS = ("double_commit", "revoked", "excused_demo", "excused_save", "nobody_challenging", "nobody_home")


def flags(ev):
    return {
        "double_commit": ev["double_commit"] is not None,
        "revoked": ev["revoked"] is not None,
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
    split = {"hunter touched": 0, "did not": 0}
    episode = None  # blue's open double-commit episode: {"touched": bool, "left": s after it ends}
    steps = int(seconds / DT)
    for _ in range(steps):
        _, _, term, trunc = env.step(bot.act(env.state, env.shared_info))
        for team, ev in env.shared_info["coordination"].items():
            for k, on in flags(ev).items():
                active[team][k] += on
                starts[team][k] += on and not prev[team][k]
                prev[team][k] = on

        ev = env.shared_info["coordination"][0]
        hunter_touch = env.state.cars[HUNTER].ball_touches > 0
        if ev["revoked"] is not None:
            split["hunter touched"] += 1  # revoked only when a hunter touches; the demo variant's only hunter is blue-1
        if ev["double_commit"] is not None:
            episode = episode or {"touched": False}
            episode["left"] = DEMO_EXCUSE_WINDOW
        if episode is not None:
            episode["touched"] |= hunter_touch
            episode["left"] -= DT
            if episode["left"] < 0 or any(term.values()) or any(trunc.values()):
                split["hunter touched" if episode["touched"] else "did not"] += 1
                episode = None

        if any(term.values()):
            goals[0 if env.state.ball.position[1] > 0 else 1] += 1
        if any(term.values()) or any(trunc.values()):
            env.reset()
    minutes = seconds / 60
    rates = {t: {k: (starts[t][k] / minutes, active[t][k] / steps * 100) for k in EVENTS} for t in (0, 1)}
    return rates, goals, {k: v / minutes for k, v in split.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    print(f"Blue plays the variant, orange the normal bot; {args.minutes:g} sim-minutes x {args.seeds} seeds each.")
    print("Each cell: starts per minute / % of time active. Split: blue double-commits per minute by blue-1 touch.\n")
    print(f"{'blue mode':9s} {'goals B-O':>9s} " + " ".join(f"{e:>20s}" for e in EVENTS) + "   split (touched / not)")
    for mode in MODES:
        results = [run(mode, args.minutes * 60, seed) for seed in range(args.seeds)]
        for team, label in ((0, mode), (1, "  (orange)")):
            mean = {e: np.mean([r[0][team][e] for r in results], axis=0) for e in EVENTS}
            cells = " ".join(f"{mean[e][0]:9.2f} / {mean[e][1]:5.1f}%" for e in EVENTS)
            score = f"{sum(r[1][0] for r in results)}-{sum(r[1][1] for r in results)}" if team == 0 else ""
            split = (f"   {np.mean([r[2]['hunter touched'] for r in results]):.2f} / "
                     f"{np.mean([r[2]['did not'] for r in results]):.2f}") if team == 0 else ""
            print(f"{label:9s} {score:>9s} {cells}{split}")


if __name__ == "__main__":
    main()
