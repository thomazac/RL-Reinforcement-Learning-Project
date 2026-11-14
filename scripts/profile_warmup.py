"""Population averages of the teammate-profile indicators from scripted normal-vs-normal play.

Used once to set profile_v1.WARMUP, which is then frozen with the observation format.
"""

import argparse
import random

import numpy as np

from playstyle import profile_v1
from playstyle.env import TICK_SKIP, make_env
from playstyle.scripted import RoleBot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    env, bot = make_env(seed=args.seed), RoleBot(seed=args.seed)
    env.reset()
    dt = TICK_SKIP / 120
    demoed = {a: False for a in env.agents}
    rows, team_touch = [], []
    for _ in range(int(args.minutes * 60 / dt)):
        _, _, term, trunc = env.step(bot.act(env.state, env.shared_info))
        s = env.state
        newly = [a for a, c in s.cars.items() if c.is_demoed and not demoed[a]]
        demoed = {a: c.is_demoed for a, c in s.cars.items()}
        x, touch = profile_v1.indicators(s, env.shared_info["obs_ttb"], newly, dt)
        rows += list(x.values())
        team_touch += list(touch.values())
        if any(term.values()) or any(trunc.values()):
            env.reset()
    mean = np.nanmean(np.array(rows), axis=0)
    rate = float(np.mean(team_touch))
    mean[profile_v1.FEATURES.index("touch_share")] = mean[profile_v1.FEATURES.index("touch_share")] / rate
    for name, v in zip(profile_v1.FEATURES, mean):
        print(f"  {name:12s} {v:.3f}")
    print("WARMUP = np.array([" + ", ".join(f"{v:.3f}" for v in mean) + "])")
    print(f"TEAM_TOUCH_RATE = {rate:.4f}  # {rate / dt:.2f} team touches per second")


if __name__ == "__main__":
    main()
