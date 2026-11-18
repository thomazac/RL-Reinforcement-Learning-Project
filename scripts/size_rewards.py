"""Size Phase 1b reward weights from how often each term fires: play a checkpoint in the Phase 1b env, total
each raw component per agent per 5 minutes of play (4,500 steps; sampled matches are shorter, since their clocks
start between 1 s and 5 min), and pick the weight that gives the target total.

    python scripts/size_rewards.py --checkpoint <folder> [--minutes 30]

Targets (per agent per 5 minutes of play; a goal is 10): outcome terms are fixed (goal +10, concede -8, win/loss +-10); each
shaping term at most about a third of a goal; coordination penalties about a tenth of a goal. ball_to_goal is
zero-sum and averages zero, so it is sized on its mean absolute value.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from playstyle.env import TICK_SKIP, build_env
from playstyle.match import MATCH_SECONDS, TICK_RATE
from playstyle.rewards import PHASE_1B

STEPS_PER_MATCH = MATCH_SECONDS * TICK_RATE / TICK_SKIP
FIXED = {"goal": 10.0, "concede": -8.0, "win": 10.0, "shot": 0.0, "practice_shot": 0.0, "practice_defended": 0.0,
         "practice_level": 0.0, "both_near": 0.0, "path_approach": 0.015, "shot_touch": 1.0}
TARGETS = {"touch": 3.0, "air_touch": 1.5, "save": 2.0, "clear": 1.5, "boost": 2.0, "speed_to_ball": 1.0,
           "ball_to_goal": 3.0, "double_commit": -1.0, "nobody_challenging": -1.0, "nobody_home": -1.0, "rotation": 1.0}
SIGNED = ("ball_to_goal",)  # sized on mean absolute value: its signed total averages zero


def main():
    import torch
    from watch import load_policy

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--minutes", type=float, default=30)
    args = ap.parse_args()

    actor = load_policy(args.checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    env = build_env(phase="1b")
    names = env.reward_fn.names
    obs = env.reset()
    agents = list(env.agents)
    total, absolute, fired = (np.zeros(len(names)) for _ in range(3))
    steps = int(args.minutes * 60 * TICK_RATE / TICK_SKIP)
    for _ in range(steps):
        with torch.no_grad():
            chosen, _ = actor.get_action(None, [obs[a] for a in agents])
        obs, rewards, term, trunc = env.step({a: np.asarray(c).reshape(1) for a, c in zip(agents, chosen)})
        raw = np.stack([rewards[a][1:] for a in agents]).astype(np.float64)
        total += raw.sum(0)
        absolute += np.abs(raw).sum(0)
        fired += (raw != 0).sum(0)
        if any(term.values()) or any(trunc.values()):
            obs = env.reset()

    per_match = STEPS_PER_MATCH / (steps * len(agents))
    print(f"{args.minutes:.0f} sim-minutes; per agent per 5 minutes of play:")
    print(f"  {'term':20s} {'raw total':>10s} {'mean |raw|':>10s} {'fires':>8s} {'weight':>9s} {'weighted total':>14s}")
    weights = {}
    for i, n in enumerate(names):
        size = absolute[i] * per_match if n in SIGNED else total[i] * per_match
        if n in FIXED:
            weights[n] = FIXED[n]
        else:
            weights[n] = TARGETS[n] / size if size > 0 else PHASE_1B[n]
        print(f"  {n:20s} {total[i] * per_match:10.3f} {absolute[i] * per_match:10.3f} {fired[i] * per_match:8.1f} "
              f"{weights[n]:9.4g} {weights[n] * size:14.2f}")
    print("\nPHASE_1B = {" + ", ".join(f'"{n}": {weights[n]:.4g}' for n in names) + "}")


if __name__ == "__main__":
    main()
