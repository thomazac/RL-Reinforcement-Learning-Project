"""Where the second man stands, for humans (GC1 2s replays) and for checkpoints, in playstyle.rewards.rotation_spot's
coordinates: progress = how far up the field the ball is (0 at his own goal line, 1 at the opponent's), and fraction =
how far he stands along the line from his far post toward the ball (rotation_spot targets 0.15 + 0.4 progress).
The second man is the teammate farther from the ball. Fits fraction = intercept + slope * progress (least squares)
and prints the median fraction per progress band.

    python scripts/second_man_depth.py [--checkpoint NAME=folder ...] [--minutes 30]
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from playstyle.rewards import GOAL_Y, own_goal_sign

BANDS = np.linspace(0, 1, 6)


def depth(ball, second, team):
    """(progress, fraction) for one second man; ball and second are (x, y)."""
    sign = own_goal_sign(team)
    far_x = -np.sign(ball[0]) * 800.0 if abs(ball[0]) > 300 else 0.0
    post = np.array([far_x, sign * (GOAL_Y - 300)])
    progress = (-sign * ball[1] + GOAL_Y) / (2 * GOAL_Y)
    line = ball - post
    fraction = float((second - post) @ line / max(float(line @ line), 1e-6))
    return progress, fraction


def report(name, samples):
    p, f = np.array(samples).T
    keep = (p >= 0) & (p <= 1)
    p, f = p[keep], f[keep]
    slope, intercept = np.polyfit(p, f, 1)
    bands = ", ".join(f"{lo:.1f}-{hi:.1f}: {np.median(f[(p >= lo) & (p < hi)]):.2f}"
                      for lo, hi in zip(BANDS[:-1], BANDS[1:]) if ((p >= lo) & (p < hi)).any())
    print(f"{name}: fraction = {intercept:.2f} + {slope:.2f} progress  ({len(p):,} samples; median by progress {bands})")


def human(pattern):
    import subtr_actor as sa

    from extract_replay_states import usable_frames
    samples = []
    for f in sorted(glob.glob(pattern)):
        try:
            meta, arr = sa.get_ndarray_with_info_from_replay_filepath(
                f, global_feature_adders=["BallRigidBody", "CurrentTime"], fps=10)
        except Exception:
            continue
        rm = meta["replay_meta"]
        if len(rm["team_zero"]) != 2 or len(rm["team_one"]) != 2:
            continue
        cars = arr[:, 13:].reshape(len(arr), 4, 14)[:, :, 0:3]
        live = usable_frames(arr[:, 12], arr[:, 0:3], arr[:, 6:9], cars, pre_goal=0.0)
        for i in np.flatnonzero(live):
            ball = arr[i, 0:2]
            for team, idx in ((0, (0, 1)), (1, (2, 3))):
                d = [np.linalg.norm(cars[i, k] - arr[i, 0:3]) for k in idx]
                samples.append(depth(ball, cars[i, idx[int(np.argmax(d))], 0:2], team))
    return samples


def bots(folder, minutes):
    import torch

    from playstyle.env import TICK_SKIP, make_env
    from playstyle.match import TICK_RATE
    from watch import load_policy

    actor = load_policy(folder, "cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(coordination=True, seed=0)
    obs = env.reset()
    samples = []
    for step in range(int(minutes * 60 * TICK_RATE / TICK_SKIP)):
        with torch.no_grad():
            chosen, _ = actor.get_action(None, [obs[a] for a in env.agents])
        obs, _, term, trunc = env.step({a: np.asarray(c).reshape(1) for a, c in zip(env.agents, chosen)})
        if step % 2 == 0:  # ~7.5 per second, close to the replays' 10
            s = env.state
            ball = s.ball.position
            for team in (0, 1):
                mates = [c for c in s.cars.values() if c.team_num == team]
                if any(c.is_demoed for c in mates):
                    continue
                second = max(mates, key=lambda c: np.linalg.norm(c.physics.position - ball))
                samples.append(depth(ball[:2], second.physics.position[:2], team))
        if any(term.values()) or any(trunc.values()):
            obs = env.reset()
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default="data/replays/*.replay")
    ap.add_argument("--checkpoint", nargs="*", default=[], metavar="NAME=FOLDER")
    ap.add_argument("--minutes", type=float, default=30)
    args = ap.parse_args()
    print("rotation_spot target:  fraction = 0.15 + 0.40 progress")
    report("human GC1", human(args.replays))
    for c in args.checkpoint:
        name, folder = c.split("=", 1)
        report(name, bots(folder, args.minutes))


if __name__ == "__main__":
    main()
