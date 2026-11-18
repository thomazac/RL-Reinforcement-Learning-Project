"""Round robin between checkpoints: full matches from kickoff (5:00, 0-0, until each match ends), every pair, sides
alternated.

Fast because matches run in lockstep: each worker process plays --per-worker matches at once and, every step, runs
one batched forward pass per checkpoint over all the cars it controls across those matches (on the GPU). Running
policies one car pair at a time is about 30 s per match on the GPU and minutes on a single CPU thread.

Per pair: A's wins and mean goal difference with its standard error. Per checkpoint: mean goal difference against all
the others, and how often both of its cars were within 1,000 uu of the ball (per team; human GC1 live play: 3.9%).

    python scripts/round_robin.py --checkpoints NAME=folder NAME=folder ... [--matches 200] [--workers 6]
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

MATCH_CAP = 900.0  # s of play
NEAR = 1000.0


def play_batch(args):
    """Play a batch of (a, b, blue_is_a) matches in lockstep. Returns one result tuple per match:
    (a, b, goals a, goals b, finished, both-near steps for a, for b, steps)."""
    checkpoints, tasks, seed = args
    import torch

    from playstyle.env import TICK_SKIP, make_env
    from playstyle.match import TICK_RATE, new_match
    from watch import load_policy

    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policies = {n: load_policy(f, device) for n, f in checkpoints.items()}
    games = []
    for k, (a, b, blue_is_a) in enumerate(tasks):
        env = make_env(coordination=True, seed=seed * 1000 + k)  # kickoff starts, as the policies' matches begin
        obs = env.reset()
        env.shared_info["match"] = new_match()
        obs = env.obs_builder.build_obs(env.agents, env.state, env.shared_info)
        side = {x: (a if (env.state.cars[x].team_num == 0) == blue_is_a else b) for x in env.agents}
        games.append({"env": env, "obs": obs, "side": side, "task": (a, b, blue_is_a), "done": False,
                      "near": {a: 0, b: 0}, "steps": 0})
    for _ in range(int(MATCH_CAP * TICK_RATE / TICK_SKIP)):
        live = [g for g in games if not g["done"]]
        if not live:
            break
        actions = {id(g): {} for g in live}
        with torch.no_grad():
            for name, policy in policies.items():
                slots = [(g, x) for g in live for x in g["env"].agents if g["side"][x] == name]
                if not slots:
                    continue
                chosen, _ = policy.get_action(None, [g["obs"][x] for g, x in slots])
                for (g, x), c in zip(slots, chosen):
                    actions[id(g)][x] = np.asarray(c).reshape(1)
        for g in live:
            env = g["env"]
            g["obs"], _, term, _ = env.step(actions[id(g)])  # truncations (no-touch timeout) are training-only
            g["steps"] += 1
            ball = env.state.ball.position
            for name in set(g["side"].values()):
                cars = [env.state.cars[x] for x in env.agents if g["side"][x] == name]
                g["near"][name] += all(not c.is_demoed and np.linalg.norm(c.physics.position - ball) < NEAR
                                       for c in cars)
            g["done"] = any(term.values())
    results = []
    for g in games:
        a, b, blue_is_a = g["task"]
        m = g["env"].shared_info["match"]
        ga, gb = (m["blue"], m["orange"]) if blue_is_a else (m["orange"], m["blue"])
        results.append((a, b, ga, gb, m["over"], g["near"][a], g["near"][b], g["steps"]))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, metavar="NAME=FOLDER")
    ap.add_argument("--matches", type=int, default=200, help="per pair")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--per-worker", type=int, default=25, help="matches each worker plays at once")
    ap.add_argument("--json", default=str(ROOT / "runs" / "round_robin.jsonl"))
    args = ap.parse_args()
    checkpoints = dict(c.split("=", 1) for c in args.checkpoints)
    names = list(checkpoints)
    tasks = [(a, b, i % 2 == 0) for a, b in itertools.combinations(names, 2) for i in range(args.matches)]
    batches = [(checkpoints, tasks[i:i + args.per_worker], i) for i in range(0, len(tasks), args.per_worker)]
    print(f"{len(tasks)} matches in {len(batches)} batches of up to {args.per_worker}, {args.workers} workers", flush=True)
    start, results = time.time(), []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for batch in pool.imap_unordered(play_batch, batches):
            results += batch
            print(f"  {len(results)}/{len(tasks)} matches, {(time.time() - start) / 60:.1f} min", flush=True)

    print(f"\n{len(results)} matches in {(time.time() - start) / 60:.1f} min; "
          f"{sum(not r[4] for r in results)} stopped at the {MATCH_CAP:.0f} s cap")
    summary = {"checkpoints": checkpoints, "matches_per_pair": args.matches, "pairs": {}, "overall": {}}
    for a, b in itertools.combinations(names, 2):
        rows = [r for r in results if (r[0], r[1]) == (a, b)]
        diff = np.array([r[2] - r[3] for r in rows], dtype=float)
        wins = int((diff > 0).sum())
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
        summary["pairs"][f"{a} vs {b}"] = {"a_wins": wins, "matches": len(rows), "goal_diff": float(diff.mean()),
                                            "goal_diff_se": float(se)}
        print(f"  {a:>12s} vs {b:<12s} {a} won {wins:3d}/{len(rows)}  goal difference {diff.mean():+.2f} "
              f"+- {se:.2f} per match")
    print()
    for n in names:
        diffs, near, team_steps = [], 0, 0
        for r in results:
            if n in (r[0], r[1]):
                diffs.append((r[2] - r[3]) if r[0] == n else (r[3] - r[2]))
                near += r[5] if r[0] == n else r[6]
                team_steps += r[7]
        d = np.array(diffs, dtype=float)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        summary["overall"][n] = {"goal_diff": float(d.mean()), "goal_diff_se": float(se),
                                 "both_near_share": near / team_steps}
        print(f"  {n:>12s} overall goal difference {d.mean():+.2f} +- {se:.2f}; "
              f"both cars near the ball {near / team_steps:.1%} of the time")
    with open(args.json, "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
