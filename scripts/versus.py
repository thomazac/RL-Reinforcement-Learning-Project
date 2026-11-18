"""Checkpoint against checkpoint: full matches from kickoff (5:00 on the clock, 0-0, until the match ends), one
policy per team, swapping sides every match. Reports A's wins and mean goal difference.

    python scripts/versus.py --a <checkpoint> --b <checkpoint> [--matches 10] [--json results.jsonl]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from playstyle.env import TICK_SKIP, build_env
from playstyle.match import TICK_RATE, new_match

MATCH_CAP = 900.0  # s of play: a full match is 5 minutes plus kickoffs and overtime


def play(env, policies, blue_is_a):
    """One match; returns (goals for A, goals for B, finished)."""
    import torch

    obs = env.reset()
    env.shared_info["match"] = new_match()
    obs = env.obs_builder.build_obs(env.agents, env.state, env.shared_info)
    teams = {"a": [x for x in env.agents if (env.state.cars[x].team_num == 0) == blue_is_a]}
    teams["b"] = [x for x in env.agents if x not in teams["a"]]
    for _ in range(int(MATCH_CAP * TICK_RATE / TICK_SKIP)):
        actions = {}
        with torch.no_grad():
            for side, agents in teams.items():
                chosen, _ = policies[side].get_action(None, [obs[x] for x in agents])
                actions.update({x: np.asarray(c).reshape(1) for x, c in zip(agents, chosen)})
        obs, _, term, _ = env.step(actions)  # truncations (no-touch timeout) are training-only
        if any(term.values()):
            break
    m = env.shared_info["match"]
    a, b = (m["blue"], m["orange"]) if blue_is_a else (m["orange"], m["blue"])
    return a, b, m["over"]


def main():
    import torch
    from watch import load_policy

    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--matches", type=int, default=10)
    ap.add_argument("--json", help="append the result to this file as one JSON line")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policies = {"a": load_policy(args.a, device), "b": load_policy(args.b, device)}
    env = build_env(coordination=True)
    scores = [play(env, policies, blue_is_a=i % 2 == 0) for i in range(args.matches)]
    a_goals, b_goals = np.array([s[0] for s in scores]), np.array([s[1] for s in scores])
    wins = int((a_goals > b_goals).sum())
    print(f"A {Path(args.a).name} vs B {Path(args.b).name}, {len(scores)} matches: A won {wins}, "
          f"goal difference {np.mean(a_goals - b_goals):+.2f} per match (A {a_goals.sum()} - {b_goals.sum()} B)"
          + ("" if all(s[2] for s in scores) else f"; {sum(not s[2] for s in scores)} stopped at the cap"))
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps({"eval": "versus", "checkpoint": str(args.a), "opponent": str(args.b),
                                "matches": len(scores), "a_wins": wins, "a_win_rate": wins / len(scores),
                                "goal_difference": float(np.mean(a_goals - b_goals)),
                                "scores": [[int(x), int(y)] for x, y, _ in scores]}) + "\n")


if __name__ == "__main__":
    main()
