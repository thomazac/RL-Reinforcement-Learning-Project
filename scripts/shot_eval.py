"""What happens on the defend_shot start states: outcome of each shot and what the keeper did.

Plays a checkpoint from shot starts only (playstyle.states.defend_shot), WINDOW seconds each, and classifies:
  save        - the save reward fired for a defender
  goal        - the shot (or anything after it) went in
  defended    - a defender touched the ball, no save fired, no goal (a block that was not a save, or a late touch)
  attacker    - an attacker touched the ball again before any defender (so the original shot no longer counts)
  wide        - the ball stopped being on course for the net with nobody touching it
  unresolved  - none of the above within the window
Keeper (the defender nearest its own net at the start): closest approach to the ball, whether it moved toward the
ball in the first second, whether it jumped, whether it touched the ball.

    python scripts/shot_eval.py --checkpoint <folder> [--shots 300]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from playstyle.ball_prediction import SLICE_TIMES
from playstyle.env import REPLAY_STATES, TICK_SKIP, build_env
from playstyle.match import TICK_RATE
from playstyle.rewards import SHOT_HORIZON, enters_net
from playstyle.states import MixedStart, ShotCurriculum

WINDOW = 8.0
DT = TICK_SKIP / TICK_RATE


def main():
    import torch
    from watch import load_policy

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shots", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=float, default=1.0,
                    help="shot difficulty (playstyle.states.defend_shot); 1 is the full range, as in earlier evals")
    ap.add_argument("--json", help="append the results to this file as one JSON line")
    args = ap.parse_args()

    actor = load_policy(args.checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    env = build_env(phase="1b")
    mutators = list(env.state_mutator.mutators)
    mutators[1] = MixedStart(env.transition_engine.kickoff, REPLAY_STATES, seed=args.seed, weights={"defend_shot": 1.0},
                             curriculum=ShotCurriculum(level=args.level))
    env.state_mutator.mutators = tuple(mutators)
    names = env.reward_fn.names
    save_i = names.index("save")
    penalties = ("double_commit", "nobody_challenging", "nobody_home")
    penalty_steps = {(p, who): 0 for p in penalties for who in ("keeper", "other defender")}
    shots_with = Counter()
    outcomes, keeper_stats, reachable, toward_path = Counter(), [], [], []
    total_steps = 0
    for _ in range(args.shots):
        obs = env.reset()
        state = env.state
        agents = list(env.agents)
        defending = 0 if state.ball.linear_velocity[1] < 0 else 1
        own_goal = np.array([0.0, (-1 if defending == 0 else 1) * 5120.0, 0.0])
        defenders = [a for a in agents if state.cars[a].team_num == defending]
        keeper = min(defenders, key=lambda a: np.linalg.norm(state.cars[a].physics.position - own_goal))
        start_dist = float(np.linalg.norm(state.cars[keeper].physics.position - state.ball.position))
        on_target = enters_net(env.shared_info["ball_slices"], defending, SHOT_HORIZON)
        slices = np.asarray(env.shared_info["ball_slices"])
        sign = -1 if defending == 0 else 1
        crossing = np.flatnonzero(slices[:, 1] * sign > 5120)
        arrival = float(SLICE_TIMES[crossing[0]]) if len(crossing) else np.inf
        reachable.append(float(env.shared_info["ttb"][keeper]) < arrival)
        cross_point = slices[crossing[0], :2] if len(crossing) else own_goal[:2]
        start_keeper = state.cars[keeper].physics.position[:2].copy()
        fired = set()
        outcome, closest, moved, jumped, touched, defender_touched = None, start_dist, None, False, False, False
        for step in range(int(WINDOW / DT)):
            with torch.no_grad():
                chosen, _ = actor.get_action(None, [obs[a] for a in agents])
            obs, rewards, term, trunc = env.step({a: np.asarray(c).reshape(1) for a, c in zip(agents, chosen)})
            state = env.state
            k = state.cars[keeper]
            d = float(np.linalg.norm(k.physics.position - state.ball.position))
            closest = min(closest, d)
            total_steps += 1
            for a in defenders:
                who = "keeper" if a == keeper else "other defender"
                for p in penalties:
                    if rewards[a][1 + names.index(p)] != 0:
                        penalty_steps[(p, who)] += 1
                        fired.add((p, who))
            if moved is None and (step + 1) * DT >= 1.0:
                moved = d < start_dist - 100
                before = np.linalg.norm(start_keeper - cross_point)
                toward_path.append(np.linalg.norm(k.physics.position[:2] - cross_point) < before - 100)
            jumped |= not k.on_ground
            touched |= k.ball_touches > 0
            if any(rewards[a][1 + save_i] > 0 for a in defenders):
                outcome = "save"
            elif state.goal_scored:
                outcome = "goal" if state.scoring_team != defending else "goal (other net)"
            elif any(state.cars[a].ball_touches > 0 for a in defenders):
                defender_touched = True
            elif not defender_touched and any(state.cars[a].ball_touches > 0 for a in agents if a not in defenders):
                if step > 2:  # the shooter starts right behind the ball and may graze it at once
                    outcome = "attacker"
            if outcome is None and not defender_touched \
                    and not enters_net(env.shared_info["ball_slices"], defending, SHOT_HORIZON) and step > 2:
                outcome = "wide"
            if outcome is not None or any(term.values()) or any(trunc.values()):
                break
        if outcome is None:
            outcome = "defended" if defender_touched else "unresolved"
        outcomes[outcome] += 1
        for f in fired:
            shots_with[f] += 1
        keeper_stats.append((start_dist, closest, bool(moved), jumped, touched, on_target, outcome))

    n = sum(outcomes.values())
    print(f"{n} shot starts, {Path(args.checkpoint).name}")
    print("outcomes: " + ", ".join(f"{k} {v / n:.0%}" for k, v in outcomes.most_common()))
    ks = np.array([(s, c, m, j, t, o) for s, c, m, j, t, o, _ in keeper_stats], dtype=float)
    print(f"on target at the start (ball prediction): {ks[:, 5].mean():.0%}; keeper's estimated time to ball shorter "
          f"than the shot's arrival: {np.mean(reachable):.0%}")
    print(f"keeper: start {np.median(ks[:, 0]):.0f} uu from the ball, closest approach median {np.median(ks[:, 1]):.0f} uu; "
          f"moved toward the ball in the first second {ks[:, 2].mean():.0%}; jumped {ks[:, 3].mean():.0%}; "
          f"touched the ball {ks[:, 4].mean():.0%}")
    print("coordination penalties on defenders during practice shots (share of shots where it fired at least once; "
          "share of defender-steps):")
    for (p, who), steps in penalty_steps.items():
        print(f"  {p:20s} {who:15s} {shots_with[(p, who)] / n:5.1%} of shots, {steps / total_steps:5.1%} of steps")
    reach = np.array(reachable, dtype=bool)
    failed = np.array([r[6] == "goal" for r in keeper_stats]) & reach
    if failed.any():
        fk = np.array([r[:5] for r in keeper_stats], dtype=float)[failed]
        path = np.array(toward_path + [False] * (len(keeper_stats) - len(toward_path)), dtype=bool)[failed]
        print(f"failed reachable shots ({failed.sum()}): keeper moved toward the ball {fk[:, 2].mean():.0%}, toward the "
              f"shot's goal-line crossing point {path.mean():.0%}, jumped {fk[:, 3].mean():.0%}, touched {fk[:, 4].mean():.0%}, "
              f"closest approach median {np.median(fk[:, 1]):.0f} uu")
    if args.json:
        result = {"eval": "shots", "checkpoint": str(args.checkpoint), "level": args.level, "shots": n,
                  **{f"outcome_{k}": v / n for k, v in outcomes.items()},
                  "keeper_moved_toward_ball": float(ks[:, 2].mean()), "keeper_jumped": float(ks[:, 3].mean()),
                  "keeper_touched": float(ks[:, 4].mean()),
                  "keeper_toward_crossing_point": float(np.mean(toward_path)) if toward_path else 0.0}
        with open(args.json, "a") as f:
            f.write(json.dumps(result) + "\n")
    for name in ("goal", "save", "defended"):
        rows = [r for r in keeper_stats if r[6] == name]
        if rows:
            r = np.array([x[:5] for x in rows], dtype=float)
            print(f"  on {name:8s} ({len(rows):3d}): keeper closest {np.median(r[:, 1]):5.0f} uu, moved {r[:, 2].mean():.0%}, "
                  f"jumped {r[:, 3].mean():.0%}, touched {r[:, 4].mean():.0%}")


if __name__ == "__main__":
    main()
