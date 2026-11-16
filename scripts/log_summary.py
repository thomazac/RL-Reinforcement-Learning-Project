"""Summary of a training log (runs/train/<run>/log.jsonl): one row per iteration with the signals that say
whether a run is healthy.

    python scripts/log_summary.py runs/train/phase1a-smoke/log.jsonl [--every 10]
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--every", type=int, default=1, help="print every n-th iteration")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.log)]
    for line in rows:
        if "resume_check" in line:
            print(f"resume check: {line['resume_check']['differences'] or 'passed'}")
    rows = [r for r in rows if "PPO Metrics" in r]
    for i, r in enumerate(rows):
        if i % args.every and i != len(rows) - 1:
            continue
        ppo, rc, mon, res, timing = (r["PPO Metrics"], r["Reward Components"], r["Monitors"], r["Resources"],
                                     r["Timing"])
        weighted = "  ".join(f"{k.split(' ')[0]} {v:+.3f}" for k, v in rc.items() if "weighted" in k)
        if "Play" in r:  # Phase 1b
            play, ev = r["Play"], r["Per 5 Minutes (events)"]
            print(f"{r['Timestep Collection']['Cumulative Timesteps'] / 1e6:6.1f}M"
                  f"{' FROZEN' if mon.get('Actor Frozen') else ''}  touches/min {play['Touches per Player per Minute']:5.2f}  "
                  f"hit {play['Median Hit Strength (uu/s)']:5.0f}  per 5 min: goals {ev['Goals (both teams)']:4.2f} "
                  f"shots {ev['Shots per Player']:4.2f} saves {ev['Saves per Player']:4.2f}  "
                  f"double-commit {play['Double-Commit Share']:.2%}  empty net {play['Empty Net Share']:.1%}  "
                  f"KL {ppo['Mean KL Divergence']:.4f}  ratio {mon['First Minibatch Ratio']:.4f}  "
                  f"clip {mon['Reward Clip Share']:.2%} (max {mon['Largest Scaled Reward']:.1f})  "
                  f"entropy {ppo['Actor Entropy']:.3f}  critic {ppo['Critic Loss']:.3f}  "
                  f"sps {timing['Overall Steps per Second']:6,.0f}" + (f"  WARN {r['Warnings']}" if r.get("Warnings") else ""))
            continue
        print(f"{r['Timestep Collection']['Cumulative Timesteps'] / 1e6:6.1f}M  touches/min {rc.get('Touches per Minute', 0):5.2f}  "
              f"entropy {ppo['Actor Entropy']:.3f}  critic loss {ppo['Critic Loss']:.4f}  KL {ppo['Mean KL Divergence']:.5f}  "
              f"ratio {mon['First Minibatch Ratio']:.4f}  clip {mon['Reward Clip Share']:.2%} "
              f"(max {mon['Largest Scaled Reward']:.1f})  jump {mon['Largest Observation Jump']:.3f}  "
              f"sps {timing['Overall Steps per Second']:6,.0f}  mem {res['Main Process GB']:.1f}+{res['Env Processes GB']:.1f} GB  "
              f"gpu {res['GPU Reserved GB']:.1f} GB  | {weighted}"
              + (f"  WARN {r['Warnings']}" if r.get("Warnings") else ""))


if __name__ == "__main__":
    main()
