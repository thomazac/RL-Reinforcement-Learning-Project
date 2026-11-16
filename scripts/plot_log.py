"""Plot a training log: touches, entropy, critic loss, throughput split, episode length.

    python scripts/plot_log.py runs/train/<run>/log.jsonl [--out plot.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(y, n=25):
    return np.convolve(y, np.ones(n) / n, mode="valid") if len(y) > n else np.asarray(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.log).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "Timing" in r and r["Timing"]["Overall Steps per Second"] > 1000]  # drop resume-timer artefacts
    steps = np.array([r["Timestep Collection"]["Cumulative Timesteps"] for r in rows]) / 1e6
    get = lambda *keys: np.array([float(_dig(r, keys)) for r in rows])
    touches = get("Play", "Touches per Player per Minute") if "Play" in rows[0] else \
        get("Reward Components", "Touches per Minute")
    panels = [
        ("Touches per car per minute", touches),
        ("Actor entropy", get("PPO Metrics", "Actor Entropy")),
        ("Critic loss", get("PPO Metrics", "Critic Loss")),
        ("Overall agent-steps/s", get("Timing", "Overall Steps per Second")),
        ("Seconds per iteration: collection (solid), update (dashed)", None),
        ("Episode length (steps, median)", get("PPO Metrics", "Natural Episode Length Median")),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (title, y) in zip(axes.flat, panels):
        if y is None:
            for key, style in (("Timestep Collection Time", "-"), ("Timestep Consumption Time", "--")):
                yk = get("Timing", key)
                ax.plot(steps[: len(smooth(yk))], smooth(yk), style, color="tab:blue")
        else:
            ax.plot(steps, y, color="0.85", lw=0.6)
            ax.plot(steps[: len(smooth(y))], smooth(y), color="tab:blue")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("agent-steps (M)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = args.out or str(Path(args.log).with_name("log.png"))
    fig.savefig(out, dpi=90)
    print(f"saved {out}")


def _dig(row, keys):
    for k in keys:
        row = row[k]
    return row


if __name__ == "__main__":
    main()
