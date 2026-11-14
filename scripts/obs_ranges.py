"""Per-feature min, max and spread of the observation over the recorded reference, flagging scaling problems."""

import argparse
import lzma
import pickle

import numpy as np

from playstyle.obs import CAR_DIM, OBS_DIM
from playstyle.profile_v1 import FEATURES

XYZ = ("x", "y", "z")


def feature_names():
    car = [f"{v}.{a}" for v in ("pos", "vel", "angvel", "fwd", "up") for a in XYZ] + \
          [f"rel_{v}.{a}" for v in ("pos", "vel") for a in XYZ] + [f"ball_local.{a}" for a in ("fwd", "right", "up")] + \
          ["boost", "on_ground", "has_flip", "has_jump", "demoed", "ttb", "ttb_reachable"]
    assert len(car) == CAR_DIM
    names = [f"{who}.{f}" for who in ("self", "mate", "opp0", "opp1") for f in car]
    names += [f"ball.{v}.{a}" for v in ("pos", "vel", "angvel", "rel_pos", "rel_vel") for a in XYZ]
    names += [f"ball.pred{t}.{a}" for t in ("0.5", "1", "2") for a in XYZ]
    names += [f"pad{i}" for i in range(34)] + ["score", "time_left", "overtime", "kickoff"]
    names += [f"prev_action{i}" for i in range(8)] + [f"profile.{f}" for f in FEATURES]
    assert len(names) == OBS_DIM
    return names


LIMIT = 1.5


def main():
    """Recorded play and the hand-built block are reported side by side; spread is judged on both together."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagged", action="store_true", help="print only flagged features")
    args = ap.parse_args()
    with lzma.open("tests/fixtures/obs_v1_reference.pkl.xz", "rb") as f:
        ref = pickle.load(f)
    play = np.array([o for r in ref["rows"] for o in r["obs"].values()])
    hand = np.array([o for r in ref["hand_rows"] for o in r["obs"].values()])
    both = np.concatenate([play, hand])
    print(f"{len(play)} recorded + {len(hand)} hand-built observations x {both.shape[1]} features")
    print(f"{'feature':24s} {'play min':>9s} {'play max':>9s} {'hand min':>9s} {'hand max':>9s} {'std':>7s}  flag")
    worst = 0.0
    for i, name in enumerate(feature_names()):
        peak = np.abs(both[:, i]).max()
        worst = max(worst, peak)
        flag = "OUT OF RANGE" if peak > LIMIT else ("NO SPREAD" if both[:, i].std() < 0.01 else "")
        if flag or not args.flagged:
            print(f"{name:24s} {play[:, i].min():9.3f} {play[:, i].max():9.3f} {hand[:, i].min():9.3f} "
                  f"{hand[:, i].max():9.3f} {both[:, i].std():7.3f}  {flag}")
    print(f"largest magnitude anywhere: {worst:.3f}")


if __name__ == "__main__":
    main()
