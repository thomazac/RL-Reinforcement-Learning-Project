"""Per-feature min, max and spread of the observation over the recorded reference, flagging scaling problems."""

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


def main():
    with lzma.open("tests/fixtures/obs_v1_reference.pkl.xz", "rb") as f:
        rows = pickle.load(f)["rows"]
    x = np.array([o for r in rows for o in r["obs"].values()])
    lo, hi, sd = x.min(0), x.max(0), x.std(0)
    print(f"{len(x)} observations x {x.shape[1]} features")
    print(f"{'feature':24s} {'min':>8s} {'max':>8s} {'std':>7s}  flag")
    for name, a, b, s in zip(feature_names(), lo, hi, sd):
        flag = "OUT OF RANGE" if max(abs(a), abs(b)) > 1.5 else ("NO SPREAD" if s < 0.01 else "")
        print(f"{name:24s} {a:8.3f} {b:8.3f} {s:7.3f}  {flag}")


if __name__ == "__main__":
    main()
