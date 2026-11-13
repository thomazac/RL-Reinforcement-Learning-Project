"""Compare predicted TTB with actual time-to-touch in human replays.

For every non-kickoff touch, sample the frame `lead` seconds earlier (only if nobody touched
the ball in between, so the ball path is undisturbed) and record the toucher's predicted TTB.

    python scripts/validate_ttb.py             # collect (cached) and report
    python scripts/validate_ttb.py --calibrate # fit constants on half the replays, report on the other half
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import subtr_actor as sa

from playstyle import ttb as ttb_module
from playstyle.ball_prediction import BallPredictor
from playstyle.ttb import estimate_ttb

LEADS = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5)
N_PLAYER_COLS = 14  # see subtr_actor.get_column_headers()
CACHE = Path("data/ttb_samples.npz")
SEARCH = {
    "CONTACT_DIST": [80, 110, 140, 170, 200, 240],
    "JUMP_TIME": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
    "AERIAL_RISE_SPEED": [500, 700, 900, 1200, 1500, 2000],
    "MIN_AERIAL_BOOST": [0, 5, 10, 20],
    "GROUND_REACH_DZ": [110, 130, 145, 165, 190],
    "JUMP_REACH_DZ": [220, 250, 285, 320, 360],
}


def player_index(remote_ids, primary):
    for i, rid in enumerate(remote_ids):
        if any(str(rid.get(k)) == str(v) for k, v in primary.items()):
            return i
    return None


def touches(path, remote_ids):
    out = []
    for e in sa.get_stats_events(path)["events"]:
        m = e["meta"]
        if m["stream"] != "touch":
            continue
        timing = m["timing"]
        p = player_index(remote_ids, m["primary_player"])
        if p is not None:
            out.append((timing.get("start_time", timing.get("time")), p))
    return sorted(out)


def samples_from_replay(path, replay_idx, predictor):
    meta, arr = sa.get_ndarray_with_info_from_replay_filepath(
        path, global_feature_adders=["BallRigidBody", "CurrentTime"], fps=30)
    rm = meta["replay_meta"]
    players = rm["team_zero"] + rm["team_one"]
    if len(players) != 4:
        return []
    team = np.array([0] * len(rm["team_zero"]) + [1] * len(rm["team_one"]))
    remote_ids = [p["remote_id"] for p in players]
    time = arr[:, 12]
    cars = arr[:, 13:].reshape(len(arr), 4, N_PLAYER_COLS)

    out, prev_t = [], -np.inf
    for t_touch, p in touches(path, remote_ids):
        ball_before = arr[min(np.searchsorted(time, t_touch - 0.1), len(arr) - 1)]
        is_kickoff = np.linalg.norm(ball_before[6:9]) < 1 and np.linalg.norm(ball_before[:2]) < 10
        for lead in LEADS:
            t0 = t_touch - lead
            if is_kickoff or t0 < prev_t + 0.1 or t0 < time[0]:
                continue
            b, c = arr[np.searchsorted(time, t0)], cars[np.searchsorted(time, t0)]
            yaw = c[:, 5]
            out.append(dict(
                replay=replay_idx, lead=lead, toucher=p,
                mate=next(j for j in range(4) if j != p and team[j] == team[p]),
                pos=c[:, 0:3], vel=c[:, 6:9],
                fwd=np.stack([np.cos(yaw), np.sin(yaw), np.zeros(4)], axis=1),
                boost=c[:, 12] * 100 / 255, demoed=np.all(c[:, 0:3] == 0, axis=1),
                slices=predictor.predict(b[0:3], b[6:9], b[9:12]),
            ))
        prev_t = t_touch
    return out


def collect(pattern):
    predictor, rows = BallPredictor(), []
    files = sorted(glob.glob(pattern))
    for i, f in enumerate(files):
        try:
            rows += samples_from_replay(f, i, predictor)
        except Exception as e:  # a few replays fail to parse; skip them
            print(f"skip {Path(f).name}: {e}")
    samples = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **samples)
    print(f"{len(files)} replays, {len(rows)} samples cached to {CACHE}")
    return samples


def predict_all(s):
    n = len(s["lead"])
    flat = lambda a: a.reshape(n * 4, *a.shape[2:])
    t = estimate_ttb(flat(s["pos"]), flat(s["vel"]), flat(s["fwd"]), flat(s["boost"]), flat(s["demoed"]),
                     np.repeat(s["slices"], 4, axis=0))
    return t.reshape(n, 4)


def objective(s, ttb):
    """Mean absolute error for committed touches (lead <= 1.5 s); unreachable counts as 1.5 s off."""
    m = s["lead"] <= 1.5
    pred = ttb[np.arange(len(ttb)), s["toucher"]][m]
    return np.minimum(np.abs(pred - s["lead"][m]), 1.5).mean()


def report(s, title):
    ttb = predict_all(s)
    idx = np.arange(len(ttb))
    pred, mate, best = ttb[idx, s["toucher"]], ttb[idx, s["mate"]], ttb.min(axis=1)
    lead = s["lead"]
    z = s["slices"][idx, np.minimum(np.rint(lead / 0.1).astype(int), s["slices"].shape[1] - 1), 2]
    print(f"\n{title}: {len(lead)} samples, objective {objective(s, ttb):.3f}")
    print(f"{'lead':>5} {'n':>6} {'inf%':>6} {'bias':>7} {'MAE':>6} {'toucher<=mate%':>14} {'toucher=min4%':>13}")
    for L in LEADS:
        m = lead == L
        fin = m & np.isfinite(pred)
        err = pred[fin] - L
        print(f"{L:5.2f} {m.sum():6d} {100 * (1 - fin.sum() / max(m.sum(), 1)):6.1f} {err.mean():+7.2f} "
              f"{np.abs(err).mean():6.2f} {(pred[m] <= mate[m]).mean() * 100:14.1f} "
              f"{(pred[m] <= best[m]).mean() * 100:13.1f}")
    print("By ball height at the touch (lead <= 1.0 s):")
    for name, lo, hi in (("ground", 0, 160), ("jump", 160, 300), ("aerial", 300, 1e9)):
        m = (lead <= 1.0) & (z >= lo) & (z < hi)
        fin = m & np.isfinite(pred)
        if m.sum():
            err = pred[fin] - lead[fin]
            print(f"  {name:7s} n={m.sum():5d} inf%={100 * (1 - fin.sum() / m.sum()):5.1f} "
                  f"bias={err.mean():+.2f} MAE={np.abs(err).mean():.2f}")
    # A human did reach the ball in `lead`, so positive error means the model is too slow; negative error
    # can just be a human not hurrying.
    print("Ground touches (lead <= 1.0 s) by toucher's angle to the ball at the touch, and speed:")
    car = s["pos"][idx, s["toucher"]]
    fwd, vel = s["fwd"][idx, s["toucher"]], s["vel"][idx, s["toucher"]]
    to_ball = s["slices"][idx, np.minimum(np.rint(lead / 0.1).astype(int), s["slices"].shape[1] - 1)] - car
    cos = (fwd[:, :2] * to_ball[:, :2]).sum(1) / np.maximum(np.linalg.norm(to_ball[:, :2], axis=1), 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    speed = np.linalg.norm(vel, axis=1)
    ground = (lead <= 1.0) & (z < 160) & (car[:, 2] < 60) & np.isfinite(pred)
    for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 135), (135, 181)):
        for slo, shi in ((0, 1000), (1000, 1800), (1800, 9999)):
            m = ground & (angle >= lo) & (angle < hi) & (speed >= slo) & (speed < shi)
            if m.sum() >= 20:
                err = pred[m] - lead[m]
                print(f"  {lo:3d}-{min(hi, 180):3d} deg, speed {slo:4d}-{shi:4d}: n={m.sum():5d} "
                      f"bias={err.mean():+.2f} median={np.median(err):+.2f} too slow by >0.2 s: {(err > 0.2).mean() * 100:4.1f}%")


def calibrate(s, rounds=3):
    best = objective(s, predict_all(s))
    for _ in range(rounds):
        for name, values in SEARCH.items():
            for v in values:
                old = getattr(ttb_module, name)
                setattr(ttb_module, name, v)
                score = objective(s, predict_all(s))
                if score < best - 1e-4:
                    best = score
                else:
                    setattr(ttb_module, name, old)
    return {name: getattr(ttb_module, name) for name in SEARCH}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default="data/replays/*.replay")
    ap.add_argument("--recollect", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    s = dict(np.load(CACHE)) if CACHE.exists() and not args.recollect else collect(args.replays)
    if not args.calibrate:
        report(s, "All replays")
        return
    fit = {k: v[s["replay"] % 2 == 0] for k, v in s.items()}
    held = {k: v[s["replay"] % 2 == 1] for k, v in s.items()}
    report(held, "Held-out, current constants")
    params = calibrate(fit)
    print("\nFitted constants:")
    for k, v in params.items():
        print(f"  {k} = {v}")
    report(held, "Held-out, fitted constants")


if __name__ == "__main__":
    main()
