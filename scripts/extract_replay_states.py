"""Extract episode start states from the GC1 2s replays into data/replay_states.npz.

Moments are taken every 0.1 s, then dropped if they fall in a kickoff countdown, within PRE_GOAL seconds before a
goal or between a goal and the next kickoff, have a demoed car, or fail playstyle.states.valid_state. Each car is
also flagged airborne when it is not resting on a surface (no jump or flip left when the state is used).

Replay conventions (checked against the data by fitting): car and ball rotation is R = Rx(rot x) Ry(rot y) Rz(rot z),
giving columns forward, right, up as in RocketSim; angular velocity is in 1/100 rad/s; boost in 0..255.

    python scripts/extract_replay_states.py
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playstyle.states import BALL_RADIUS, FIELD_Y, resting, valid_state

FPS = 10
PRE_GOAL = 3.0  # s before a goal that are skipped, so episodes do not mostly start mid-shot
N_PLAYER_COLS = 14


def rotations(euler):
    """(n, 3) replay rotations (x, y, z) -> (n, 3, 3) matrices R = Rx Ry Rz."""
    a, b, c = euler[:, 0], euler[:, 1], euler[:, 2]
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    o, z = np.ones_like(a), np.zeros_like(a)
    rx = np.stack([np.stack([o, z, z], -1), np.stack([z, ca, -sa], -1), np.stack([z, sa, ca], -1)], -2)
    ry = np.stack([np.stack([cb, z, sb], -1), np.stack([z, o, z], -1), np.stack([-sb, z, cb], -1)], -2)
    rz = np.stack([np.stack([cc, -sc, z], -1), np.stack([sc, cc, z], -1), np.stack([z, z, o], -1)], -2)
    return rx @ ry @ rz


def usable_frames(time, ball_pos, ball_vel, cars_pos, pre_goal=PRE_GOAL):
    """Mask of moments outside kickoff countdowns, the pre_goal seconds before goals and goal-to-kickoff gaps,
    with no demoed car. pre_goal=0 gives all live play."""
    kickoff = (np.linalg.norm(ball_pos[:, :2], axis=1) < 10) & (np.linalg.norm(ball_vel, axis=1) < 1)
    in_goal = np.abs(ball_pos[:, 1]) > FIELD_Y + BALL_RADIUS
    keep = np.zeros(len(time), dtype=bool)
    live, start = False, None
    for i in range(len(time)):
        if kickoff[i]:
            live, start = True, i
            continue
        if live and in_goal[i]:  # goal: drop the run-up, then wait for the next kickoff
            keep[(time >= time[i] - pre_goal) & (time <= time[i])] = False
            live = False
            continue
        keep[i] = live
    demoed = np.all(cars_pos == 0, axis=2).any(axis=1)
    return keep & ~kickoff & ~demoed


def extract(path):
    import subtr_actor as sa

    meta, arr = sa.get_ndarray_with_info_from_replay_filepath(
        path, global_feature_adders=["BallRigidBody", "CurrentTime"], fps=FPS)
    rm = meta["replay_meta"]
    if len(rm["team_zero"]) != 2 or len(rm["team_one"]) != 2:
        return None
    time = arr[:, 12]
    cars = arr[:, 13:].reshape(len(arr), 4, N_PLAYER_COLS)
    ball_pos, ball_vel, ball_spin = arr[:, 0:3], arr[:, 6:9], arr[:, 9:12] / 100
    keep = usable_frames(time, ball_pos, ball_vel, cars[:, :, 0:3])
    kick = (np.linalg.norm(ball_pos[:, :2], axis=1) < 10) & (np.linalg.norm(ball_vel, axis=1) < 1)
    if kick.any() and cars[kick][:, :2, 1].mean() > 0:  # team zero must be blue (defends -y)
        raise ValueError("team zero starts at +y")
    rot = rotations(cars[:, :, 3:6].reshape(-1, 3)).reshape(len(arr), 4, 3, 3)
    out_ball, out_cars, out_air = [], [], []
    for i in np.flatnonzero(keep):
        pos, up = cars[i, :, 0:3].astype(np.float64), rot[i, :, :, 2]
        if not valid_state(ball_pos[i].astype(np.float64), pos, rot[i]):
            continue
        out_ball.append(np.concatenate([ball_pos[i], ball_vel[i], ball_spin[i]]))
        out_cars.append(np.concatenate([pos, rot[i].reshape(4, 9), cars[i, :, 6:9], cars[i, :, 9:12] / 100,
                                        cars[i, :, 12:13] * 100 / 255], axis=1))
        out_air.append([not resting(pos[k], up[k]) for k in range(4)])
    return np.array(out_ball), np.array(out_cars), np.array(out_air), int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default="data/replays/*.replay")
    ap.add_argument("--out", default="data/replay_states.npz")
    args = ap.parse_args()
    balls, cars, air, game = [], [], [], []
    for g, f in enumerate(sorted(glob.glob(args.replays))):
        try:
            result = extract(f)
        except Exception as e:  # a few replays fail to parse
            print(f"skip {Path(f).name}: {e}")
            continue
        if result is None or len(result[0]) == 0:
            print(f"skip {Path(f).name}: not a 2v2 game or no usable moments")
            continue
        b, c, a, usable = result
        print(f"{Path(f).name}: {len(b)} states ({usable - len(b)} of {usable} usable moments failed validation)")
        balls.append(b), cars.append(c), air.append(a), game.append(np.full(len(b), g))
    ball, car, airborne, games = map(np.concatenate, (balls, cars, air, game))
    np.savez_compressed(args.out, ball=ball.astype(np.float32), cars=car.astype(np.float32), airborne=airborne,
                        game=games)
    print(f"{len(ball)} states from {len(balls)} games -> {args.out}; airborne cars {airborne.mean():.1%}; "
          f"max car speed {np.linalg.norm(car[:, :, 12:15], axis=2).max():.0f}, max ball speed "
          f"{np.linalg.norm(ball[:, 3:6], axis=1).max():.0f}")


if __name__ == "__main__":
    main()
