"""Is RocketSim itself repeatable? No RLGym, no set_state after the start, identical controls every run.

  play      four cars under seeded random controls, several fresh processes, compared bit for bit
  lockstep  two arenas in one process fed the same controls
  respawn   a car is demolished repeatedly; where does it come back?

Result (RocketSim 2.2.1): runs are identical for thousands of steps, then split at a contact (a tumbling car
meeting the floor) into one of a few outcomes; some runs take the same branch, and two arenas in one process
can split from each other. Respawn points are drawn from a generator the bindings give no seed for, so they
differ between processes from the first demo. Neither can be fixed from our side.

    python -m scripts.rocketsim_determinism
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SPAWN = [(-2048, -2560, 0.25), (256, -3840, 0.5), (2048, 2560, -0.75), (-256, 3840, -0.5)]


def worker(mode, steps, out):
    import RocketSim as rs

    from playstyle.ball_prediction import _ensure_init
    _ensure_init()

    def make():
        arena = rs.Arena(rs.GameMode.SOCCAR)
        cars = [arena.add_car(rs.Team.BLUE if i < 2 else rs.Team.ORANGE, rs.CarConfig(0)) for i in range(4)]
        for c, (x, y, yaw) in zip(cars, SPAWN):
            s = c.get_state()
            s.pos, s.vel, s.ang_vel, s.boost = rs.Vec(x, y, 17), rs.Vec(0, 0, 0), rs.Vec(0, 0, 0), 33.3
            s.rot_mat = rs.Angle(yaw * np.pi, 0, 0).as_rot_mat()
            c.set_state(s)
        return arena, cars

    def snap(arena, cars):
        row = []
        for c in cars:
            s = c.get_state()
            row += [s.pos.x, s.pos.y, s.pos.z, s.vel.x, s.vel.y, s.vel.z, s.ang_vel.x, s.ang_vel.y, s.ang_vel.z,
                    s.boost]
        b = arena.ball.get_state()
        return row + [b.pos.x, b.pos.y, b.pos.z, b.vel.x, b.vel.y, b.vel.z]

    if mode == "respawn":
        arena, cars = make()
        rows = [[]]
        for _ in range(12):
            cars[0].demolish()
            arena.step(400)  # past the 3 s respawn timer
            rows[0].append(snap(arena, cars))
        return np.save(out, np.array(rows, dtype=np.float32))
    worlds = [make() for _ in range(2 if mode == "lockstep" else 1)]
    rng = np.random.default_rng(0)
    rows = [[] for _ in worlds]
    for _ in range(steps):
        controls = []
        for _ in range(4):
            u, k = rng.uniform(-1, 1, 5), rng.random(3)
            ctl = rs.CarControls()
            ctl.throttle, ctl.steer, ctl.pitch, ctl.yaw, ctl.roll = abs(float(u[0])), *(float(x) for x in u[1:])
            ctl.jump, ctl.boost, ctl.handbrake = bool(k[0] < 0.1), bool(k[1] < 0.5), bool(k[2] < 0.1)
            controls.append(ctl)
        for w, (arena, cars) in enumerate(worlds):
            for c, ctl in zip(cars, controls):
                c.set_controls(ctl)
            arena.step(8)
            rows[w].append(snap(arena, cars))
    np.save(out, np.array(rows, dtype=np.float32))


def first_difference(a, b):
    same = (a.view(np.uint32) == b.view(np.uint32)).all(axis=1)
    if same.all():
        return "identical"
    i = int(np.argmin(same))
    d = np.abs(a[i] - b[i])
    c = int(np.argmax(d))
    who = "ball" if c >= 40 else f"car {c // 10} at height {a[i][c // 10 * 10 + 2]:.0f}"
    return f"first difference at step {i}, largest {d.max():.3g} ({who})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--worker", nargs=2, metavar=("MODE", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(args.worker[0], args.steps, args.worker[1])
    with tempfile.TemporaryDirectory() as tmp:
        def launch(mode, k):
            out = str(Path(tmp) / f"{mode}{k}.npy")
            return out, subprocess.Popen([sys.executable, "-m", "scripts.rocketsim_determinism", "--steps",
                                          str(args.steps), "--worker", mode, out])
        jobs = {(m, k): launch(m, k) for m, n in (("play", args.runs), ("lockstep", 2), ("respawn", 2))
                for k in range(n)}
        if any(p.wait() for _, p in jobs.values()):
            raise SystemExit("a run failed")
        data = {key: np.load(out) for key, (out, _) in jobs.items()}
    for k in range(1, args.runs):
        print(f"play, separate processes, run 0 vs {k}: {first_difference(data['play', 0][0], data['play', k][0])}")
    for k in range(2):
        a, b = data["lockstep", k]
        print(f"lockstep, two arenas in process {k}: {first_difference(a, b)}")
    for k in range(2):
        print(f"respawn points, process {k}: {data['respawn', k][0][:, :2].round().astype(int).tolist()}")


if __name__ == "__main__":
    main()
