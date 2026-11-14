"""Do two runs with the same seed stay identical? If not, where do they first differ, and at what kind of step?

Each run is a fresh process with the same seed for the environment and the policy; the global random streams
are left unseeded on purpose, so anything that still draws from them shows up. Every step the full game state
(cars, ball, pads, clock, score) goes into one float32 row; rows are compared bit for bit. The first differing
step is reported with the events at and just before it:
  reset      a new episode (environment reset)
  kickoff    a mid-match kickoff after a goal
  demo       a car was demoed on this step
  respawn    a demoed car came back on this step
A first difference at a reset, kickoff or respawn points at an unseeded random number generator or state that
survives set_state. One in open play points at the physics engine itself; scripts/rocketsim_determinism.py
checks that directly, without RLGym or any reset.

    python -m scripts.seed_divergence --policy scripted --steps 6000

Result (RocketSim 2.2.1, RLGym 2.0.1; 16 pairs of 6000 steps): 4 pairs identical throughout, 12 diverged,
after a median of 98 s of play. None first differed at a reset or kickoff; runs pass through up to 14 resets
and 8 kickoffs bit-identical.
  8 in open play   a car off the ground in contact with the floor, a wall or another car; the difference
                   starts under 0.02 uu and grows. Bare RocketSim does the same: the engine, not our code.
  4 at a respawn   the car comes back at a different one of the four respawn points. RocketSim draws them
                   from a generator its Python bindings give no seed for.
So a seed fixes the start of a match (spawns, score, clock) but not the match. Evaluations must treat
same-seed matches as independent samples, and anything worth rewatching has to be logged, not re-simulated.
"""

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CAR_FIELDS = ("pos", "vel", "ang_vel", "rot", "boost", "demo_timer", "on_ground", "has_jumped", "has_flipped",
              "has_double_jumped", "is_jumping", "jump_time", "flip_time", "air_time", "supersonic", "handbrake")
CAR_WIDTHS = (3, 3, 3, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
EVENTS = ("reset", "kickoff", "demo", "respawn")


def state_row(state, match):
    parts = []
    for a in sorted(state.cars):
        c = state.cars[a]
        p = c.physics
        parts += [p.position, p.linear_velocity, p.angular_velocity, p.rotation_mtx.ravel(),
                  [c.boost_amount, c.demo_respawn_timer, c.on_ground, c.has_jumped, c.has_flipped,
                   c.has_double_jumped, c.is_jumping, c.jump_time, c.flip_time, c.air_time_since_jump,
                   c.supersonic_time, c.handbrake]]
    b = state.ball
    parts += [b.position, b.linear_velocity, b.angular_velocity, b.rotation_mtx.ravel(), state.boost_pad_timers,
              [state.tick_count, match["blue"], match["orange"], match["time_remaining"], match["overtime"]]]
    return np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in parts])


def column_names(agents):
    names = [f"{a}.{f}" for a in agents for f, w in zip(CAR_FIELDS, CAR_WIDTHS) for _ in range(w)]
    names += [f"ball.{f}" for f, w in (("pos", 3), ("vel", 3), ("ang_vel", 3), ("rot", 9)) for _ in range(w)]
    return names + ["pads"] * 34 + ["tick", "score.blue", "score.orange", "clock", "overtime"]


def run(policy, steps, seed, out):
    from playstyle.env import make_env
    from playstyle.scripted import RoleBot

    rng = np.random.default_rng(seed)
    env = make_env(coordination=policy == "scripted", seed=seed)
    bot = RoleBot(seed=seed, team_modes={0: "demo", 1: "normal"})
    env.reset()
    rows, events = [state_row(env.state, env.shared_info["match"])], [[True, False, False, False]]
    demoed, goal = {a: False for a in env.agents}, False
    for _ in range(steps):
        actions = bot.act(env.state, env.shared_info) if policy == "scripted" else \
            {a: rng.integers(0, 90, size=1) for a in env.agents}
        _, _, term, trunc = env.step(actions)
        now = {a: c.is_demoed for a, c in env.state.cars.items()}
        ev = [False, goal, any(now[a] and not demoed[a] for a in now), any(demoed[a] and not now[a] for a in now)]
        demoed, goal = now, bool(env.shared_info["match"]["goal"])
        if any(term.values()) or any(trunc.values()):
            env.reset()
            ev[0], goal = True, False
            demoed = {a: c.is_demoed for a, c in env.state.cars.items()}
        rows.append(state_row(env.state, env.shared_info["match"]))
        events.append(ev)
    np.savez(out, rows=np.array(rows), events=np.array(events), agents=np.array(sorted(env.agents)))


def compare(a, b, label):
    rows_a, rows_b, events, agents = a["rows"], b["rows"], a["events"], list(a["agents"])
    same = (rows_a.view(np.uint32) == rows_b.view(np.uint32)).all(axis=1)
    counts = ", ".join(f"{int(events[:, i].sum())} {name}" for i, name in enumerate(EVENTS))
    digest = hashlib.sha1(rows_a.tobytes()).hexdigest()[:12]
    print(f"{label}: {len(rows_a) - 1} steps ({counts}), run A sha1 {digest}")
    if same.all():
        print("  identical in every step")
        return None
    first = int(np.argmin(same))
    here = [name for i, name in enumerate(EVENTS) if events[first, i]] or ["open play"]
    print(f"  first difference at step {first} ({first * 8 / 120:.1f} s): {', '.join(here)}")
    for i, name in enumerate(EVENTS):
        before = np.flatnonzero(events[:first + 1, i])
        if len(before):
            print(f"    last {name:8s} {first - before[-1]:5d} steps earlier")
    names = np.array(column_names(agents))
    diff = np.abs(rows_a[first].astype(np.float64) - rows_b[first])
    for field in dict.fromkeys(names[diff > 0]):
        cols = names == field
        print(f"    {field:22s} max difference {diff[cols].max():.6g}   A {np.round(rows_a[first][cols], 3).tolist()}"
              f"   B {np.round(rows_b[first][cols], 3).tolist()}")
    return first, here


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("scripted", "random", "both"), default="both")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--worker", nargs=4, metavar=("POLICY", "STEPS", "SEED", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        policy, steps, seed, out = args.worker
        return run(policy, int(steps), int(seed), out)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for policy in (("scripted", "random") if args.policy == "both" else (args.policy,)):
            for seed in args.seeds:
                outs = [str(Path(tmp) / f"{policy}-{seed}-{k}.npz") for k in "AB"]
                procs = [subprocess.Popen([sys.executable, "-m", "scripts.seed_divergence", "--worker", policy,
                                           str(args.steps), str(seed), out]) for out in outs]
                if any(p.wait() for p in procs):
                    raise SystemExit("a run failed")
                results.append(compare(np.load(outs[0]), np.load(outs[1]), f"{policy} policy, seed {seed}"))
    diverged = [r for r in results if r]
    print(f"\n{len(results)} pairs of {args.steps} steps: {len(results) - len(diverged)} identical throughout, "
          f"{len(diverged)} diverged")
    if diverged:
        steps = sorted(first for first, _ in diverged)
        kinds = [", ".join(here) for _, here in diverged]
        print(f"  first difference at steps {steps} (median {int(np.median(steps))}, "
              f"{np.median(steps) * 8 / 120:.0f} s of play)")
        print("  at: " + ", ".join(f"{kinds.count(k)} {k}" for k in dict.fromkeys(kinds)))


if __name__ == "__main__":
    main()
