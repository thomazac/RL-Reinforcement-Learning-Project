"""Record raw game states and frozen TTB v1 outputs for tests/test_ttb_v1.py. Run once; the file is then permanent.

Consecutive steps are stored so the test also exercises the prediction cache. Half the steps use scripted
bots (ground play, rotations), half random actions (jumps, aerial and flipping cars).
"""

import argparse
import random
from pathlib import Path

import numpy as np

from playstyle import ttb_v1
from playstyle.env import make_env
from playstyle.scripted import RoleBot

OUT = Path("tests/fixtures/ttb_v1_reference.npz")


def record(steps_per_policy=400, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    rows, predictor = [], ttb_v1.BallPredictor()
    for policy in ("scripted", "random"):
        env, bot = make_env(), RoleBot(seed=seed)
        env.reset()
        for _ in range(steps_per_policy):
            s = env.state
            cars = [s.cars[a] for a in sorted(s.cars)]
            ttb, slices = ttb_v1.ttb_from_state(s, predictor)
            rows.append(dict(
                tick=s.tick_count,
                ball=np.stack([s.ball.position, s.ball.linear_velocity, s.ball.angular_velocity]),
                pos=[c.physics.position for c in cars], vel=[c.physics.linear_velocity for c in cars],
                fwd=[c.physics.forward for c in cars], boost=[c.boost_amount for c in cars],
                demoed=[c.is_demoed for c in cars], on_ground=[c.on_ground for c in cars],
                ttb=[ttb[a] for a in sorted(s.cars)], slices=slices,
            ))
            actions = bot.act(s, env.shared_info) if policy == "scripted" else \
                {a: rng.integers(0, 90, size=1) for a in env.agents}
            _, _, term, trunc = env.step(actions)
            if any(term.values()) or any(trunc.values()):
                env.reset()
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite; only for a new version, never for v1")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} exists; the TTB v1 reference is permanent")
    data = record()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, version=ttb_v1.VERSION, **data)
    finite = np.isfinite(data["ttb"])
    print(f"{len(data['tick'])} steps, {finite.mean() * 100:.0f}% finite TTB, "
          f"{(~data['on_ground']).mean() * 100:.0f}% car-steps airborne, {data['demoed'].sum()} demoed car-steps -> {OUT}")


if __name__ == "__main__":
    main()
