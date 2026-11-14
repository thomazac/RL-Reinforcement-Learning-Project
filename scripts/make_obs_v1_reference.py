"""Record full game states with the observation builder's and profile tracker's outputs, for
tests/test_obs_v1_reference.py. Run once, before any rewrite of those modules; the file is then permanent.

800 consecutive steps (~53 s, more than a 25 s averaging window): 400 scripted (demo variant vs normal, for
demo events) and 400 random actions (jumps, aerials). Episode starts are flagged so the profile replay
resets where the original did. A separate block of hand-built states (scripts/obs_hand_states.py) covers
what play doesn't reach: aerials, the ceiling, goals, corners, top speeds, overtime, multiple demos.
"""

import argparse
import collections
import copy
import lzma
import pickle
import random
from pathlib import Path

import numpy as np

from playstyle.env import make_env
from playstyle.obs import OBS_LAYOUT
from playstyle.scripted import RoleBot
from playstyle.versions import OBS_VERSIONS
from scripts.obs_hand_states import hand_rows

OUT = Path("tests/fixtures/obs_v1_reference.pkl.xz")
INPUT_KEYS = ("obs_ttb", "obs_ball_slices", "profile", "prev_action", "match", "kickoff")


def run(policy, steps, seed, around_demo=False, max_steps=12000):
    """Record `steps` consecutive steps. With around_demo, keep a rolling window and stop once the first demo
    sits in its middle (single pass: long seeded runs aren't reproducible step for step)."""
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env, bot = make_env(seed=seed), RoleBot(seed=seed, team_modes={0: "demo", 1: "normal"})
    env.reset()
    rows, start, demo_at = collections.deque(maxlen=steps), True, None
    for i in range(max_steps):
        rows.append(dict(
            state=copy.deepcopy(env.state), start=start,
            info=copy.deepcopy({k: env.shared_info[k] for k in INPUT_KEYS}),
            profile_snapshot=env.shared_info_provider.profile.snapshot(),
            obs=env.obs_builder.build_obs(env.agents, env.state, env.shared_info),
        ))
        if demo_at is None and any(c.is_demoed for c in env.state.cars.values()):
            demo_at = i
        if (not around_demo and len(rows) == steps) or (demo_at is not None and i - demo_at >= steps // 2):
            return list(rows)
        actions = bot.act(env.state, env.shared_info) if policy == "scripted" else \
            {a: rng.integers(0, 90, size=1) for a in env.agents}
        _, _, term, trunc = env.step(actions)
        start = any(term.values()) or any(trunc.values())
        if start:
            env.reset()
    raise SystemExit(f"no demo in {max_steps} steps of scripted demo play; try another seed")


def record(steps_per_policy=400, seed=0):
    return run("scripted", steps_per_policy, seed, around_demo=True) + run("random", steps_per_policy, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite; only for a new version, never for v1")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} exists; the observation v1 reference is permanent")
    rows, hand = record(), hand_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(OUT, "wb") as f:
        pickle.dump({"versions": OBS_VERSIONS, "layout": OBS_LAYOUT, "rows": rows, "hand_rows": hand}, f)
    print(f"{len(hand)} hand-built states: " + ", ".join(
        f"{n} {sum(r['name'] == n for r in hand)}" for n in dict.fromkeys(r["name"] for r in hand)))
    demos = sum(c.is_demoed for r in rows for c in r["state"].cars.values())
    touches = sum(c.ball_touches > 0 for r in rows for c in r["state"].cars.values())
    goals = sum(bool(r["info"]["match"]["goal"]) for r in rows)
    print(f"{len(rows)} steps, {sum(r['start'] for r in rows)} episode starts, {goals} goals, "
          f"{touches} car-steps touching, {demos} demoed car-steps, {OUT.stat().st_size / 1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
