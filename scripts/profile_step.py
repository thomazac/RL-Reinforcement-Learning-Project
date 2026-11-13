"""Time each part of an env step (no rendering). Counts are per env step = 4 agent-steps in 2v2."""

import argparse
import time
from collections import defaultdict

import numpy as np

import playstyle.ttb as ttb_module
import playstyle.ttb_v1 as ttb_v1
from playstyle.env import make_env

T = defaultdict(float)


def timed(obj, name, key):
    orig = getattr(obj, name)

    def wrapper(*a, **kw):
        t = time.perf_counter()
        out = orig(*a, **kw)
        T[key] += time.perf_counter() - t
        return out

    setattr(obj, name, wrapper)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--reward-ttb", default="v1", choices=("v1", "live"))
    ap.add_argument("--no-coordination", action="store_true", help="Phase 1a setup: no roles or coordination")
    args = ap.parse_args()

    env = make_env(reward_ttb=args.reward_ttb, coordination=not args.no_coordination)
    provider = env.shared_info_provider
    timed(env.transition_engine, "step", "physics (RocketSim)")
    timed(provider.obs_predictor, "predict", "ball prediction (v1)")
    timed(ttb_v1, "estimate_ttb", "TTB (v1)")
    if provider.predictor is not None:
        timed(provider.predictor, "predict", "ball prediction (live)")
        timed(ttb_module, "estimate_ttb", "TTB (live)")
    timed(provider.roles, "update", "roles")
    timed(provider.coordination, "update", "coordination")
    timed(provider, "step", "shared-info provider total")
    timed(env.obs_builder, "build_obs", "observations")
    timed(env.reward_fn, "get_rewards", "reward")
    timed(env.action_parser, "parse_actions", "action parsing")

    rng = np.random.default_rng(0)
    env.reset()
    T.clear()
    start = time.perf_counter()
    for _ in range(args.steps):
        _, _, term, trunc = env.step({a: np.array([rng.integers(90)]) for a in env.agents})
        if any(term.values()) or any(trunc.values()):
            env.reset()
    total = time.perf_counter() - start

    print(f"{args.steps} env steps, {args.steps / total:.0f} env steps/s = "
          f"{4 * args.steps / total:.0f} agent-steps/s (one process, random actions, no policy; "
          f"reward_ttb={args.reward_ttb}, coordination={not args.no_coordination})")
    for key, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print(f"  {key:28s} {v / args.steps * 1e6:7.0f} us/step  {v / total * 100:5.1f}%")


if __name__ == "__main__":
    main()
