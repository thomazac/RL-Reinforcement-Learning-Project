"""Time RLGym-Learn's update step (value predictions, returns, PPO) for one configuration, without its env
processes: real env steps are collected once in this process and handed to the PPO agent controller the way
the env processes would hand them over.

    python scripts/learner_cost.py --width 128 --precision float16 --epochs 2 --buffer 300000

This measures the learner only. Collection speed under RLGym-Learn needs its env processes (scripts/throughput.py).
"""

import argparse
import json
import pickle
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rlgym_learn.api import DerivedAgentControllerConfig
from rlgym_learn_algos.ppo import Trajectory

from playstyle.env import make_env
from playstyle.network import N_ACTIONS
from playstyle.obs import OBS_DIM
from playstyle.training import CONTROLLER, make_agent_controller, make_config

SCRATCH = ROOT / "runs" / "learner_cost"


def collect(n_steps, chunk, seed=0):
    """n_steps agent-steps of the Phase 1a env under random actions, as per-agent pieces of at most `chunk`
    steps (an iteration under RLGym-Learn cuts every env's episode off where the iteration ends)."""
    env = make_env(coordination=False, seed=seed)
    rng = np.random.default_rng(seed)
    pieces, open_pieces, total = [], {}, 0

    def close(agent, final_obs, truncated):
        p = open_pieces.pop(agent)
        pieces.append({"agent": agent, "obs": np.array(p["obs"]), "actions": np.array(p["actions"]),
                       "rewards": p["rewards"], "final_obs": final_obs, "truncated": truncated})

    obs = env.reset()
    while total < n_steps:
        actions = {a: np.array([rng.integers(N_ACTIONS)]) for a in env.agents}
        next_obs, rewards, terminated, truncated = env.step(actions)
        for a in env.agents:
            p = open_pieces.setdefault(a, {"obs": [], "actions": [], "rewards": []})
            p["obs"].append(obs[a])
            p["actions"].append(actions[a])
            p["rewards"].append(float(rewards[a]))
            total += 1
            if terminated[a] or truncated[a]:
                close(a, next_obs[a], bool(truncated[a]))
            elif len(p["obs"]) >= chunk:
                close(a, next_obs[a], True)
        if total >= n_steps:
            for a in list(open_pieces):
                close(a, next_obs[a], True)
        obs = env.reset() if any(terminated.values()) or any(truncated.values()) else next_obs
    return pieces


def load_steps(n_steps, chunk):
    path = ROOT / "runs" / f"learner_cost_steps_{n_steps}_{chunk}.pkl"
    if not path.exists():
        print(f"collecting {n_steps:,} agent-steps once (cached in {path.name})...", flush=True)
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(pickle.dumps(collect(n_steps, chunk)))
    return pickle.loads(path.read_bytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256, choices=(128, 256))
    ap.add_argument("--precision", default="float16", choices=("float16", "float32"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=100_000, help="agent-steps per iteration")
    ap.add_argument("--buffer", type=int, default=300_000, help="experience buffer size")
    ap.add_argument("--chunk", type=int, default=1500, help="longest piece of an episode within an iteration")
    ap.add_argument("--iterations", type=int, default=3, help="iterations timed once the buffer is full")
    ap.add_argument("--json", help="append the result to this file as one JSON line")
    args = ap.parse_args()

    pieces = load_steps(args.steps, args.chunk)
    assert sum(len(p["obs"]) for p in pieces) == args.steps
    controller = make_agent_controller(width=args.width, precision=args.precision, overlap=False)
    config = make_config(timesteps_per_iteration=args.steps, buffer_size=args.buffer, batch_size=args.steps,
                         n_minibatches=max(1, args.steps // 50_000), n_epochs=args.epochs, save_folder=str(SCRATCH))
    controller.set_space_types(("real", OBS_DIM), ("discrete", N_ACTIONS))
    controller.load(DerivedAgentControllerConfig(
        CONTROLLER, controller.validate_config(config.agent_controllers_config[CONTROLLER]), config.base_config,
        config.process_config, str(SCRATCH)))
    actor = controller.learner.actor

    spent = defaultdict(float)
    ppo_data = []

    def timed(obj, name, key):
        original = getattr(obj, name)

        def wrapper(*a, **kw):
            t = time.perf_counter()
            out = original(*a, **kw)
            spent[key] += time.perf_counter() - t
            if key == "ppo":
                ppo_data.append(out)
            return out

        setattr(obj, name, wrapper)

    timed(controller, "_update_value_predictions", "values")
    timed(controller.experience_buffer, "submit_experience", "returns")
    timed(controller.learner, "learn", "ppo")

    def trajectories():
        """Fresh Trajectory objects, with the log-probability the current actor (float32) gives each action."""
        out = []
        with torch.no_grad():
            for p in pieces:
                log_probs = torch.log_softmax(actor.logits(p["obs"]), dim=-1)
                taken = log_probs.gather(-1, torch.as_tensor(p["actions"], device=log_probs.device)).view(-1).cpu()
                out.append(Trajectory(p["agent"], list(p["obs"]), list(p["actions"]), taken, list(p["rewards"]),
                                      None, p["final_obs"], torch.tensor(0, dtype=torch.float32), p["truncated"]))
        return out

    warmup = max(1, args.buffer // args.steps)  # until the buffer is full; the first iteration also warms the GPU up
    rows = []
    for i in range(warmup + args.iterations):
        controller.iteration_trajectories = trajectories()
        controller.iteration_timesteps = args.steps
        controller.timestep_collection_end_time = time.perf_counter()
        spent.clear()
        if i == warmup:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        controller._learn()
        rows.append({"total": time.perf_counter() - start, **spent})
    shutil.rmtree(SCRATCH, ignore_errors=True)

    kept = rows[warmup:]
    mean = {k: sum(r[k] for r in kept) / len(kept) for k in ("total", "values", "returns", "ppo")}
    passes = ppo_data[-1].cumulative_model_updates - ppo_data[-2].cumulative_model_updates
    result = {"width": args.width, "precision": args.precision, "epochs": args.epochs, "buffer": args.buffer,
              "steps": args.steps, "passes_per_step": passes, "update_s": mean["total"], "ppo_s": mean["ppo"],
              "values_s": mean["values"], "returns_s": mean["returns"],
              "us_per_step": mean["total"] / args.steps * 1e6,
              "first_kl": ppo_data[0].kl_divergence, "first_clip_fraction": ppo_data[0].sb3_clip_fraction,
              "gpu_gb": torch.cuda.max_memory_allocated() / 2**30, "pieces": len(pieces)}
    print(f"width {args.width}, {args.precision}, {args.epochs} epoch(s), buffer {args.buffer:,}: "
          f"update {mean['total']:.2f} s per {args.steps:,} steps = {result['us_per_step']:.1f} us per agent-step "
          f"(PPO {mean['ppo']:.2f} s for {passes} passes, value predictions {mean['values']:.2f} s, "
          f"returns and buffer {mean['returns']:.2f} s), peak GPU {result['gpu_gb']:.1f} GB, "
          f"first update KL {result['first_kl']:.1e}, clip fraction {result['first_clip_fraction']:.4f}", flush=True)
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
