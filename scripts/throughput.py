"""Throughput of one training configuration under RLGym-Learn: how many agent-steps per second are collected,
how long the update takes, and what that leaves overall. Each run is a fresh short training run; nothing is kept.

    python scripts/throughput.py --width 128 --precision float16 --epochs 2 --buffer 100000

Collection and update are timed per iteration. Iterations before the experience buffer is full (and the first
one, which includes startup) are left out of the averages.

NOT YET RUN TO COMPLETION. On Windows, RLGym-Learn 2.0.0 fails while starting its env processes
("FileExistsError: entity already exists" from init_processes), before any step is collected, so nothing below
the LearningCoordinator(...) line has been exercised. scripts/learner_cost.py measures the update without the
env processes.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"  # stops numpy in the env processes from competing for cores

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCRATCH = Path(__file__).resolve().parents[1] / "runs" / "throughput"


def main():
    import torch
    from rlgym_learn import LearningCoordinator
    from rlgym_learn_algos.ppo import PPOMetricsLogger

    from playstyle.env import build_env
    from playstyle.training import make_agent_controller, make_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128, choices=(128, 256))
    ap.add_argument("--precision", default="float16", choices=("float16", "float32"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100_000, help="agent-steps collected per iteration")
    ap.add_argument("--buffer", type=int, default=100_000, help="experience buffer size (design doc: 300,000)")
    ap.add_argument("--iterations", type=int, default=5, help="iterations measured after the buffer is full")
    ap.add_argument("--json", help="append the result to this file as one JSON line")
    args = ap.parse_args()

    warmup = 1 + max(0, args.buffer // args.steps - 1)  # startup iteration, then until the buffer is full
    n_iterations = warmup + args.iterations
    spent = defaultdict(float)  # seconds per part of the main loop, for the iteration in progress
    rows = []
    after_update = []  # agent-steps that arrived in the first collection after each update

    class Logger(PPOMetricsLogger):
        def collect_agent_metrics(self, data):
            rows.append({"steps": data.timesteps_collected, "collect": data.timestep_collection_time,
                         "total": data.iteration_time, "updates": data.ppo_data.cumulative_model_updates,
                         "kl": data.ppo_data.kl_divergence, "clip": data.ppo_data.sb3_clip_fraction, **spent})
            spent.clear()
            state["just_updated"] = True

        def report_metrics(self):
            r = rows[-1]
            print(f"iteration {len(rows)}: {r['steps']} steps, collect {r['collect']:.2f} s, "
                  f"update {r['total'] - r['collect']:.2f} s", flush=True)

    def timed(obj, name, key):
        original = getattr(obj, name)

        def wrapper(*a, **kw):
            t = time.perf_counter()
            out = original(*a, **kw)
            spent[key] += time.perf_counter() - t
            return out

        setattr(obj, name, wrapper)

    state = {"just_updated": False}
    controller = make_agent_controller(width=args.width, precision=args.precision, metrics_logger=Logger())
    config = make_config(n_proc=args.procs, timesteps_per_iteration=args.steps, buffer_size=args.buffer,
                         batch_size=args.steps, n_minibatches=max(1, args.steps // 50_000), n_epochs=args.epochs,
                         timestep_limit=n_iterations * args.steps + args.steps // 2,
                         save_folder=str(SCRATCH), save_every_ts=10**12)
    coordinator = LearningCoordinator(build_env, agent_controller=controller, config=config)

    timed(controller, "_update_value_predictions", "values")
    timed(controller.experience_buffer, "submit_experience", "gae")
    timed(controller.learner, "learn", "ppo")
    timed(controller, "get_actions", "inference")
    timed(coordinator.env_process_interface, "send_env_actions", "send")
    collect = coordinator.env_process_interface.collect_env_responses

    def collect_and_count(*a, **kw):
        t = time.perf_counter()
        out = collect(*a, **kw)
        spent["wait"] += time.perf_counter() - t
        if state["just_updated"]:
            after_update.append(out[0])
            state["just_updated"] = False
        return out

    coordinator.env_process_interface.collect_env_responses = collect_and_count
    torch.cuda.reset_peak_memory_stats()
    coordinator.start()
    shutil.rmtree(SCRATCH, ignore_errors=True)

    kept = rows[warmup:]
    if len(kept) < args.iterations:
        raise SystemExit(f"only {len(rows)} iterations finished; expected {n_iterations}")
    n = len(kept)
    steps = sum(r["steps"] for r in kept)
    collect_s = sum(r["collect"] for r in kept)
    total_s = sum(r["total"] for r in kept)
    part = {k: sum(r.get(k, 0.0) for r in kept) for k in ("values", "gae", "ppo", "inference", "send", "wait")}
    passes = kept[-1]["updates"] - kept[-2]["updates"]  # batches per iteration; a batch is one iteration's steps
    result = {
        "width": args.width, "precision": args.precision, "epochs": args.epochs, "procs": args.procs,
        "buffer": args.buffer, "steps_per_iteration": steps / n, "passes_per_step": passes,
        "collect_sps": steps / collect_s, "overall_sps": steps / total_s,
        "collect_s": collect_s / n, "update_s": (total_s - collect_s) / n,
        "ppo_s": part["ppo"] / n, "values_s": part["values"] / n, "gae_s": part["gae"] / n,
        "inference_share": part["inference"] / collect_s, "wait_share": part["wait"] / collect_s,
        "send_share": part["send"] / collect_s,
        "steps_in_first_collection_after_update": max(after_update) if after_update else None,
        "kl": kept[-1]["kl"], "clip_fraction": kept[-1]["clip"],
        "gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    print()
    print(f"width {args.width}, {args.precision}, {args.epochs} epoch(s), buffer {args.buffer:,}, {args.procs} processes "
          f"({n} iterations of {steps / n:,.0f} steps; each step trained on {passes:.0f} times)")
    print(f"  collection      {result['collect_sps']:8,.0f} agent-steps/s   ({result['collect_s']:.2f} s per iteration)")
    print(f"    waiting on env processes {result['wait_share']:.0%}, actor inference {result['inference_share']:.0%}, "
          f"sending actions {result['send_share']:.0%} of collection time")
    print(f"  update          {result['update_s']:8.2f} s per iteration  "
          f"(PPO {result['ppo_s']:.2f}, value predictions {result['values_s']:.2f}, returns and buffer {result['gae_s']:.2f})")
    print(f"  overall         {result['overall_sps']:8,.0f} agent-steps/s")
    print(f"  most steps arriving in the first collection after an update: "
          f"{result['steps_in_first_collection_after_update']} (4 per waiting process = {4 * args.procs})")
    print(f"  peak GPU memory {result['gpu_gb']:.1f} GB, last KL {result['kl']:.4f}, clip fraction {result['clip_fraction']:.3f}")
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
