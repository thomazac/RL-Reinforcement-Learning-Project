"""Throughput of one training configuration under RLGym-Learn: how many agent-steps per second are collected,
how long the update takes, and what that leaves overall. Each run is a fresh short training run; nothing is kept.

    python scripts/throughput.py --width 128 --procs 16

Collection and update are timed per iteration. Iterations before the experience buffer is full (and the first
one, which includes startup) are left out of the averages.

It also checks that each trajectory's final observation (what a truncated trajectory bootstraps from) is the
step after its last one, not the first observation after a reset: within one 8-tick step the observer's car
and the ball move less than 0.1 on the observation's 1/5120 position scale; a reset teleports the car.
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"  # one math thread per process; env processes inherit this before importing numpy

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCRATCH = ROOT / "runs" / "throughput"
SELF_POS, BALL_POS = slice(0, 3), slice(124, 127)
JUMP_LIMIT = 0.15


def final_obs_jumps(trajectories):
    """Largest position change between each trajectory's last observation and its final observation."""
    return np.array([max(np.abs(t.final_obs[s] - t.obs_list[-1][s]).max() for s in (SELF_POS, BALL_POS))
                     for t in trajectories])


def main():
    import torch
    from rlgym_learn import LearningCoordinator

    from playstyle.env import build_env
    from playstyle.training import CONTROLLER, RunLogger, make_agent_controller, make_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256, choices=(128, 256))
    ap.add_argument("--precision", default="float16", choices=("float16", "float32"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=100_000, help="agent-steps collected per iteration")
    ap.add_argument("--buffer", type=int, default=300_000, help="experience buffer size")
    ap.add_argument("--iterations", type=int, default=3, help="iterations measured after the buffer is full")
    ap.add_argument("--inference-procs", type=int, help="processes per actor call (default: half)")
    ap.add_argument("--json", help="append the result to this file as one JSON line")
    args = ap.parse_args()

    warmup = 1 + max(0, args.buffer // args.steps - 1)  # startup iteration, then until the buffer is full
    n_iterations = warmup + args.iterations
    spent = defaultdict(float)  # seconds per part of the main loop, for the iteration in progress
    rows, jumps, ratios = [], [], []

    monitors = []

    class Logger(RunLogger):  # the run's monitors, kept quiet: only warnings print
        def report_metrics(self):
            monitors.append(self.agent_metrics["Monitors"])
            for w in self.warnings:
                print(f"WARNING: {w}", flush=True)

    controller = make_agent_controller(width=args.width, precision=args.precision, metrics_logger=Logger())
    config = make_config(n_proc=args.procs, timesteps_per_iteration=args.steps, buffer_size=args.buffer,
                         batch_size=args.steps, n_minibatches=max(1, args.steps // 50_000), n_epochs=args.epochs,
                         timestep_limit=n_iterations * args.steps + args.steps // 2,
                         save_folder=str(SCRATCH), save_every_ts=10**12, inference_procs=args.inference_procs)
    coordinator = LearningCoordinator(build_env, {CONTROLLER: controller}, config)

    def timed(obj, name, key, after=None):
        original = getattr(obj, name)

        def wrapper(*a, **kw):
            if after is None:
                t = time.perf_counter()
                out = original(*a, **kw)
                spent[key] += time.perf_counter() - t
                return out
            return after(original, *a, **kw)

        setattr(obj, name, wrapper)

    def learn(original):
        collect = controller.timestep_collection_end_time - controller.timestep_collection_start_time
        steps = controller.iteration_timesteps
        t = time.perf_counter()
        original()
        rows.append({"steps": steps, "collect": collect, "update": time.perf_counter() - t, **spent})
        spent.clear()
        print(f"iteration {len(rows)}: {steps} steps, collect {collect:.2f} s, update {rows[-1]['update']:.2f} s",
              flush=True)

    def value_predictions(original):
        jumps.append(final_obs_jumps(controller.iteration_trajectories))
        t = time.perf_counter()
        original()
        spent["values"] += time.perf_counter() - t

    def ppo(original, exp):
        t = time.perf_counter()
        out = original(exp)
        spent["ppo"] += time.perf_counter() - t
        ratios.append(out.first_minibatch_ratio)
        return out

    timed(controller, "_learn", None, after=learn)
    timed(controller, "_update_value_predictions", None, after=value_predictions)
    timed(controller.learner, "learn", None, after=ppo)
    timed(controller.experience_buffer, "submit_experience", "returns")
    timed(controller, "get_actions", "inference")
    timed(coordinator.env_process_interface, "collect_step_data", "wait")
    timed(coordinator.env_process_interface, "send_env_actions", "send")
    interface = coordinator.env_process_interface
    cleanup, exit_codes = interface.cleanup, []

    def record_exits_then_cleanup():
        exit_codes.extend(p.exitcode for p, *_ in interface.processes if p is not None)
        cleanup()

    interface.cleanup = record_exits_then_cleanup
    torch.cuda.reset_peak_memory_stats()
    coordinator.start()
    shutil.rmtree(SCRATCH, ignore_errors=True)

    kept = rows[warmup:]
    if len(kept) < args.iterations:
        raise SystemExit(f"only {len(rows)} iterations finished; expected {n_iterations}")
    n = len(kept)
    steps = sum(r["steps"] for r in kept)
    collect_s = sum(r["collect"] for r in kept)
    update_s = sum(r["update"] for r in kept)
    part = {k: sum(r.get(k, 0.0) for r in kept) for k in ("values", "returns", "ppo", "inference", "send", "wait")}
    all_jumps = np.concatenate(jumps)
    result = {
        "width": args.width, "precision": args.precision, "epochs": args.epochs, "procs": args.procs,
        "inference_procs": coordinator.config.process_config.min_process_steps_per_inference,
        "buffer": args.buffer, "steps_per_iteration": steps / n,
        "collect_sps": steps / collect_s, "collect_sps_per_proc": steps / collect_s / args.procs,
        "overall_sps": steps / (collect_s + update_s),
        "collect_s": collect_s / n, "update_s": update_s / n,
        "ppo_s": part["ppo"] / n, "values_s": part["values"] / n, "returns_s": part["returns"] / n,
        "inference_share": part["inference"] / collect_s, "wait_share": part["wait"] / collect_s,
        "send_share": part["send"] / collect_s,
        "first_minibatch_ratio": ratios, "trajectories": len(all_jumps),
        "final_obs_jumps_over_limit": int((all_jumps > JUMP_LIMIT).sum()), "largest_final_obs_jump": float(all_jumps.max()),
        "gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    print()
    print(f"width {args.width}, {args.precision}, {args.epochs} epoch(s), buffer {args.buffer:,}, {args.procs} processes, "
          f"actor runs once {result['inference_procs']} have stepped ({n} iterations of {steps / n:,.0f} steps)")
    print(f"  collection      {result['collect_sps']:8,.0f} agent-steps/s = {result['collect_sps_per_proc']:,.0f} per process "
          f"({result['collect_s']:.2f} s per iteration)")
    print(f"    in the main loop: waiting on env processes {result['wait_share']:.0%}, actor inference "
          f"{result['inference_share']:.0%}, sending actions {result['send_share']:.0%} of collection time")
    print(f"  update          {result['update_s']:8.2f} s per iteration  "
          f"(PPO {result['ppo_s']:.2f}, value predictions {result['values_s']:.2f}, returns and buffer {result['returns_s']:.2f})")
    print(f"  overall         {result['overall_sps']:8,.0f} agent-steps/s")
    print(f"  first-minibatch mean ratio per update: {', '.join(f'{r:.6f}' for r in ratios)}")
    print(f"  final observations: {result['trajectories']} trajectories, {result['final_obs_jumps_over_limit']} jump more "
          f"than {JUMP_LIMIT} (largest {result['largest_final_obs_jump']:.3f})")
    print(f"  peak GPU memory {result['gpu_gb']:.1f} GB")
    worst = {k: max(m[k] for m in monitors) for k in monitors[0]}
    print("  monitors, worst over all iterations: " + ", ".join(f"{k} {v:.4g}" for k, v in worst.items()))
    result["monitors_worst"] = worst
    print(f"  env processes at the end: {sum(c is None for c in exit_codes)} running, exited: "
          f"{[c for c in exit_codes if c is not None]}")
    result["env_process_exit_codes"] = exit_codes
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
