"""Train under RLGym-Learn. Defaults are Phase 1a: core reward only (playstyle/rewards.py PHASE_1A), no
coordination, team spirit 0, learning rate 2e-4, gamma 0.99, width 256, 32 env processes.

    python scripts/train.py --run phase1a-smoke --steps 50000000
    python scripts/train.py --run phase1a-smoke --steps 50000000 --stop-at 10000000     # stop early, checkpoint
    python scripts/train.py --run phase1a-smoke --steps 50000000 --resume latest        # carry on from there

Everything lands in runs/train/<run>/: config.json (or resume-<steps>.json) with every setting, versions and
the git commit; log.jsonl with one line of metrics per iteration; checkpoints/. Each checkpoint also gets
resume_state.json, a summary of the training state at save time (step count, optimizer moments, running
return statistics, gradient scale, experience buffer size). On --resume the freshly loaded state is compared
with it, and the run stops if anything did not carry over.
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"  # one math thread per process; env processes inherit this before importing numpy

import argparse
import functools
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESUME_STATE = "resume_state.json"


def git_state():
    run = functools.partial(subprocess.run, cwd=ROOT, capture_output=True, text=True)
    return {"commit": run(["git", "rev-parse", "HEAD"]).stdout.strip(),
            "dirty_files": run(["git", "status", "--porcelain"]).stdout.split("\n")[:-1]}


def checkpoints(run_dir):
    """Complete checkpoints, oldest first. resume_state.json is written last, after a successful save, so a folder
    without it is a save that failed partway (an error-save after a CUDA failure left one)."""
    folders = sorted(run_dir.glob("checkpoints/*/*/*/resume_state.json"), key=lambda p: int(p.parent.name))
    return [p.parent for p in folders]


def state_summary(controller):
    """What a resumed run must get back exactly."""
    import torch

    learner, buffer = controller.learner, controller.experience_buffer
    stats = buffer.trajectory_processor.return_stats

    def optimizer(opt):
        state = opt.state_dict()["state"]
        return {"steps": int(next(iter(state.values()))["step"]) if state else 0,
                "exp_avg_norm": float(torch.sqrt(sum(s["exp_avg"].double().pow(2).sum() for s in state.values()))),
                "exp_avg_sq_sum": float(sum(s["exp_avg_sq"].double().sum() for s in state.values()))}

    def norm(module):
        return float(torch.nn.utils.parameters_to_vector(module.parameters()).detach().double().norm())

    return {"cumulative_timesteps": controller.cumulative_timesteps, "iteration": controller.cur_iteration,
            "model_updates": learner.cumulative_model_updates,
            "return_stats": {"count": int(stats.count), "mean": float(stats.running_mean[0]),
                             "variance_sum": float(stats.running_variance[0])},
            "actor_weights_norm": norm(learner.actor), "critic_weights_norm": norm(learner.critic),
            "actor_optimizer": optimizer(learner.actor_optimizer),
            "critic_optimizer": optimizer(learner.critic_optimizer),
            "grad_scale": float(learner.scaler.get_scale()) if learner.scaler.is_enabled() else 1.0,
            "experience_buffer_size": int(buffer.values.shape[0])}


def compare(saved, loaded, prefix=""):
    """Differences between two state summaries (relative tolerance 1e-9 on floats)."""
    diffs = []
    for k, v in saved.items():
        w = loaded.get(k)
        if isinstance(v, dict):
            diffs += compare(v, w or {}, f"{prefix}{k}.")
        elif isinstance(v, float):
            if w is None or abs(v - w) > 1e-9 * max(1.0, abs(v)):
                diffs.append(f"{prefix}{k}: saved {v!r}, loaded {w!r}")
        elif v != w:
            diffs.append(f"{prefix}{k}: saved {v!r}, loaded {w!r}")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name; output goes to runs/train/<run>")
    ap.add_argument("--steps", type=int, default=50_000_000, help="total agent-steps for the run")
    ap.add_argument("--stop-at", type=int, help="stop (with a checkpoint) once this many steps are done")
    ap.add_argument("--resume", help="checkpoint folder, or 'latest' for the run's newest checkpoint")
    ap.add_argument("--init-from", help="start a new run from another run's checkpoint: weights and optimizer only "
                                        "(fresh step count, return statistics and experience buffer)")
    ap.add_argument("--critic-warmup", type=int, default=0,
                    help="iterations at the start that train only the critic (use with --init-from on a new reward)")
    ap.add_argument("--no-overlap", action="store_true", help="run the update in line instead of collecting during it")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--width", type=int, default=256, choices=(128, 256))
    ap.add_argument("--precision", default="float16", choices=("float16", "float32"))
    ap.add_argument("--phase", default="1a", choices=("1a", "1b", "nx"),
                    help="1b: Phase 1b reward (playstyle.rewards.PHASE_1B), coordination on, mixed start states")
    ap.add_argument("--lr", type=float, help="default 2e-4 in Phase 1a, 1e-4 in 1b")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--team-spirit", type=float, help="default 0 in Phase 1a, 0.4 in 1b (the first 500M ran at 0.2)")
    ap.add_argument("--opponent-scale", type=float, default=1.0)
    ap.add_argument("--practice-share", type=float,
                    help="Phase 1b and nx: share of episode starts that are practice shots (default playstyle.states."
                         "START_WEIGHTS, 0.8); whole-match starts keep their proportions")
    ap.add_argument("--aerial-share", type=float, default=0.0,
                    help="share of episode starts that are aerial practice (playstyle.states.aerial; reward nx "
                         "logs it); taken from the whole-match starts")
    ap.add_argument("--weight", action="append", default=[], metavar="NAME=VALUE",
                    help="override one reward weight, e.g. --weight shot_path=0 (repeatable; applied after --core-scale)")
    ap.add_argument("--core-scale", type=float, default=1.0,
                    help="Phase 1b reward: multiplies the core shaping weights (playstyle.rewards.CORE_TERMS); "
                         "Phase 1c anneals it")
    ap.add_argument("--save-every", type=int, default=5_000_000)
    ap.add_argument("--keep-every", type=int, default=100_000_000,
                    help="also copy the first checkpoint past every multiple of this to runs/train/<run>/kept/")
    ap.add_argument("--video-every", type=int, default=50_000_000,
                    help="record one full match (scripts/watch.py --full-game) from the first checkpoint past every "
                         "multiple of this, in the background, to runs/train/<run>/games/, and export it as an RLBot "
                         "bot to runs/train/<run>/rlbot/; 0 turns both off")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 2e-4 if args.phase == "1a" else 1e-4
    if args.team_spirit is None:
        args.team_spirit = 0.0 if args.phase == "1a" else 0.4

    import numpy as np
    import rlgym_learn
    import rlgym_learn_algos
    import torch
    from rlgym_learn import LearningCoordinator

    from playstyle import versions
    from playstyle.env import build_env
    from playstyle.network import N_ACTIONS
    from playstyle.obs import OBS_DIM
    from playstyle.rewards import PHASE_1A, PHASE_1B, scaled_core
    from playstyle.rewards_nx import CORE_TERMS_NX, PHASE_NX
    from playstyle.states import START_WEIGHTS, start_weights
    from playstyle.training import CONTROLLER, NETWORKS, RunLogger, make_agent_controller, make_config

    run_dir = ROOT / "runs" / "train" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    resume = None
    if args.resume:
        resume = checkpoints(run_dir)[-1] if args.resume == "latest" else Path(args.resume)
    done = json.loads((resume / "ppo_agent.json").read_text())["cumulative_timesteps"] if resume else 0
    stop = min(args.steps, args.stop_at or args.steps)
    if done >= stop:
        raise SystemExit(f"{done:,} steps already done, nothing to do before {stop:,}")

    if args.phase == "1a":
        weights = dict(PHASE_1A)
    elif args.phase == "1b":
        weights = scaled_core(PHASE_1B, args.core_scale)
    else:  # nx: playstyle.rewards_nx, Necto-style potentials
        weights = scaled_core(PHASE_NX, args.core_scale, CORE_TERMS_NX)
    for override in args.weight:
        name, value = override.split("=")
        if name not in weights:
            raise SystemExit(f"--weight: no reward term {name!r}; terms are {list(weights)}")
        weights[name] = float(value)
    starts = START_WEIGHTS if args.practice_share is None and not args.aerial_share else \
        start_weights(START_WEIGHTS["defend_shot"] if args.practice_share is None else args.practice_share,
                      args.aerial_share)
    env_fn = functools.partial(build_env, weights=weights, team_spirit=args.team_spirit,
                               opponent_scale=args.opponent_scale, phase=args.phase, start_weights=starts,
                               gamma=args.gamma)
    if args.init_from and resume:
        raise SystemExit("--init-from starts a new run; use --resume to carry on an existing one")
    controller = make_agent_controller(width=args.width, precision=args.precision,
                                       metrics_logger=RunLogger(path=run_dir / "log.jsonl", weights=weights),
                                       overlap=not args.no_overlap, init_from=args.init_from,
                                       component_names=tuple(weights))
    config = make_config(n_proc=args.procs, lr=args.lr, gamma=args.gamma, ent_coef=args.ent_coef,
                         timestep_limit=stop - done,  # the coordinator counts from 0 on every start
                         save_folder=str(run_dir / "checkpoints"), run_name=args.run,
                         save_every_ts=args.save_every, seed=args.seed, overlap=not args.no_overlap)
    if resume:
        config.agent_controllers_config[CONTROLLER]["checkpoint_load_folder"] = str(resume)

    record = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args), "steps_done_at_start": done,
        "resume_from": str(resume) if resume else None, "init_from": args.init_from,
        "reward": {"phase": args.phase, "core_scale": args.core_scale, "weights": weights, "team_spirit": args.team_spirit,
                   "opponent_scale": args.opponent_scale, "coordination": args.phase != "1a", "reward_ttb": "v1"},
        "start_states": starts if args.phase != "1a" else "kickoff",
        "network": {"width": args.width, **NETWORKS[args.width], "obs_dim": OBS_DIM, "actions": N_ACTIONS},
        "versions": {"obs": versions.OBS_VERSIONS, "reward": versions.reward_versions("v1")},
        "git": git_state(),
        "software": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "rlgym_learn": rlgym_learn.__version__ if hasattr(rlgym_learn, "__version__") else "1.0.5",
                     "rlgym_learn_algos": getattr(rlgym_learn_algos, "__version__", "0.2.6"),
                     "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None},
        "coordinator_config": json.loads(config.model_dump_json()),
    }
    name = f"resume-{done}.json" if resume else "config.json"
    (run_dir / name).write_text(json.dumps(record, indent=2, default=str))
    print(f"settings written to {run_dir / name}")

    save = controller.save_checkpoint
    kept = {"mark": done // args.keep_every, "video": done // args.video_every if args.video_every else None}

    def record_game(checkpoint, steps):
        """A full match from this checkpoint, rendered by a separate process so training does not wait for it."""
        games = run_dir / "games"
        games.mkdir(exist_ok=True)
        name = games / f"game_{steps // 1_000_000}M"
        with open(name.with_suffix(".log"), "w") as out:
            subprocess.Popen([sys.executable, str(ROOT / "scripts" / "watch.py"), "--checkpoint", str(checkpoint),
                              "--full-game", "--save", str(name.with_suffix(".mp4"))],
                             cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        print(f"recording a full game to {name.with_suffix('.mp4')}")

    def export_bot(checkpoint, steps):
        """An RLBot v5 bot folder for this checkpoint (scripts/export_rlbot.py), alongside each game video."""
        out = run_dir / "rlbot" / f"{steps // 1_000_000}M"
        out.parent.mkdir(exist_ok=True)
        with open(out.with_suffix(".log"), "w") as log:
            subprocess.Popen([sys.executable, str(ROOT / "scripts" / "export_rlbot.py"), "--checkpoint", str(checkpoint),
                              "--out", str(out)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"exporting an RLBot bot to {out}")

    def save_with_state():
        save()
        newest = max(Path(controller.checkpoints_save_folder).iterdir(), key=lambda p: int(p.name))
        (newest / RESUME_STATE).write_text(json.dumps(state_summary(controller), indent=2))
        print(f"checkpoint {newest} at {controller.cumulative_timesteps:,} steps")
        mark = controller.cumulative_timesteps // args.keep_every
        if mark > kept["mark"]:  # the rotating checkpoints keep only the newest few; keep one per mark for evals
            kept["mark"] = mark
            target = run_dir / "kept" / f"{mark * args.keep_every // 1_000_000}M"
            shutil.copytree(newest, target, dirs_exist_ok=True)
            print(f"kept a copy at {target}")
        if args.video_every and controller.cumulative_timesteps // args.video_every > kept["video"]:
            kept["video"] = controller.cumulative_timesteps // args.video_every
            record_game(newest, controller.cumulative_timesteps)
            export_bot(newest, controller.cumulative_timesteps)

    controller.save_checkpoint = save_with_state
    coordinator = LearningCoordinator(env_fn, {CONTROLLER: controller}, config)
    controller.learner.frozen_updates = args.critic_warmup

    if resume:
        saved = json.loads((resume / RESUME_STATE).read_text())
        diffs = compare(saved, state_summary(controller))
        with open(run_dir / "log.jsonl", "a") as f:
            f.write(json.dumps({"resume_check": {"from": str(resume), "differences": diffs}}) + "\n")
        if diffs:
            coordinator.cleanup()
            raise SystemExit("RESUME CHECK FAILED:\n  " + "\n  ".join(diffs))
        print(f"resume check passed: {len(saved)} items match the state saved at {saved['cumulative_timesteps']:,} steps")

    coordinator.start()
    controller.save_checkpoint()  # the coordinator only saves on errors or a keypress
    print(f"done: {controller.cumulative_timesteps:,} steps")


if __name__ == "__main__":
    main()
