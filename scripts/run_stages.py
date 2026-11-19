"""Run a multi-stage training schedule unattended (Phase 1c by default).

Each stage is a resume of the same run with that stage's settings (core shaping scale, gamma, team spirit), so a
schedule needs nothing inside training, and train.py's resume check confirms the state carries over at every
boundary. The first stage starts from --init-from if the run has no checkpoints yet; later stages start with a short
critic warm-up, since the reward changes. If train.py exits before its stage is done (a crash, a killed process),
the run is resumed from its newest checkpoint, giving up after --max-restarts tries in a row without progress.

Every kept checkpoint (train.py --keep-every) gets 200 full matches (scripts/round_robin.py) against the previous kept
checkpoint and against --init-from (the gate: it must not lose to where the run started), the clean kickoff eval and
the shot eval at full difficulty and at level 0, run alongside training and appended to runs/train/<run>/evals.jsonl.
Runner events go to stages.log.

While running it asks Windows not to sleep. It cannot stop Windows Update restarts: pause updates for the run.

    python scripts/run_stages.py --run phase1c --init-from runs/train/phase1b-5/kept/100M
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# Phase 1c (design doc): outcome-dominant, the core shaping layer annealed, gamma raised. Step counts are this run's.
# Stage 1 runs to 2B: at 750M of phase1c-nx (gold-level play) the bots still did not aerial (their airborne touches
# were jump touches) and positioning was still improving, so the core shaping stays at full strength longer.
STAGES_1C = [
    {"until": 2_000_000_000, "core_scale": 1.0, "gamma": 0.99, "team_spirit": 0.4},
    {"until": 3_000_000_000, "core_scale": 0.5, "gamma": 0.993, "team_spirit": 0.6},
    {"until": 5_000_000_000, "core_scale": 0.25, "gamma": 0.995, "team_spirit": 0.8},
]
# Full opponent subtraction: at 0.25, crowding rose from 12% to ~16% within 50M steps in two runs (phase1b-5, -6).
# Practice shots at 30% of episodes: at 80% the run lost 0.6-0.8 goals per match to phase1b-3; a 30% branch beat it
# (+0.41) and the 80% control (+0.60) while keeping most practice-save skill (round robin, 200 matches per pair).
# Aerial practice (playstyle.states.aerial) at 25% of episodes, ~1% of steps: drills last at most 4 s.
COMMON = ["--opponent-scale", "1.0", "--ent-coef", "0.003", "--save-every", "10000000", "--practice-share", "0.3",
          "--aerial-share", "0.25"]
FIRST_WARMUP, STAGE_WARMUP = 10, 5  # critic-only iterations at the start of the run / of each later stage


def steps_done(run_dir):
    states = sorted(Path(run_dir).glob("checkpoints/*/*/*/resume_state.json"), key=lambda p: int(p.parent.name))
    return json.loads(states[-1].read_text())["cumulative_timesteps"] if states else 0


def stage_index(stages, done):
    """Index of the stage that the next steps belong to, or None when the schedule is finished."""
    return next((i for i, s in enumerate(stages) if s["until"] > done), None)


def command(run, stages, i, done, init_from, keep_every, warmup, phase="1b"):
    s = stages[i]
    cmd = [PYTHON, str(ROOT / "scripts" / "train.py"), "--run", run, "--phase", phase, "--steps", str(stages[-1]["until"]),
           "--stop-at", str(s["until"]), "--keep-every", str(keep_every), "--core-scale", str(s["core_scale"]),
           "--gamma", str(s["gamma"]), "--team-spirit", str(s["team_spirit"]), *COMMON]
    if done == 0:
        return cmd + ["--init-from", str(init_from), "--critic-warmup", str(FIRST_WARMUP)]
    return cmd + ["--resume", "latest"] + (["--critic-warmup", str(warmup)] if warmup else [])


class Evaluator(threading.Thread):
    """Evaluates each new kept checkpoint once, while training continues."""

    EVALS = (["touch_check.py", "--minutes", "30"], ["shot_eval.py", "--shots", "300", "--level", "1.0"],
             ["shot_eval.py", "--shots", "300", "--level", "0.0"])

    def __init__(self, run_dir, log, start_checkpoint=None):
        super().__init__(daemon=True)
        self.run_dir, self.log, self.stop = Path(run_dir), log, threading.Event()
        self.results = self.run_dir / "evals.jsonl"
        self.start_checkpoint = start_checkpoint  # the first kept checkpoint plays against this

    def done_checkpoints(self):
        if not self.results.exists():
            return set()
        return {json.loads(line).get("checkpoint") for line in self.results.read_text().splitlines() if line.strip()}

    def pending(self):
        kept = sorted((p for p in (self.run_dir / "kept").glob("*M") if p.is_dir()), key=lambda p: int(p.name[:-1]))
        done = self.done_checkpoints()
        return [p for p in kept if str(p) not in done]

    def previous(self, checkpoint):
        kept = sorted((p for p in (self.run_dir / "kept").glob("*M") if p.is_dir()), key=lambda p: int(p.name[:-1]))
        earlier = [p for p in kept if int(p.name[:-1]) < int(checkpoint.name[:-1])]
        return earlier[-1] if earlier else self.start_checkpoint

    def run(self):
        while True:
            for checkpoint in self.pending():
                self.log(f"evaluating {checkpoint.name}")
                # Head to head, 200 full matches each: against the previous kept checkpoint and the starting one.
                for opponent in dict.fromkeys(p for p in (self.previous(checkpoint), self.start_checkpoint) if p):
                    subprocess.run([PYTHON, str(ROOT / "scripts" / "round_robin.py"), "--checkpoints",
                                    f"{checkpoint.name}={checkpoint}", f"opponent={opponent}", "--matches", "200",
                                    "--workers", "4", "--json", str(self.results)], cwd=ROOT,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for script, *args in self.EVALS:
                    subprocess.run([PYTHON, str(ROOT / "scripts" / script), "--checkpoint", str(checkpoint), *args,
                                    "--json", str(self.results)], cwd=ROOT, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                self.log(f"evaluated {checkpoint.name}")
            if self.stop.wait(120) and not self.pending():
                return


def keep_awake():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--init-from", help="checkpoint to start from when the run has no checkpoints yet")
    ap.add_argument("--keep-every", type=int, default=250_000_000)
    ap.add_argument("--max-restarts", type=int, default=5)
    ap.add_argument("--phase", default="1b", choices=("1b", "nx"),
                    help="reward: 1b = playstyle.rewards.Phase1bReward, nx = playstyle.rewards_nx.NextoStyleReward")
    args = ap.parse_args()

    run_dir = ROOT / "runs" / "train" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(run_dir / "stages.log", "a") as f:
            f.write(line + "\n")

    if steps_done(run_dir) == 0 and not args.init_from:
        raise SystemExit("the run has no checkpoints yet: give --init-from")
    keep_awake()
    evaluator = Evaluator(run_dir, log, start_checkpoint=args.init_from)
    evaluator.start()
    stages, failures = STAGES_1C, 0
    log(f"schedule: {json.dumps(stages)}")
    previous = None
    while True:
        done = steps_done(run_dir)
        i = stage_index(stages, done)
        if i is None:
            break
        # A short critic warm-up when this segment starts a later stage (the reward and gamma just changed), not when
        # it resumes the same stage after a crash.
        warmup = STAGE_WARMUP if done > 0 and i > 0 and (previous is None or previous != i) \
            and done < stages[i - 1]["until"] + 20_000_000 else 0
        cmd = command(args.run, stages, i, done, args.init_from, args.keep_every, warmup, args.phase)
        log(f"stage {i + 1}/{len(stages)} from {done:,} steps: {' '.join(cmd[2:])}")
        code = subprocess.run(cmd, cwd=ROOT).returncode
        after = steps_done(run_dir)
        log(f"train.py exited with {code} at {after:,} steps")
        if after >= stages[i]["until"]:
            previous, failures = i, 0
            continue
        failures = 0 if after > done else failures + 1
        previous = i
        if failures > args.max_restarts:
            log(f"giving up: {failures} restarts in a row without progress")
            break
        time.sleep(60)
    evaluator.stop.set()
    evaluator.join()
    log("schedule finished" if stage_index(stages, steps_done(run_dir)) is None else "stopped")


if __name__ == "__main__":
    main()
