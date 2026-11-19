"""Export a checkpoint as an RLBot v5 bot folder: bot.toml, bot.py (rlbot_bot/), the actor weights and a snapshot of
the playstyle package, so later code changes cannot break the export. The bot runs with this project's venv, which
has RocketSim, rlgym and the RLBot packages. Add the folder's bot.toml to a match in RLBot.

    python scripts/export_rlbot.py --checkpoint runs/train/phase1c-nx/kept/250M --out runs/train/phase1c-nx/rlbot/250M
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export(checkpoint, out, name):
    checkpoint, out = Path(checkpoint), Path(out)
    learner = checkpoint / "ppo_learner"
    if out.exists():
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True)
    for f in ("actor.pt", "versions.json"):
        shutil.copy2(learner / f, out / "model" / f)
    shutil.copytree(ROOT / "playstyle", out / "playstyle", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "rlbot_bot" / "bot.py", out / "bot.py")
    run_command = '"' + str(ROOT / ".venv" / "Scripts" / "python.exe") + '" bot.py'
    (out / "bot.toml").write_text(
        "[settings]\n"
        f"name = {json.dumps(name)}\n"
        f"agent_id = {json.dumps('playstyle/' + name.lower().replace(' ', '-'))}\n"
        f"run_command = {json.dumps(run_command)}\n"
        "\n[details]\n"
        f"description = {json.dumps(f'2v2 PPO bot exported from {checkpoint}')}\n"
        'language = "python"\n')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="a checkpoint folder holding ppo_learner/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", help="bot name in RLBot (default: the output folder's parent run and name)")
    args = ap.parse_args()
    out = Path(args.out)
    name = args.name or f"{out.parent.parent.name} {out.name}"
    print(f"exported {name!r} to {export(args.checkpoint, out, name)}")


if __name__ == "__main__":
    sys.exit(main())
