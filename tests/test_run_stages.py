import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_stages import FIRST_WARMUP, STAGES_1C, command, stage_index, steps_done


def test_stage_index_follows_the_schedule():
    assert stage_index(STAGES_1C, 0) == 0
    assert stage_index(STAGES_1C, 999_999_999) == 0
    assert stage_index(STAGES_1C, 1_000_000_000) == 1
    assert stage_index(STAGES_1C, 2_500_000_000) == 2
    assert stage_index(STAGES_1C, 4_000_000_000) is None


def test_commands_start_from_init_then_resume_with_each_stages_settings():
    first = command("r", STAGES_1C, 0, 0, "ckpt", 250_000_000, 0)
    assert first[first.index("--init-from") + 1] == "ckpt"
    assert first[first.index("--critic-warmup") + 1] == str(FIRST_WARMUP) and "--resume" not in first
    assert first[first.index("--stop-at") + 1] == "1000000000"
    later = command("r", STAGES_1C, 1, 1_000_000_000, "ckpt", 250_000_000, 5)
    assert later[later.index("--resume") + 1] == "latest" and "--init-from" not in later
    for flag, value in (("--core-scale", "0.5"), ("--gamma", "0.993"), ("--team-spirit", "0.6"),
                        ("--critic-warmup", "5"), ("--steps", "4000000000"), ("--opponent-scale", "0.25")):
        assert later[later.index(flag) + 1] == value, flag
    assert "--critic-warmup" not in command("r", STAGES_1C, 1, 1_300_000_000, "ckpt", 250_000_000, 0)


def test_steps_done_reads_the_newest_checkpoint(tmp_path):
    assert steps_done(tmp_path) == 0
    for name, steps in (("100", 10), ("300", 30), ("200", 20)):
        folder = tmp_path / "checkpoints" / "ppo" / "run-1" / name
        folder.mkdir(parents=True)
        (folder / "resume_state.json").write_text(json.dumps({"cumulative_timesteps": steps}))
    assert steps_done(tmp_path) == 30
