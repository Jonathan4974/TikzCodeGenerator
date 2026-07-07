import tempfile
from pathlib import Path

from training.sketch_agent.checkpoint_utils import resolve_latest_dir
from training.sketch_agent.config import build_training_config
from training.sketch_agent.trainer import _resolve_self_resubmit_command


def test_resolve_checkpoint_dir_picks_highest_step_across_double_digits():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-resume-") as tmp:
        checkpoint_dir = Path(tmp)
        for step in (2, 8, 10, 12):
            (checkpoint_dir / f"step_{step:06d}").mkdir()
        (checkpoint_dir / "latest_checkpoint.json").write_text(
            '{"step": 12, "path": "%s"}' % (checkpoint_dir / "step_000012")
        )

        resolved = resolve_latest_dir(checkpoint_dir, "latest_checkpoint.json")
        assert resolved is not None
        assert resolved[1] == 12


def test_resolve_checkpoint_dir_falls_back_to_numeric_sort_without_manifest():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-resume-") as tmp:
        checkpoint_dir = Path(tmp)
        # Out of order steps ("8" sorts after "12" and "2" as strings).
        for step in (2, 8, 10, 12):
            (checkpoint_dir / f"step_{step:06d}").mkdir()

        resolved = resolve_latest_dir(checkpoint_dir, "latest_checkpoint.json")
        assert resolved is not None
        assert resolved[1] == 12


def test_resolve_self_resubmit_command_targets_sbatch_script_not_train_py():
    config = build_training_config({"self_resubmit": True})

    command = _resolve_self_resubmit_command(config, is_main_process=True, job_id="12345")

    assert command is not None
    assert command[0] == "sbatch"
    assert command[1] == "--dependency=afterany:12345"
    assert command[2].endswith("train.sbatch")


def test_resolve_self_resubmit_command_is_none_without_job_id():
    config = build_training_config({"self_resubmit": True})
    assert _resolve_self_resubmit_command(config, is_main_process=True, job_id=None) is None


def test_resolve_self_resubmit_command_is_none_when_disabled():
    config = build_training_config({"self_resubmit": False})
    assert _resolve_self_resubmit_command(config, is_main_process=True, job_id="12345") is None
