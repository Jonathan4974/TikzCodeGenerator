import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from training.sketch_agent.config import build_training_config
from training.sketch_agent.train import _load_checkpoint, _maybe_self_resubmit, _save_checkpoint


def test_load_checkpoint_picks_highest_step_across_double_digits():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-resume-") as tmp:
        checkpoint_dir = Path(tmp)
        config = build_training_config()

        # Write out-of-lexical-order steps (8 sorts after 12 and 2 as strings).
        for step in (2, 8, 10, 12):
            _save_checkpoint(checkpoint_dir, step, config, metadata={"step_marker": step})

        resumed = _load_checkpoint(checkpoint_dir)
        assert resumed is not None
        assert resumed["step"] == 12
        assert resumed["metadata"]["step_marker"] == 12


def test_load_checkpoint_falls_back_to_numeric_sort_without_manifest():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-resume-") as tmp:
        checkpoint_dir = Path(tmp)
        config = build_training_config()

        for step in (2, 8, 10, 12):
            _save_checkpoint(checkpoint_dir, step, config, metadata={})
        (checkpoint_dir / "latest_checkpoint.json").unlink()

        resumed = _load_checkpoint(checkpoint_dir)
        assert resumed is not None
        assert resumed["step"] == 12


def test_maybe_self_resubmit_targets_sbatch_script_not_train_py():
    config = build_training_config({"self_resubmit": True})

    with patch.dict("os.environ", {"SLURM_JOB_ID": "12345"}), patch(
        "training.sketch_agent.train.subprocess.run"
    ) as mock_run:
        command = _maybe_self_resubmit(config, step=8)

    mock_run.assert_called_once()
    called_command = mock_run.call_args[0][0]
    assert called_command[0] == "sbatch"
    assert called_command[1] == "--dependency=afterany:12345"
    assert called_command[2].endswith("train.sbatch")
    assert command.endswith("train.sbatch")
