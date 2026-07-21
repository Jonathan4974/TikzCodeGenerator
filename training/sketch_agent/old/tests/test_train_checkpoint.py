import tempfile
from pathlib import Path

from training.sketch_agent.checkpoint_utils import clear_resumable_state, resolve_latest_dir, resolve_run_name
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


def test_resolve_run_name_mints_new_name_when_starting_fresh():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-run-") as tmp:
        output_dir = Path(tmp) / "output"
        checkpoint_dir = output_dir / "checkpoints"

        run_name = resolve_run_name(output_dir, checkpoint_dir)

        assert run_name
        assert (output_dir / "current_run_name.txt").read_text(encoding="utf-8") == run_name


def test_resolve_run_name_honors_explicit_override():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-run-") as tmp:
        output_dir = Path(tmp) / "output"
        checkpoint_dir = output_dir / "checkpoints"

        run_name = resolve_run_name(output_dir, checkpoint_dir, explicit_name="lr_scheduler_fix")

        assert run_name == "lr_scheduler_fix"


def test_resolve_run_name_reuses_saved_name_when_resuming():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-run-") as tmp:
        output_dir = Path(tmp) / "output"
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "step_000200").mkdir()
        (checkpoint_dir / "latest_checkpoint.json").write_text(
            '{"step": 200, "path": "%s"}' % (checkpoint_dir / "step_000200")
        )
        output_dir.mkdir(exist_ok=True)
        (output_dir / "current_run_name.txt").write_text("original_run", encoding="utf-8")

        run_name = resolve_run_name(output_dir, checkpoint_dir, explicit_name="should_be_ignored")

        assert run_name == "original_run"


def test_resolve_run_name_mints_new_name_when_no_checkpoint_even_if_file_exists():
    # A leftover run-name file from a since-cleared run shouldn't be reused once its
    # checkpoints are gone - that would silently glue an unrelated new attempt onto old curves.
    with tempfile.TemporaryDirectory(prefix="sketch-agent-run-") as tmp:
        output_dir = Path(tmp) / "output"
        checkpoint_dir = output_dir / "checkpoints"
        output_dir.mkdir(parents=True)
        (output_dir / "current_run_name.txt").write_text("stale_run", encoding="utf-8")

        run_name = resolve_run_name(output_dir, checkpoint_dir, explicit_name="fresh_run")

        assert run_name == "fresh_run"


def test_clear_resumable_state_removes_checkpoints_and_run_name_file():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-cleanup-") as tmp:
        output_dir = Path(tmp) / "output"
        checkpoint_dir = output_dir / "checkpoints"
        (checkpoint_dir / "step_005000").mkdir(parents=True)
        (checkpoint_dir / "step_005000" / "model.safetensors").write_bytes(b"")
        run_name_file = output_dir / "current_run_name.txt"
        run_name_file.write_text("finished_run", encoding="utf-8")

        clear_resumable_state(checkpoint_dir, run_name_file)

        assert not checkpoint_dir.exists()
        assert not run_name_file.exists()


def test_clear_resumable_state_leaves_results_dirs_alone():
    # Cleanup must only ever touch checkpoint_dir/run_name_file -- output/lora,
    # eval_previews, and tensorboard are the actual results and must survive.
    with tempfile.TemporaryDirectory(prefix="sketch-agent-cleanup-") as tmp:
        output_dir = Path(tmp) / "output"
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        run_name_file = output_dir / "current_run_name.txt"
        run_name_file.write_text("finished_run", encoding="utf-8")

        lora_dir = output_dir / "lora" / "step_005000"
        lora_dir.mkdir(parents=True)
        (lora_dir / "adapter.safetensors").write_bytes(b"")

        clear_resumable_state(checkpoint_dir, run_name_file)

        assert (lora_dir / "adapter.safetensors").exists()


def test_clear_resumable_state_is_a_noop_when_nothing_to_clear():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-cleanup-") as tmp:
        output_dir = Path(tmp) / "output"
        checkpoint_dir = output_dir / "checkpoints"
        run_name_file = output_dir / "current_run_name.txt"

        clear_resumable_state(checkpoint_dir, run_name_file)  # must not raise

        assert not checkpoint_dir.exists()
        assert not run_name_file.exists()
