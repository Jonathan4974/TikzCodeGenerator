import importlib
import json
import tempfile
from pathlib import Path


def test_sketch_agent_package_exposes_training_entrypoints():
    module = importlib.import_module("training.sketch_agent")

    assert hasattr(module, "build_training_config")
    assert hasattr(module, "build_synthetic_dataset")
    assert hasattr(module, "run_training")


def test_checkpoint_resume_writes_latest_manifest():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-checkpoint-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        from training.sketch_agent import build_training_config, run_training

        config = build_training_config(
            {
                "output_dir": str(temp_dir / "output"),
                "checkpoint_dir": str(temp_dir / "output" / "checkpoints"),
                "synthetic_dir": str(temp_dir / "output" / "synthetic_pairs"),
                "max_steps": 2,
                "checkpoint_interval_steps": 2,
                "synthetic_samples": 2,
                "image_size": 128,
                "seed": 3,
            }
        )
        first_run = run_training(config=config)
        manifest_path = Path(config.checkpoint_dir) / "latest_checkpoint.json"

        assert first_run["status"] == "ok"
        assert manifest_path.exists(), "Latest checkpoint manifest should be written"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["step"] == 2
