from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


def resolve_latest_dir(base_dir: Path, manifest_name: str) -> Optional[tuple[Path, int]]:
    """Pick a step_XXXXXX subdirectory of base_dir: prefer the manifest's pointer,
    falling back to a numeric sort of step_* directories."""
    manifest_path = base_dir / manifest_name
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        latest = Path(manifest["path"])
        if latest.exists():
            return latest, int(manifest["step"])

    dirs = sorted(base_dir.glob("step_*"), key=lambda path: int(path.name.rsplit("_", 1)[-1]))
    if not dirs:
        return None
    return dirs[-1], int(dirs[-1].name.rsplit("_", 1)[-1])


def write_latest_manifest(base_dir: Path, manifest_name: str, step: int, path: Path) -> None:
    (base_dir / manifest_name).write_text(json.dumps({"step": step, "path": str(path)}), encoding="utf-8")


def resolve_run_name(output_dir: Path, checkpoint_dir: Path, explicit_name: Optional[str] = None) -> str:
    """Pick TensorBoard run name.

    Resuming from an existing checkpoint reuses the name already recorded for this run (so
    self-resubmits keep logging into the same TensorBoard run and the curve stays continuous)

    starting a new one creates a different run name
    """
    run_name_file = output_dir / "current_run_name.txt"
    resuming = checkpoint_dir.exists() and resolve_latest_dir(checkpoint_dir, "latest_checkpoint.json") is not None
    if resuming and run_name_file.exists():
        return run_name_file.read_text(encoding="utf-8").strip()

    run_name = explicit_name or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_job{os.environ.get('SLURM_JOB_ID', 'local')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name_file.write_text(run_name, encoding="utf-8")
    return run_name


def clear_resumable_state(checkpoint_dir: Path, run_name_file: Path) -> None:
    """Remove the accelerate resume state and run-name bookkeeping once a run
    finished (reached max_steps)."""
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    run_name_file.unlink(missing_ok=True)
