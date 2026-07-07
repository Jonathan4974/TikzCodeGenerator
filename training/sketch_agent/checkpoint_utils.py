from __future__ import annotations

import json
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
