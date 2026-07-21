from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .data import SyntheticPair


@dataclass
class SketchFigSplit:
    train: List[SyntheticPair]
    eval: List[SyntheticPair]


def _safe_stem(uri: Optional[str], index: int) -> str:
    if not uri:
        return f"sketchfig_{index:04d}"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", uri).strip("_")[:80] or f"sketchfig_{index:04d}"


def _load_cached_split(
    train_dir: Path, eval_dir: Path, split_record_path: Path, seed: int, train_fraction: float
) -> Optional[SketchFigSplit]:
    """Reconstructs a SketchFigSplit straight from already-cached files on disk, if a split
    with this exact seed/train_fraction was already written  skips re-downloading and
    re-writing everything."""
    if not split_record_path.exists():
        return None
    try:
        record = json.loads(split_record_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if record.get("seed") != seed or record.get("train_fraction") != train_fraction:
        return None

    def _pairs_from(out_dir: Path) -> List[SyntheticPair]:
        pairs = []
        for input_path in sorted(out_dir.glob("*_input.png")):
            stem = input_path.stem.replace("_input", "")
            target_path = out_dir / f"{stem}_target.png"
            if target_path.exists():
                pairs.append(
                    SyntheticPair(input_path=input_path, target_path=target_path, source_name=stem, method="real_sketchfig")
                )
        return pairs

    train_pairs = _pairs_from(train_dir)
    eval_pairs = _pairs_from(eval_dir)
    if len(train_pairs) != len(record.get("train_indices", [])) or len(eval_pairs) != len(record.get("eval_indices", [])):
        return None  # incomplete/corrupted cache - fall through and regenerate for real

    return SketchFigSplit(train=train_pairs, eval=eval_pairs)


def load_sketchfig_dataset(
    cache_dir: str | Path,
    train_fraction: float = 0.0,
    seed: int = 3407,
    dataset_name: str = "nllg/sketchfig",
    split: str = "train",
) -> SketchFigSplit:
    """Load SketchFig, splitting into a train slice and an eval-only holdout."""
    cache_root = Path(cache_dir)
    train_dir = cache_root / "train"
    eval_dir = cache_root / "eval"
    split_record_path = cache_root / "sketchfig_split.json"

    cached = _load_cached_split(train_dir, eval_dir, split_record_path, seed, train_fraction)
    if cached is not None:
        return cached

    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)

    # Clear any stale files from a previous seed/train_fraction before writing the new split
    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(eval_dir, ignore_errors=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    num_rows = len(ds)
    order = np.random.default_rng(seed).permutation(num_rows)
    num_train = int(num_rows * train_fraction)
    train_indices = set(order[:num_train].tolist())

    train_pairs: List[SyntheticPair] = []
    eval_pairs: List[SyntheticPair] = []
    split_record = {"seed": seed, "train_fraction": train_fraction, "train_indices": [], "eval_indices": []}

    for index in range(num_rows):
        row = ds[index]
        stem = _safe_stem(row.get("uri"), index)
        is_train = index in train_indices
        out_dir = train_dir if is_train else eval_dir

        sketch_path = out_dir / f"{stem}_input.png"
        target_path = out_dir / f"{stem}_target.png"
        row["sketch"].convert("RGB").save(sketch_path)
        row["image"].convert("RGB").save(target_path)

        pair = SyntheticPair(
            input_path=sketch_path,
            target_path=target_path,
            source_name=row.get("uri") or stem,
            method="real_sketchfig",
        )
        if is_train:
            train_pairs.append(pair)
            split_record["train_indices"].append(index)
        else:
            eval_pairs.append(pair)
            split_record["eval_indices"].append(index)

    (cache_root / "sketchfig_split.json").write_text(json.dumps(split_record, indent=2), encoding="utf-8")

    return SketchFigSplit(train=train_pairs, eval=eval_pairs)
