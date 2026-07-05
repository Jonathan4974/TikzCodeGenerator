from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
from PIL import Image

from .data import SyntheticPair


def load_real_pairs(data_dir: str | Path) -> List[SyntheticPair]:
    root = Path(data_dir)
    if not root.exists():
        return []

    sketch_dir = root / "sketches"
    target_dir = root / "targets"
    if not sketch_dir.exists() or not target_dir.exists():
        return []

    pairs: List[SyntheticPair] = []
    for sketch_path in sorted(sketch_dir.glob("*.png")):
        stem = sketch_path.stem
        target_path = target_dir / f"{stem}_target.png"
        if target_path.exists():
            pairs.append(
                SyntheticPair(
                    input_path=sketch_path,
                    target_path=target_path,
                    source_name=stem,
                    method="real_data",
                )
            )
    return pairs


@dataclass
class SketchFigSplit:
    train: List[SyntheticPair]
    eval: List[SyntheticPair]


def _safe_stem(uri: Optional[str], index: int) -> str:
    if not uri:
        return f"sketchfig_{index:04d}"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", uri).strip("_")[:80] or f"sketchfig_{index:04d}"


def _default_sketchfig_loader(dataset_name: str, split: str) -> Any:
    from datasets import load_dataset

    return load_dataset(dataset_name, split=split)


def load_sketchfig_dataset(
    cache_dir: str | Path,
    train_fraction: float = 0.0,
    seed: int = 3407,
    dataset_name: str = "nllg/sketchfig",
    split: str = "train",
    dataset_loader: Optional[Callable[[], Any]] = None,
) -> SketchFigSplit:
    """Load SketchFig
    """
    loader = dataset_loader or (lambda: _default_sketchfig_loader(dataset_name, split))
    ds = loader()

    cache_root = Path(cache_dir)
    train_dir = cache_root / "train"
    eval_dir = cache_root / "eval"
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
