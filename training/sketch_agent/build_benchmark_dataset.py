"""Offline benchmark-set builder: 500 rows from a local manifest (`simple_llm_description`), randomly split 50/50 into UltraSketch / displacement. No checkpoint/resume here as its's only 500 samples.

Usage:

    python -m training.sketch_agent.build_benchmark_dataset \\
        --manifest /usr/prakt/s0031/projects/data/simple_llm_description/manifest.csv \\
        --output-dir training/sketch_agent/output_final/benchmark
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import SketchAugmentationConfig
from .sketch_generation import (
    generate_synthetic_sketch,
    load_ultrasketch_pipeline,
    release_ultrasketch_pipeline,
)

def _resolve_row_paths(row: Dict[str, str], source_root: Path) -> Dict[str, Path]:
    """Use correct paths"""
    return {
        "image": source_root / "images" / Path(row["reference_image"]).name,
        "code": source_root / "references" / Path(row["reference_code"]).name,
        "description": source_root / "descriptions" / Path(row["llm_description"]).name,
    }


def _read_description(text: str) -> Optional[str]:
    """Each descriptions/*.txt's whole content"""
    text = text.strip()
    return text or None


def _load_manifest_rows(manifest_path: Path, source_root: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({"id": row["id"], **_resolve_row_paths(row, source_root)})
    return rows


def _assign_methods(row_ids: List[str], seed: int) -> Dict[str, str]:
    """Random 50/50 assignment: half the rows get "ultrasketch",
    half get "displacement"."""
    import numpy as np

    order = np.random.default_rng(seed).permutation(len(row_ids))
    half = len(row_ids) // 2
    return {
        row_ids[index]: ("ultrasketch" if position < half else "displacement")
        for position, index in enumerate(order)
    }


def _build_features():
    """Lazy import: keeps this module importable without `datasets` installed. Same column
    names as build_sketch_dataset.py's training-split schema (image/code/description/
    sketch_method) for consistency but no source_variant (this has no with-text/
    without-text pairing)."""
    from datasets import Features, Image as HFImage, Value

    return Features(
        {
            "image": HFImage(),  # the sketch
            "code": Value("string"),
            "description": Value("string"),
            "sketch_method": Value("string"),  # "ultrasketch" or "displacement"
        }
    )


def _write_shard(output_dir: Path, split_name: str, rows: List[dict]) -> None:
    from datasets import Dataset

    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{split_name}_0000000_{len(rows):07d}.parquet"
    Dataset.from_list(rows, features=_build_features()).to_parquet(str(shard_path))


def _generate_group(
    rows: List[Dict[str, Any]],
    method: str,
    cfg: SketchAugmentationConfig,
    base_seed: int,
    pipe: Any = None,
) -> List[dict]:
    from PIL import Image as PILImage

    ultrasketch_probability = 1.0 if method == "ultrasketch" else 0.0
    output_rows = []
    for row in rows:
        image = PILImage.open(row["image"]).convert("RGB")
        code = row["code"].read_text(encoding="utf-8")
        description = _read_description(row["description"].read_text(encoding="utf-8"))
        seed = base_seed + int(row["id"])

        sketch, sketch_method = generate_synthetic_sketch(
            image,
            seed=seed,
            sketch_probability=1.0,
            ultrasketch_probability=ultrasketch_probability,
            displacement_alpha=cfg.displacement_alpha,
            displacement_sigma=cfg.displacement_sigma,
            pipe=pipe,
        )
        output_rows.append(
            {
                "image": sketch,
                "code": code,
                "description": description,
                "sketch_method": sketch_method,
            }
        )
    return output_rows


def build_benchmark_dataset(
    manifest_path: str | Path,
    output_dir: str | Path,
    source_root: Optional[str | Path] = None,
    cfg: Optional[SketchAugmentationConfig] = None,
    base_seed: int = 3407,
    max_rows: Optional[int] = None,
) -> None:
    """Runs the whole 500-row (or `max_rows`-capped)
    benchmark build."""
    manifest_path = Path(manifest_path)
    source_root = Path(source_root) if source_root is not None else manifest_path.parent
    output_dir = Path(output_dir)
    cfg = cfg or SketchAugmentationConfig()

    rows = _load_manifest_rows(manifest_path, source_root)
    if max_rows is not None:
        rows = rows[:max_rows]
    methods = _assign_methods([row["id"] for row in rows], base_seed)

    displacement_rows = [row for row in rows if methods[row["id"]] == "displacement"]
    ultrasketch_rows = [row for row in rows if methods[row["id"]] == "ultrasketch"]

    displacement_output = _generate_group(displacement_rows, "displacement", cfg, base_seed)
    _write_shard(output_dir, "displacement", displacement_output)

    pipe = load_ultrasketch_pipeline()
    try:
        ultrasketch_output = _generate_group(
            ultrasketch_rows, "ultrasketch", cfg, base_seed, pipe=pipe
        )
    finally:
        release_ultrasketch_pipeline(pipe)
    _write_shard(output_dir, "ultrasketch", ultrasketch_output)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "DONE").write_text("finished\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument(
        "--source-root", type=str, default=None,
        help="root containing images/, references/, descriptions/; defaults to --manifest's parent dir",
    )
    parser.add_argument(
        "--output-dir", type=str, default="training/sketch_agent/output_final/benchmark",
    )
    parser.add_argument("--base-seed", type=int, default=3407)
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="cap total rows processed",
    )
    args = parser.parse_args()

    build_benchmark_dataset(
        args.manifest,
        args.output_dir,
        source_root=args.source_root,
        base_seed=args.base_seed,
        max_rows=args.max_rows,
    )

    print(f"Finished: {args.output_dir}/DONE written.")
    print(f"Shards under {args.output_dir}/shards/. Nothing was pushed: review, then push yourself, e.g.:")
    print("  from datasets import DatasetDict")
    print("  from training.sketch_agent.build_sketch_dataset import assemble_dataset_dict")
    print(
        f"  benchmark = assemble_dataset_dict({args.output_dir!r}, rename="
        "{'ultrasketch': 'benchmark_ultrasketch', 'displacement': 'benchmark_displacement'})"
    )
    print("  DatasetDict(benchmark).push_to_hub('loss-boss/tikz-train')")


if __name__ == "__main__":
    main()
