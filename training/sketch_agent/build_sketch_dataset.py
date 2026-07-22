"""Offline dataset builder: generates a `sketch` column for every row of the source
split(s) and writes image/sketch/code/description/source rows to disk.

Per row, one draw picks the sketch method - 50% UltraSketch, 50% displacement field,
never both. Uses a reproducible seed (`base_seed + row_index`) so the job is resumable.
The clean-vs-sketch choice used for actual training happens later, in
sketch_choice_dataset.py - this always generates a sketch.

Checkpoint/resume: progress per split is written to `<output_dir>/checkpoint.json`, and
rows are flushed to parquet shards under `<output_dir>/shards/` every
`checkpoint_interval` rows, so a killed/resubmitted job picks up where it left off. If
`--time-limit-hours` is hit before every split finishes, it self-resubmits via
`sbatch --dependency=afterany:$SLURM_JOB_ID build_sketch_dataset.sbatch` - this only runs
from inside an already-submitted SLURM job, never on its own.

Never pushes to the Hub. Once `<output_dir>/DONE` exists:
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files="<output_dir>/shards/*.parquet", split="train")
    ds.push_to_hub("your-username/your-repo")

Usage:

    python -m training.sketch_agent.build_sketch_dataset \\
        --split datikz_v4 --source-label datikzv4 \\
        --split geotikz_bridge_base --source-label geotikz \\
        --output-dir training/sketch_agent/output_final/sketch_dataset \\
        --max-rows 100
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .config import SketchAugmentationConfig
from .dataset_loader import IMAGE_COLUMN, code_column, load_split
from .sketch_generation import (
    generate_synthetic_sketch,
    load_ultrasketch_pipeline,
    release_ultrasketch_pipeline,
)


def _build_features():
    """Lazy import - keeps this module importable without `datasets` installed."""
    from datasets import Features, Image as HFImage, Value

    return Features(
        {
            "image": HFImage(),
            "sketch": HFImage(),
            "code": Value("string"),
            "description": Value("string"),
            "source": Value("string"),
            # Which method produced `sketch` - useful for verifying the realized split
            # lands near 50/50. Drop it before pushing if you don't want it.
            "sketch_method": Value("string"),
        }
    )


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint.json"


def _load_checkpoint(output_dir: Path) -> dict:
    path = _checkpoint_path(output_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_checkpoint(output_dir: Path, progress: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _checkpoint_path(output_dir).write_text(json.dumps(progress, indent=2), encoding="utf-8")


def _write_shard(output_dir: Path, split_name: str, start_index: int, rows: List[dict]) -> None:
    from datasets import Dataset

    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    end_index = start_index + len(rows)
    shard_path = shard_dir / f"{split_name}_{start_index:07d}_{end_index:07d}.parquet"
    Dataset.from_list(rows, features=_build_features()).to_parquet(str(shard_path))


def _flush_and_checkpoint(
    output_dir: Path, split_name: str, row_index: int, buffer: List[dict], progress: dict
) -> None:
    if buffer:
        _write_shard(output_dir, split_name, row_index - len(buffer), buffer)
    progress[split_name] = row_index
    _save_checkpoint(output_dir, progress)


def _process_split(
    split_name: str,
    source_label: str,
    pipe: Any,
    cfg: SketchAugmentationConfig,
    base_seed: int,
    output_dir: Path,
    progress: dict,
    checkpoint_interval: int,
    deadline: float,
) -> bool:
    """Processes `split_name` from wherever `progress` says it left off. Returns True if
    the split finished (exhausted the stream, or hit cfg.max_rows), False if it stopped
    early because `deadline` (time.monotonic()) was reached."""
    start_index = progress.get(split_name, 0)
    if cfg.max_rows is not None and start_index >= cfg.max_rows:
        return True

    ds = load_split(split_name, streaming=True)
    window = itertools.islice(ds, start_index, cfg.max_rows)

    sketch_probability = 1.0 if cfg.sketch_always_populated else 0.5

    buffer: List[dict] = []
    row_index = start_index
    for row in window:
        if time.monotonic() >= deadline:
            _flush_and_checkpoint(output_dir, split_name, row_index, buffer, progress)
            return False

        image = row[IMAGE_COLUMN]
        if not hasattr(image, "convert"):
            from PIL import Image as PILImage

            image = PILImage.open(image)
        image = image.convert("RGB")

        code = row[code_column(split_name)]
        description = row.get("description")
        source = row.get("source") or source_label

        seed = base_seed + row_index
        sketch, method = generate_synthetic_sketch(
            image,
            seed=seed,
            sketch_probability=sketch_probability,
            ultrasketch_probability=cfg.ultrasketch_probability,
            displacement_alpha=cfg.displacement_alpha,
            displacement_sigma=cfg.displacement_sigma,
            pipe=pipe,
        )

        buffer.append(
            {
                "image": image,
                "sketch": sketch,
                "code": code,
                "description": description,
                "source": source,
                "sketch_method": method,
            }
        )
        row_index += 1

        if len(buffer) >= checkpoint_interval:
            _flush_and_checkpoint(output_dir, split_name, row_index, buffer, progress)
            buffer = []

    if buffer:
        _flush_and_checkpoint(output_dir, split_name, row_index, buffer, progress)
    return True


def build_sketch_dataset_incremental(
    splits_and_labels: List[Tuple[str, str]],
    output_dir: str | Path,
    cfg: Optional[SketchAugmentationConfig] = None,
    base_seed: int = 3407,
    checkpoint_interval: int = 100,
    time_limit_hours: float = 7.5,
) -> bool:
    """Runs (or resumes) offline sketch-column generation for each split in order.
    Returns True if every split finished; False if the time limit was hit first, in
    which case re-calling this with the same `output_dir` resumes from
    `<output_dir>/checkpoint.json`."""
    cfg = cfg or SketchAugmentationConfig()
    output_dir = Path(output_dir)
    progress = _load_checkpoint(output_dir)
    deadline = time.monotonic() + time_limit_hours * 3600

    pipe = load_ultrasketch_pipeline()
    try:
        for split_name, source_label in splits_and_labels:
            finished = _process_split(
                split_name, source_label, pipe, cfg, base_seed, output_dir, progress,
                checkpoint_interval, deadline,
            )
            if not finished:
                return False
    finally:
        release_ultrasketch_pipeline(pipe)

    (output_dir / "DONE").write_text("all splits complete\n", encoding="utf-8")
    return True


def _resolve_self_resubmit_command(
    sbatch_script: Optional[str], job_id: Optional[str], resume_args: List[str]
) -> Optional[List[str]]:
    """Only resolves to a command when SLURM_JOB_ID is set, i.e. this is running inside
    a submitted SLURM job and not when run interactively. `resume_args` (the --split/
    --source-label/--output-dir/etc. this run was given) is re-passed explicitly since
    nothing about them is recoverable from disk."""
    if not job_id:
        return None
    script = sbatch_script or str(Path(__file__).with_name("build_sketch_dataset.sbatch"))
    return ["sbatch", "--dependency=afterany:" + job_id, script, *resume_args]


def _maybe_self_resubmit(sbatch_script: Optional[str], resume_args: List[str]) -> Optional[str]:
    command = _resolve_self_resubmit_command(sbatch_script, os.environ.get("SLURM_JOB_ID"), resume_args)
    if command is None:
        return None
    subprocess.run(command, check=False)
    return " ".join(command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", action="append", required=True, dest="splits",
        help="a split name from dataset_loader.SPLITS; repeat for multiple splits",
    )
    parser.add_argument(
        "--source-label", action="append", required=True, dest="source_labels",
        help="source label for the --split at the same position (e.g. datikzv4, geotikz); "
             "used unless the row already has its own 'source' value",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--base-seed", type=int, default=3407)
    parser.add_argument(
        "--checkpoint-interval", type=int, default=100,
        help="rows to buffer before flushing a parquet shard + updating checkpoint.json",
    )
    parser.add_argument(
        "--time-limit-hours", type=float, default=7.5,
        help="stop (and self-resubmit) before this many hours elapse, to leave headroom under an 8h SLURM job",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="cap rows PER SPLIT - final size isn't decided yet; leave unset to process a whole split, "
             "or pass e.g. 50-100 for a first correctness pass",
    )
    parser.add_argument(
        "--ultrasketch-probability", type=float, default=None,
        help="override SketchAugmentationConfig.ultrasketch_probability (default 0.5)",
    )
    parser.add_argument(
        "--no-self-resubmit", action="store_true",
        help="don't sbatch a follow-up job if the time limit is hit; just stop (checkpoint is still saved)",
    )
    parser.add_argument("--sbatch-script", type=str, default=None)
    args = parser.parse_args()

    if len(args.splits) != len(args.source_labels):
        raise ValueError(
            "--split and --source-label must be passed the same number of times, in "
            f"matching order (got {len(args.splits)} splits, {len(args.source_labels)} labels)"
        )

    cfg = SketchAugmentationConfig(max_rows=args.max_rows)
    if args.ultrasketch_probability is not None:
        cfg.ultrasketch_probability = args.ultrasketch_probability

    splits_and_labels = list(zip(args.splits, args.source_labels))
    finished = build_sketch_dataset_incremental(
        splits_and_labels,
        args.output_dir,
        cfg=cfg,
        base_seed=args.base_seed,
        checkpoint_interval=args.checkpoint_interval,
        time_limit_hours=args.time_limit_hours,
    )

    if finished:
        print(f"All splits complete - {args.output_dir}/DONE written.")
        print(f"Shards under {args.output_dir}/shards/. Nothing was pushed - review, then push yourself, e.g.:")
        print("  from datasets import load_dataset")
        print(f"  ds = load_dataset('parquet', data_files='{args.output_dir}/shards/*.parquet', split='train')")
        print("  ds.push_to_hub('your-username/your-repo')")
    else:
        print(f"Time limit reached before finishing - checkpoint saved under {args.output_dir}/checkpoint.json.")
        if args.no_self_resubmit:
            print("--no-self-resubmit set: not resubmitting. Re-run the same command to resume.")
        else:
            resume_args: List[str] = []
            for split_name, source_label in zip(args.splits, args.source_labels):
                resume_args += ["--split", split_name, "--source-label", source_label]
            resume_args += ["--output-dir", args.output_dir, "--base-seed", str(args.base_seed)]
            resume_args += ["--checkpoint-interval", str(args.checkpoint_interval)]
            resume_args += ["--time-limit-hours", str(args.time_limit_hours)]
            if args.max_rows is not None:
                resume_args += ["--max-rows", str(args.max_rows)]
            if args.ultrasketch_probability is not None:
                resume_args += ["--ultrasketch-probability", str(args.ultrasketch_probability)]

            command = _maybe_self_resubmit(args.sbatch_script, resume_args)
            if command:
                print(f"Self-resubmitted: {command}")
            else:
                print(
                    "No SLURM_JOB_ID in environment - not self-resubmitting (not running inside a "
                    "submitted job). Re-run the same command manually to resume."
                )


if __name__ == "__main__":
    main()
