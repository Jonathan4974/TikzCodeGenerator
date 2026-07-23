"""Offline dataset builder: generates one method-dedicated split for `loss-boss/tikz-train`.

Usage:

    python -m training.sketch_agent.build_sketch_dataset \\
        --method displacement \\
        --output-dir training/sketch_agent/output_final/sketch_dataset_displacement \\
        --max-rows 100

    python -m training.sketch_agent.build_sketch_dataset \\
        --method ultrasketch \\
        --output-dir training/sketch_agent/output_final/sketch_dataset_ultrasketch \\
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
from typing import Any, List, Optional

from .config import SketchAugmentationConfig
from .dataset_loader import load_train_split, pick_variant
from .sketch_generation import (
    generate_synthetic_sketch,
    load_ultrasketch_pipeline,
    release_ultrasketch_pipeline,
)

METHODS = ("ultrasketch", "displacement")


def _build_features():
    """Lazy import: keeps this module importable without `datasets` installed."""
    from datasets import Features, Image as HFImage, Value

    return Features(
        {
            "image": HFImage(),  # the sketch
            "code": Value("string"),
            "description": Value("string"),
            "source_variant": Value("string"),  # "with_text" or "without_text"
            "sketch_method": Value("string"),  # "ultrasketch" or "displacement"
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


def _flush_and_checkpoint(output_dir: Path, split_name: str, row_index: int, buffer: List[dict], progress: dict) -> None:
    if buffer:
        _write_shard(output_dir, split_name, row_index - len(buffer), buffer)
    progress["row_index"] = row_index
    _save_checkpoint(output_dir, progress)


def _process_stream(
    method: str,
    split_name: str,
    pipe: Any,
    cfg: SketchAugmentationConfig,
    base_seed: int,
    output_dir: Path,
    progress: dict,
    checkpoint_interval: int,
    deadline: float,
) -> bool:
    """Processes the `train` split from wherever `progress` says it left off. Returns True if
    it finished (exhausted the stream, or hit cfg.max_rows), False if it stopped early because
    `deadline` (time.monotonic()) was reached."""
    start_index = progress.get("row_index", 0)
    if cfg.max_rows is not None and start_index >= cfg.max_rows:
        return True

    ds = load_train_split(streaming=True)
    window = itertools.islice(ds, start_index, cfg.max_rows)

    ultrasketch_probability = 1.0 if method == "ultrasketch" else 0.0

    buffer: List[dict] = []
    row_index = start_index
    for row in window:
        if time.monotonic() >= deadline:
            _flush_and_checkpoint(output_dir, split_name, row_index, buffer, progress)
            return False

        seed = base_seed + row_index
        variant, clean_image, code, description = pick_variant(row, seed=seed)
        if not hasattr(clean_image, "convert"):
            from PIL import Image as PILImage

            clean_image = PILImage.open(clean_image)
        clean_image = clean_image.convert("RGB")

        sketch, sketch_method = generate_synthetic_sketch(
            clean_image,
            seed=seed,
            sketch_probability=1.0,
            ultrasketch_probability=ultrasketch_probability,
            displacement_alpha=cfg.displacement_alpha,
            displacement_sigma=cfg.displacement_sigma,
            pipe=pipe,
        )

        buffer.append(
            {
                "image": sketch,
                "code": code,
                "description": description,
                "source_variant": variant,
                "sketch_method": sketch_method,
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
    method: str,
    split_name: str,
    output_dir: str | Path,
    cfg: Optional[SketchAugmentationConfig] = None,
    base_seed: int = 3407,
    checkpoint_interval: int = 100,
    time_limit_hours: float = 7.5,
) -> bool:
    """Runs (or resumes) offline sketch generation for one method-dedicated split. Returns
    True if it finished; False if the time limit was hit first, in which case re-calling this
    with the same `output_dir` resumes from `<output_dir>/checkpoint.json`."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")

    cfg = cfg or SketchAugmentationConfig()
    output_dir = Path(output_dir)
    progress = _load_checkpoint(output_dir)
    deadline = time.monotonic() + time_limit_hours * 3600

    pipe = load_ultrasketch_pipeline() if method == "ultrasketch" else None
    try:
        finished = _process_stream(
            method, split_name, pipe, cfg, base_seed, output_dir, progress, checkpoint_interval, deadline,
        )
    finally:
        if pipe is not None:
            release_ultrasketch_pipeline(pipe)

    if not finished:
        return False

    (output_dir / "DONE").write_text("finished\n", encoding="utf-8")
    return True


def assemble_dataset_dict(output_dir: str | Path, rename: Optional[dict[str, str]] = None):
    """Groups `<output_dir>/shards/*.parquet` by split and loads them into one
    `datasets.DatasetDict`, all sharing the same schema (`_build_features()`). Push this
    as a single `DatasetDict.push_to_hub(...)` call rather than pushing splits one at a
    time: that's what keeps every split in the same file format in the pushed repo.

    `rename` maps the split name inferred from shard filenames (the `--split-name` value the
    builder run used, e.g. `"ultrasketch"`) to a different key in the returned dict, if you
    want the pushed split named something else."""
    from datasets import DatasetDict, load_dataset

    rename = rename or {}
    shard_dir = Path(output_dir) / "shards"
    split_names = sorted({path.name.rsplit("_", 2)[0] for path in shard_dir.glob("*.parquet")})
    return DatasetDict(
        {
            rename.get(split_name, split_name): load_dataset(
                "parquet", data_files=str(shard_dir / f"{split_name}_*.parquet"), split="train"
            )
            for split_name in split_names
        }
    )


def _resolve_self_resubmit_command(
    sbatch_script: Optional[str], job_id: Optional[str], resume_args: List[str]
) -> Optional[List[str]]:
    """Only resolves to a command when SLURM_JOB_ID is set, i.e. this is running inside
    a submitted SLURM job and not when run interactively. `resume_args` (the --method/
    --output-dir/etc. this run was given) is re-passed explicitly since nothing about them
    is recoverable from disk."""
    if not job_id:
        return None
    script = sbatch_script or str(Path(__file__).with_name("build_sketch_dataset.sbatch"))
    return ["sbatch", "--dependency=afterany:" + job_id, script, *resume_args]


def _maybe_self_resubmit(sbatch_script: Optional[str], resume_args: List[str]) -> Optional[str]:
    command = _resolve_self_resubmit_command(sbatch_script, os.environ.get("SLURM_JOB_ID"), resume_args)
    if command is None:
        return None
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Self-resubmit command failed (exit {result.returncode}): {' '.join(command)}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return " ".join(command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True, help="which sketch method this run is dedicated to")
    parser.add_argument(
        "--split-name", type=str, default=None,
        help="push-time split name (also the shard-filename prefix); defaults to --method",
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
        help="cap total rows for this run",
    )
    parser.add_argument(
        "--no-self-resubmit", action="store_true",
        help="don't sbatch a follow-up job if the time limit is hit; just stop (checkpoint is still saved)",
    )
    parser.add_argument("--sbatch-script", type=str, default=None)
    args = parser.parse_args()

    split_name = args.split_name or args.method
    cfg = SketchAugmentationConfig(max_rows=args.max_rows)

    finished = build_sketch_dataset_incremental(
        args.method,
        split_name,
        args.output_dir,
        cfg=cfg,
        base_seed=args.base_seed,
        checkpoint_interval=args.checkpoint_interval,
        time_limit_hours=args.time_limit_hours,
    )

    if finished:
        print(f"Finished: {args.output_dir}/DONE written.")
        print(f"Shards under {args.output_dir}/shards/. Nothing was pushed: review, then push yourself, e.g.:")
        print("  from datasets import DatasetDict")
        print("  from training.sketch_agent.build_sketch_dataset import assemble_dataset_dict")
        print(f"  {split_name} = assemble_dataset_dict({args.output_dir!r})")
        print("  # ... assemble the OTHER method's split the same way, then in ONE call:")
        print(f"  DatasetDict({{**{split_name}, **other_split}}).push_to_hub('loss-boss/tikz-train')")
    else:
        print(f"Time limit reached before finishing: checkpoint saved under {args.output_dir}/checkpoint.json.")
        if args.no_self_resubmit:
            print("--no-self-resubmit set: not resubmitting. Re-run the same command to resume.")
        else:
            resume_args: List[str] = [
                "--method", args.method,
                "--split-name", split_name,
                "--output-dir", args.output_dir,
                "--base-seed", str(args.base_seed),
                "--checkpoint-interval", str(args.checkpoint_interval),
                "--time-limit-hours", str(args.time_limit_hours),
            ]
            if args.max_rows is not None:
                resume_args += ["--max-rows", str(args.max_rows)]
            if args.sbatch_script is not None:
                resume_args += ["--sbatch-script", args.sbatch_script]

            command = _maybe_self_resubmit(args.sbatch_script, resume_args)
            if command:
                print(f"Self-resubmitted: {command}")
            else:
                print(
                    "No SLURM_JOB_ID in environment. Re-run the same command manually to resume."
                )


if __name__ == "__main__":
    main()
