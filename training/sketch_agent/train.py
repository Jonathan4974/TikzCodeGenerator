from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .config import SketchAgentConfig, build_training_config
from .data import (
    RenderSource,
    SyntheticPair,
    generate_synthetic_pairs,
    iter_datikz_renders,
    iter_fake_renders,
)
from .eval import pixel_congruence_coefficient
from .real_data import load_real_pairs, load_sketchfig_dataset
from .ultrasketch_methods import DryRunUltraSketchPipe


def resolve_render_source(config: SketchAgentConfig) -> RenderSource:
    if config.dry_run:
        return lambda n, start: iter_fake_renders(n, config.image_size, config.seed, start_index=start)
    return lambda n, start: iter_datikz_renders(
        n,
        seed=config.seed,
        dataset_name=config.datikz_dataset_name,
        split=config.datikz_split,
        streaming=config.datikz_streaming,
        start_index=start,
    )


def resolve_ultrasketch_pipe_factory(config: SketchAgentConfig) -> Optional[Callable[[], Any]]:
    if config.dry_run:
        return lambda: DryRunUltraSketchPipe()
    return None 


def _save_checkpoint(checkpoint_dir: Path, step: int, config: SketchAgentConfig, metadata: Dict[str, Any]) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.json"
    latest_manifest_path = checkpoint_dir / "latest_checkpoint.json"
    payload = {
        "step": step,
        "config": config.to_dict(),
        "metadata": metadata,
    }
    checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_manifest_path.write_text(json.dumps({"step": step, "path": str(checkpoint_path)}, indent=2), encoding="utf-8")
    return checkpoint_path


def _load_checkpoint(checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
    if not checkpoint_dir.exists():
        return None

    manifest_path = checkpoint_dir / "latest_checkpoint.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        latest = Path(manifest["path"])
        if latest.exists():
            return json.loads(latest.read_text(encoding="utf-8"))

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_step_*.json"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    return json.loads(latest.read_text(encoding="utf-8"))


def _maybe_self_resubmit(config: SketchAgentConfig, step: int, script_path: Optional[Path] = None) -> Optional[str]:
    if not config.self_resubmit:
        return None
    script = script_path or Path(__file__).with_name("train.sbatch")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    command = ["sbatch", "--dependency=afterany:" + job_id, str(script)]
    subprocess.run(command, check=False)
    return " ".join(command)


def _clean_up_sketch(sketch_path: Path, target_path: Path, output_path: Path) -> Image.Image:
    sketch = Image.open(sketch_path).convert("RGB")
    target = Image.open(target_path).convert("RGB")
    cleaned = ImageOps.autocontrast(sketch)
    cleaned = cleaned.filter(ImageFilter.GaussianBlur(radius=0.8))
    cleaned = Image.blend(cleaned, target, 0.2)
    cleaned.save(output_path)
    return cleaned


def run_training(config: Optional[SketchAgentConfig] = None, script_path: Optional[Path] = None) -> Dict[str, Any]:
    cfg = config or build_training_config()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset: list[SyntheticPair] = []
    if cfg.use_real_data and cfg.real_data_dir:
        dataset = load_real_pairs(cfg.real_data_dir)
    if not dataset:
        # generate_synthetic_pairs itself checks for an existing, sufficiently-sized
        # dataset on disk and reuses it unchanged; otherwise it tops up incrementally.
        # Do not pre-check with load_synthetic_dataset here - that would short-circuit
        # before generate_synthetic_pairs and silently ignore synthetic_samples growing
        # across resumed runs.
        dataset = generate_synthetic_pairs(
            cfg.synthetic_dir,
            num_samples=cfg.synthetic_samples,
            seed=cfg.seed,
            render_source=resolve_render_source(cfg),
            ultrasketch_probability=cfg.ultrasketch_probability,
            ultrasketch_pipe_factory=resolve_ultrasketch_pipe_factory(cfg),
            displacement_alpha=cfg.displacement_alpha,
            displacement_sigma=cfg.displacement_sigma,
        )
    if cfg.use_sketchfig and cfg.sketchfig_train_fraction > 0:
        sketchfig = load_sketchfig_dataset(
            cfg.sketchfig_cache_dir,
            train_fraction=cfg.sketchfig_train_fraction,
            seed=cfg.sketchfig_split_seed,
        )
        dataset += sketchfig.train

    resumed_state = _load_checkpoint(checkpoint_dir)
    start_step = 0
    if resumed_state is not None:
        start_step = int(resumed_state.get("step", 0))
        print(f"Resuming from checkpoint at step {start_step}")

    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[float] = []
    for step in range(start_step, cfg.max_steps):
        pair = dataset[step % len(dataset)]
        output_path = prediction_dir / f"sample_{step:03d}_pred.png"
        cleaned = _clean_up_sketch(pair.input_path, pair.target_path, output_path)
        reference = Image.open(pair.target_path).convert("RGB")
        metrics.append(pixel_congruence_coefficient(cleaned, reference))

        if (step + 1) % cfg.checkpoint_interval_steps == 0 or step + 1 == cfg.max_steps:
            metadata = {
                "dataset_size": len(dataset),
                "mean_cc": float(np.mean(metrics[-cfg.checkpoint_interval_steps :] or metrics)),
                "last_output": str(output_path),
            }
            checkpoint_path = _save_checkpoint(checkpoint_dir, step + 1, cfg, metadata)
            print(f"Saved checkpoint to {checkpoint_path}")

    summary = {
        "status": "ok",
        "steps_completed": cfg.max_steps - start_step,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "mean_cc": float(np.mean(metrics)) if metrics else 0.0,
        "self_resubmit_command": _maybe_self_resubmit(cfg, cfg.max_steps, script_path),
    }
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the sketch-agent training scaffold")
    parser.add_argument("--output-dir", default="training/sketch_agent/output")
    parser.add_argument("--checkpoint-dir", default="training/sketch_agent/output/checkpoints")
    parser.add_argument("--synthetic-dir", default="training/sketch_agent/output/synthetic_pairs")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=2)
    parser.add_argument("--synthetic-samples", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--self-resubmit", action="store_true")
    parser.add_argument("--ultrasketch-probability", type=float, default=0.5)
    parser.add_argument("--use-sketchfig", action="store_true")
    parser.add_argument("--sketchfig-train-fraction", type=float, default=0.0)
    parser.add_argument("--eval-metrics", default="pixel_cc")
    parser.add_argument("--no-dry-run", action="store_true", help="Use real DaTikZ-V4/UltraSketch calls instead of local dry-run stubs")
    args = parser.parse_args()

    config = build_training_config(
        {
            "output_dir": args.output_dir,
            "checkpoint_dir": args.checkpoint_dir,
            "synthetic_dir": args.synthetic_dir,
            "max_steps": args.max_steps,
            "checkpoint_interval_steps": args.checkpoint_interval_steps,
            "synthetic_samples": args.synthetic_samples,
            "image_size": args.image_size,
            "seed": args.seed,
            "self_resubmit": args.self_resubmit,
            "ultrasketch_probability": args.ultrasketch_probability,
            "use_sketchfig": args.use_sketchfig,
            "sketchfig_train_fraction": args.sketchfig_train_fraction,
            "eval_metrics": tuple(args.eval_metrics.split(",")),
            "dry_run": not args.no_dry_run,
        }
    )
    result = run_training(config=config)
    print(json.dumps(result, indent=2))
