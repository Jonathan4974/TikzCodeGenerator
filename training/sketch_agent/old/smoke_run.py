"""Local smoke-test: runs the real training loop end-to-end at a tiny scale,
fully isolated so it doesn't touch the real run's output/ artifacts.

Usage:
  conda activate sketch-agent
  python -m training.sketch_agent.smoke_run --max-steps 12
  python -m training.sketch_agent.smoke_run --max-steps 20   # resumes from the step-12 checkpoint
  python -m training.sketch_agent.smoke_run --image-size 1024 --checkpoint-interval-steps 4 --max-steps 12
  etc
"""
from __future__ import annotations

import argparse

import torch

from .config import build_training_config, seed_everything
from .data import SketchAgentDataset
from .model_loader import SketchAgentModelLoader
from .real_data import load_sketchfig_dataset
from .trainer import SketchAgentTrainer

SMOKE_DIR = "training/sketch_agent/output_smoke"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    args = parser.parse_args()

    overrides = {
        "output_dir": SMOKE_DIR,
        "checkpoint_dir": f"{SMOKE_DIR}/checkpoints",
        "lora_output_dir": f"{SMOKE_DIR}/lora",
        "synthetic_dir": f"{SMOKE_DIR}/synthetic_pairs",
        "sketchfig_cache_dir": f"{SMOKE_DIR}/sketchfig_cache",
        "max_steps": args.max_steps,
        "checkpoint_interval_steps": args.checkpoint_interval_steps,
        "synthetic_samples": 6,
        "eval_sample_size": 2,
        "self_resubmit": False,
    }
    if args.image_size is not None:
        overrides["image_size"] = args.image_size
    cfg = build_training_config(overrides)
    seed_everything(cfg.seed)
    torch.cuda.reset_peak_memory_stats()

    sketchfig = (
        load_sketchfig_dataset(
            cfg.sketchfig_cache_dir,
            train_fraction=cfg.sketchfig_train_fraction,
            seed=cfg.sketchfig_split_seed,
            dataset_name=cfg.sketchfig_dataset_name,
        )
        if cfg.use_sketchfig
        else None
    )

    train_dataset = SketchAgentDataset(cfg, extra_pairs=sketchfig.train if sketchfig else [])
    models = SketchAgentModelLoader(cfg).load()
    trainer = SketchAgentTrainer(
        cfg=cfg,
        models=models,
        train_dataset=train_dataset,
        eval_pairs=sketchfig.eval if sketchfig else [],
    )
    trainer.train()
    print(f"Smoke run finished at max_steps={args.max_steps}, image_size={cfg.image_size}, output under {SMOKE_DIR}")
    print(f"PEAK VRAM (allocated): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"PEAK VRAM (reserved):  {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
