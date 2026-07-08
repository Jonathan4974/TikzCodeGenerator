"""Local smoke-test: runs the real training loop end-to-end at a tiny scale,
fully isolated so it doesn't touch the real run's output/ artifacts.

Usage:
  conda activate sketch-agent
  python -m training.sketch_agent.smoke_run --max-steps 12
  python -m training.sketch_agent.smoke_run --max-steps 20   # resumes from the step-12 checkpoint
  etc
"""
from __future__ import annotations

import argparse

import torch

from .config import build_training_config
from .data import SketchAgentDataset
from .model_loader import SketchAgentModelLoader
from .real_data import load_sketchfig_dataset
from .trainer import SketchAgentTrainer

SMOKE_DIR = "training/sketch_agent/output_smoke"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=4)
    args = parser.parse_args()

    cfg = build_training_config(
        {
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
    )
    torch.manual_seed(cfg.seed)

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
    print(f"Smoke run finished at max_steps={args.max_steps}, output under {SMOKE_DIR}")


if __name__ == "__main__":
    main()
