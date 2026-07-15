"""Overfit-to-1-sample: trains on one (sketch, clean_render) pair
and evaluates on that same pair.

Usage:
  conda activate sketch-agent
  python -m training.sketch_agent.overfit_run --max-steps 600
  python -m training.sketch_agent.overfit_run --max-steps 1200   # resumes from checkpoint

Then:
  tensorboard --logdir training/sketch_agent/output_overfit/tensorboard
  training/sketch_agent/output_overfit/eval_previews/step_*/0.png
"""
from __future__ import annotations

import argparse

from .config import build_training_config, seed_everything
from .data import SketchAgentDataset
from .model_loader import SketchAgentModelLoader
from .trainer import SketchAgentTrainer

OVERFIT_DIR = "training/sketch_agent/output_overfit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=50)
    args = parser.parse_args()

    cfg = build_training_config(
        {
            "output_dir": OVERFIT_DIR,
            "checkpoint_dir": f"{OVERFIT_DIR}/checkpoints",
            "lora_output_dir": f"{OVERFIT_DIR}/lora",
            "synthetic_dir": f"{OVERFIT_DIR}/synthetic_pairs",
            "sketchfig_cache_dir": f"{OVERFIT_DIR}/sketchfig_cache",
            "max_steps": args.max_steps,
            "checkpoint_interval_steps": args.checkpoint_interval_steps,
            "synthetic_samples": 1,
            "use_sketchfig": False,
            "eval_sample_size": 1,
            "self_resubmit": False,
        }
    )
    seed_everything(cfg.seed)

    train_dataset = SketchAgentDataset(cfg)
    assert len(train_dataset.pairs) == 1, f"expected exactly 1 pair, got {len(train_dataset.pairs)}"

    models = SketchAgentModelLoader(cfg).load()
    trainer = SketchAgentTrainer(
        cfg=cfg,
        models=models,
        train_dataset=train_dataset,
        eval_pairs=train_dataset.pairs,  # eval on the exact same sample we're training on
    )
    trainer.train()
    print(f"Overfit run finished at max_steps={args.max_steps}, output under {OVERFIT_DIR}")


if __name__ == "__main__":
    main()
