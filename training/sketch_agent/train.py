from __future__ import annotations

import argparse

import torch

from .config import build_training_config
from .data import SketchAgentDataset
from .model_loader import SketchAgentModelLoader
from .real_data import load_sketchfig_dataset
from .trainer import SketchAgentTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit-hours", type=float, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.time_limit_hours is not None:
        overrides["time_limit_hours"] = args.time_limit_hours
    cfg = build_training_config(overrides)
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


if __name__ == "__main__":
    main()
