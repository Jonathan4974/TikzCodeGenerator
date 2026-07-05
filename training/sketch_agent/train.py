from __future__ import annotations

import torch

from .config import SketchAgentConfig
from .data import SketchAgentDataset
from .model_loader import SketchAgentModelLoader
from .real_data import load_sketchfig_dataset
from .trainer import SketchAgentTrainer


def main() -> None:
    cfg = SketchAgentConfig()
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
