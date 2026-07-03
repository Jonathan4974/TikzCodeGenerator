import os
import torch

from .config import TrainingConfig
from .data import DaTikZDatasetBuilder
from .model_loader import GemmaVisionModelLoader
from .trainer import GemmaGRPOTrainer


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cfg = TrainingConfig()
    torch.manual_seed(cfg.seed)

    dataset_builder = DaTikZDatasetBuilder(cfg)
    train_dataset = dataset_builder.load()

    model_loader = GemmaVisionModelLoader(cfg)
    model, tokenizer = model_loader.load()

    trainer = GemmaGRPOTrainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()