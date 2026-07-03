import os
import torch

from config import TrainingConfig
from data import DaTikZDataset
from model_loader import GemmaVisionModelLoader
from trainer import GemmaSFTTrainer


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cfg = TrainingConfig()
    torch.manual_seed(cfg.seed)

    train_dataset = DaTikZDataset(cfg)

    model_loader = GemmaVisionModelLoader(cfg)
    model, tokenizer = model_loader.load()

    trainer = GemmaSFTTrainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()