from __future__ import annotations

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import torch
from trl import SFTConfig, SFTTrainer


def train_sft(cfg, model, processor, train_dataset, eval_dataset) -> None:
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = SFTConfig(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps,
        warmup_steps=cfg.warmup_steps,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        remove_unused_columns=False,
        dataset_text_field="",
        max_length=None,
        completion_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="tensorboard",
        logging_dir=str(cfg.output_dir / "logs"),
    )

    FastVisionModel.for_training(model)
    collator = UnslothVisionDataCollator(
        model,
        processor,
        max_seq_length=cfg.max_seq_length,
        resize=cfg.image_resize,
        completion_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        processing_class=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(str(cfg.lora_output_dir))
    processor.save_pretrained(str(cfg.lora_output_dir))
