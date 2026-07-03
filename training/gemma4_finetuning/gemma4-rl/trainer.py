from trl import GRPOConfig, GRPOTrainer
from unsloth import FastVisionModel

from rewards import TikZReward
from tb_callback import RewardTensorBoardCallback


class GemmaGRPOTrainer:
    def __init__(self, cfg, model, tokenizer, train_dataset, val_dataset=None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.reward_fn = TikZReward(self.cfg)

    def build_training_args(self):
        return GRPOConfig(
            output_dir=self.cfg.output_dir,

            learning_rate=self.cfg.learning_rate,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",

            logging_steps=1,
            log_completions=False,

            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            num_generations=self.cfg.num_generations,

            max_prompt_length=self.cfg.max_prompt_length,
            max_completion_length=self.cfg.max_completion_length,

            max_steps=self.cfg.max_steps,
            save_steps=self.cfg.save_steps,
            max_grad_norm=0.1,

            report_to="tensorboard",
            logging_dir=f"{self.cfg.output_dir}/logs",

            remove_unused_columns=False,

            importance_sampling_level="sequence",
            mask_truncated_completions=False,
            loss_type="dr_grpo",
        )

    def train(self):
        FastVisionModel.for_training(self.model)

        trainer = GRPOTrainer(
            model=self.model,
            args=self.build_training_args(),
            processing_class=self.tokenizer,
            reward_funcs=[self.reward_fn],
            train_dataset=self.train_dataset,
            callbacks=[
                RewardTensorBoardCallback(
                    reward_fn=self.reward_fn,
                    log_dir=f"{self.cfg.output_dir}/logs",
                    every_n_steps=self.cfg.log_reward_every,
                )
            ],
        )

        trainer.train()

        self.model.save_pretrained(self.cfg.lora_output_dir)
        self.tokenizer.save_pretrained(self.cfg.lora_output_dir)