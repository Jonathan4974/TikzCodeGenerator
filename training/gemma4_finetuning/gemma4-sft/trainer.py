from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator


class GemmaSFTTrainer:
    def __init__(self, cfg, model, tokenizer, train_dataset):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset

    def build_args(self):
        return SFTConfig(
            output_dir=self.cfg.output_dir,

            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,

            learning_rate=self.cfg.learning_rate,
            num_train_epochs=self.cfg.epochs,
            max_steps=self.cfg.max_steps,
            save_steps=self.cfg.save_steps,

            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=5,

            seed=self.cfg.seed,

            remove_unused_columns=False,
            max_length=None,
            dataset_kwargs={"skip_prepare_dataset": True},

            report_to="tensorboard",
            logging_dir=f"{self.cfg.output_dir}/logs",
            logging_strategy="steps",
            logging_steps=1,
            logging_first_step=True,
        )

    def train(self):
        FastVisionModel.for_training(self.model)

        trainer = SFTTrainer(
            model=self.model,
            args=self.build_args(),
            processing_class=self.tokenizer,
            data_collator=UnslothVisionDataCollator(
                self.model,
                self.tokenizer,
                max_seq_length=self.cfg.max_seq_length,
                train_on_responses_only=True,
                instruction_part="<|turn>user\n",
                response_part="<|turn>model\n",
            ),
            train_dataset=self.train_dataset,
            eval_dataset=self.train_dataset,
        )

        trainer.train()

        self.model.save_pretrained(self.cfg.lora_output_dir)
        self.tokenizer.save_pretrained(self.cfg.lora_output_dir)