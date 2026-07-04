from unsloth import FastVisionModel


class GemmaVisionModelLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def load(self):
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=self.cfg.model_name,
            max_seq_length=self.cfg.max_seq_length,
            load_in_4bit=True,
            fast_inference=False,
        )

        already_has_lora = hasattr(model, "peft_config") and len(model.peft_config) > 0

        if already_has_lora:
            print("Loaded SFT LoRA adapter. Continuing RL training from existing adapter.")
            return model, tokenizer

        print("Loaded base model. Adding new LoRA adapter.")

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,

            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_rank,
            lora_dropout=0,
            bias="none",
            random_state=self.cfg.seed,
            use_rslora=False,
            loftq_config=None,
            use_gradient_checkpointing="unsloth",
        )

        return model, tokenizer