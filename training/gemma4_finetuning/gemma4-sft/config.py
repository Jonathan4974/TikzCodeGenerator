from dataclasses import dataclass
import os

@dataclass
class TrainingConfig:
    model_name: str = "/models/huggingface/hub/gemma-4-31B-it-unsloth-bnb-4bit"
    dataset_path: str = "/data"

    image_column: str = "image_path"
    code_column: str = "code_path"
    vlm_description_column: str = "vlm_description_path"

    output_dir: str = "/models/gemma4-sft/gemma4_grpo"
    lora_output_dir: str = "/models/gemma4-sft/gemma4_grpo_lora"

    max_seq_length: int = 4096
    image_size: int = int(os.getenv("REF_IMAGE_SIZE", "512"))


    lora_rank: int = 8
    seed: int = 3407

    num_examples: int | None = 10

    learning_rate: float = 2e-4
    epochs: int = 100
    max_steps: int = -1
    save_steps: int = 60

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    