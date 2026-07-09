from dataclasses import dataclass
import os

@dataclass
class TrainingConfig:
    model_name: str = "/models/huggingface/hub/gemma-4-31B-it-unsloth-bnb-4bit"
    dataset_path: str = "/data"

    train_manifest: str = "manifest_train.csv"
    val_manifest: str = "manifest_val.csv"

    eval_steps: int = 5

    image_column: str = "image_path"
    code_column: str = "code_path"
    vlm_description_column: str = "vlm_description_path"

    output_dir: str = "/models/gemma4-sft/gemma4_sft"
    lora_output_dir: str = "/models/gemma4-sft/gemma4_sft_lora"

    max_seq_length: int = 512
    image_size: int = int(os.getenv("REF_IMAGE_SIZE", "512"))

    lora_rank: int = 8
    seed: int = 3407

    num_examples_train: int | None = 10
    num_examples_val: int | None = 5

    learning_rate: float = 2e-4
    epochs: int = 10
    max_steps: int = -1
    save_steps: int = 1000

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 10