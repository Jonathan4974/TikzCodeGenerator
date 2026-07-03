from dataclasses import dataclass


@dataclass
class TrainingConfig:
    model_name: str = "/home/jonas/.cache/huggingface/hub/gemma-4-31B-it-unsloth-bnb-4bit"
    dataset_path: str = "/home/jonas/Datasets/TikZ/train-big"

    image_column: str = "image_path"
    code_column: str = "code_path"
    vlm_description_column: str = "vlm_description_path"

    output_dir: str = "gemma4_grpo"
    lora_output_dir: str = "gemma4_grpo_lora"

    max_seq_length: int = 16384
    image_size: int = 448

    lora_rank: int = 16
    seed: int = 3407

    num_examples: int | None = 10

    learning_rate: float = 2e-4
    max_steps: int = 60
    save_steps: int = 60

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    