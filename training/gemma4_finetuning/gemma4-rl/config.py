from dataclasses import dataclass


@dataclass
class TrainingConfig:
    model_name: str = "/home/jonas/.cache/huggingface/hub/gemma-4-31B-it-unsloth-bnb-4bit"
    dataset_path: str = "/home/jonas/.cache/huggingface/datasets/nllg___da_tik_z-v4"

    output_dir: str = "gemma4_datikz_grpo"
    lora_output_dir: str = "gemma4_datikz_grpo_lora"

    max_seq_length: int = 16384
    image_size: int = 448

    lora_rank: int = 16
    seed: int = 3407

    # Zum Testen klein lassen
    num_examples: int | None = 10

    max_steps: int = 60
    save_steps: int = 60

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    num_generations: int = 2

    max_prompt_length: int = 512
    max_completion_length: int = 1024