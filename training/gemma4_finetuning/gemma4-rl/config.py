from dataclasses import dataclass
import os

@dataclass
class TrainingConfig:
    model_name: str = "/models/gemma4-sft/gemma4_grpo_lora"
    dataset_path: str = "/data/"

    image_column: str = "image_path"
    code_column: str = "code_path"
    vlm_description_column: str = "vlm_description_path"

    output_dir: str = "/models/gemma4-rl/gemma4_grpo"
    lora_output_dir: str = "/models/gemma4-rl/gemma4_grpo_lora"
    
    max_seq_length: int = 4096
    image_size: int = int(os.getenv("REF_IMAGE_SIZE", "512"))

    lora_rank: int = 8
    seed: int = 3407

    # Zum Testen klein lassen
    num_examples: int | None = 10

    learning_rate: float = 5e-6
    max_steps: int = 60
    save_steps: int = 60

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    num_generations: int = 2

    max_prompt_length: int = 512
    max_completion_length: int = 4096


    #logging
    log_examples_every: int = 1
    log_examples_max: int = 2
    max_logged_code_chars: int = 8000




    # reward stuff
    crystalbleu_corpus_dir: str = "/data/code_corpus"
    crystalbleu_k: int = 500
    crystalbleu_n: int = 4
    crystalbleu_use_cache: bool = True

    crystalbleu_weight: float = 1.0
    ted_weight: float = 0.5
    ted_scale: float = 100.0

    visual_threshold: float = 0.8


    # reward_scores
    not_renderable_score = -2.0
    renderable_score = 1.0
    visual_reward_multiplier = 2.0
    
    # diagnostic reward calc
    error_multiplier = 0.10
    warning_multiplier = 0.05
    badboxes_multiplier = 0.01
    diagnostic_base_max_score = 1.0
    diagnostic_base_min_score = -2.0
    
    # visual reward calc
    siglip_multiplier = 0.15
    lpips_multiplier = 0.5
    dreamsim_multiplier = 0.35