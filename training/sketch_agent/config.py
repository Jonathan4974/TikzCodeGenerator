from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple


@dataclass
class SketchAgentConfig:
    """Configuration for the sketch-agent SDXL+ControlNet+LoRA training pipeline"""

    # paths
    output_dir: str = "training/sketch_agent/output"
    checkpoint_dir: str = "training/sketch_agent/output/checkpoints"
    lora_output_dir: str = "training/sketch_agent/output/lora"
    synthetic_dir: str = "training/sketch_agent/output/synthetic_pairs"
    sketchfig_cache_dir: str = "training/sketch_agent/output/sketchfig_cache"

    # models
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_model: str = "diffusers/controlnet-canny-sdxl-1.0"
    controlnet_alt_model: str = "xinsir/controlnet-scribble-sdxl-1.0"
    vae_model: str = "madebyollin/sdxl-vae-fp16-fix"

    # LoRA / optimization
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    lr_warmup_ratio: float = 0.1
    mixed_precision: str = "bf16"

    # training loop
    batch_size: int = 1
    gradient_accumulation_steps: int = 4  # real optimizer steps = max_steps / 4 = 1250
    image_size: int = 512
    # max_steps counts dataloader draws (batch_size=1), not accumulated optimizer steps, so it's
    # directly comparable to dataset size: 5000 / 1637 (1500 synthetic + 0.25*549 real) ~= 3 epochs
    max_steps: int = 5000
    checkpoint_interval_steps: int = 200
    time_limit_hours: float = 7.5
    dataloader_num_workers: int = 2
    seed: int = 3407

    # SDXL cross-attention prompt + ControlNet canny prep
    training_prompt: str = "a clean technical line drawing"
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200
    controlnet_conditioning_scale: float = 1.0

    # data sourcing
    use_synthetic_data: bool = True
    synthetic_samples: int = 1500
    ultrasketch_probability: float = 0.5
    displacement_alpha: float = 6.0
    displacement_sigma: float = 12.0
    datikz_dataset_name: str = "nllg/DaTikZ-V4"
    datikz_split: str = "train"
    datikz_streaming: bool = True
    use_sketchfig: bool = True
    sketchfig_dataset_name: str = "nllg/sketchfig"
    sketchfig_train_fraction: float = 0.25
    sketchfig_split_seed: int = 3407
    eval_sample_size: int = 6
    eval_metrics: Tuple[str, ...] = ("pixel_cc", "siglip", "dreamsim")

    # cluster / resume
    save_checkpoints: bool = True
    self_resubmit: bool = True
    sbatch_script: Optional[str] = None

    # observability
    run_name: Optional[str] = None  # TensorBoard run name; auto-generated (timestamp + job id) if unset

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_training_config(overrides: Optional[dict[str, Any]] = None) -> SketchAgentConfig:
    config = SketchAgentConfig()
    if overrides:
        for key, value in overrides.items():
            setattr(config, key, value)
    return config
