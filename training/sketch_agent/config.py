from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple


@dataclass
class SketchAgentConfig:
    """Configuration for the sketch-agent training basic implementation"""

    output_dir: str = "training/sketch_agent/output"
    checkpoint_dir: str = "training/sketch_agent/output/checkpoints"
    synthetic_dir: str = "training/sketch_agent/output/synthetic_pairs"
    checkpoint_interval_steps: int = 2
    max_steps: int = 8
    batch_size: int = 1
    image_size: int = 256
    seed: int = 3407
    synthetic_samples: int = 8
    dry_run: bool = True
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_model: str = "diffusers/controlnet-canny-sdxl-1.0"
    controlnet_alt_model: str = "xinsir/controlnet-scribble-sdxl-1.0"
    ultrasketch_probability: float = 0.5
    displacement_alpha: float = 6.0
    displacement_sigma: float = 12.0
    datikz_dataset_name: str = "nllg/DaTikZ-V4"
    datikz_split: str = "train"
    datikz_streaming: bool = True
    use_sketchfig: bool = False
    sketchfig_cache_dir: str = "training/sketch_agent/output/sketchfig_cache"
    sketchfig_train_fraction: float = 0.0
    sketchfig_split_seed: int = 3407
    eval_metrics: Tuple[str, ...] = ("pixel_cc",)
    self_resubmit: bool = False
    sbatch_script: Optional[str] = None
    real_data_dir: Optional[str] = None
    use_real_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_training_config(overrides: Optional[dict[str, Any]] = None) -> SketchAgentConfig:
    config = SketchAgentConfig()
    if overrides:
        for key, value in overrides.items():
            setattr(config, key, value)
    return config
