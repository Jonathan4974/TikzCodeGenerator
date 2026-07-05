"""Sketch-agent SDXL+ControlNet+LoRA training pipeline for the sketch-to-clean-image task."""

from .config import SketchAgentConfig, build_training_config
from .data import (
    SketchAgentDataset,
    SyntheticPair,
    generate_synthetic_pairs,
    iter_datikz_renders,
    load_synthetic_dataset,
)
from .eval import pixel_congruence_coefficient
from .model_loader import SketchAgentModelLoader, SketchAgentModels
from .real_data import SketchFigSplit, load_sketchfig_dataset
from .trainer import SketchAgentTrainer

__all__ = [
    "SketchAgentConfig",
    "build_training_config",
    "SketchAgentDataset",
    "SyntheticPair",
    "generate_synthetic_pairs",
    "iter_datikz_renders",
    "load_synthetic_dataset",
    "pixel_congruence_coefficient",
    "SketchAgentModelLoader",
    "SketchAgentModels",
    "SketchFigSplit",
    "load_sketchfig_dataset",
    "SketchAgentTrainer",
]
