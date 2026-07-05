"""Sketch-agent training basic implementation for the sketch-to-clean-image pipeline."""

from .config import SketchAgentConfig, build_training_config
from .data import (
    SyntheticPair,
    build_synthetic_dataset,
    generate_synthetic_pairs,
    iter_datikz_renders,
    iter_fake_renders,
    load_synthetic_dataset,
)
from .eval import (
    dreamsim_similarity,
    evaluate_generated_outputs,
    pixel_congruence_coefficient,
    siglip_similarity,
)
from .real_data import SketchFigSplit, load_real_pairs, load_sketchfig_dataset
from .train import run_training

__all__ = [
    "SketchAgentConfig",
    "build_training_config",
    "SyntheticPair",
    "build_synthetic_dataset",
    "generate_synthetic_pairs",
    "iter_datikz_renders",
    "iter_fake_renders",
    "load_synthetic_dataset",
    "evaluate_generated_outputs",
    "pixel_congruence_coefficient",
    "siglip_similarity",
    "dreamsim_similarity",
    "load_real_pairs",
    "load_sketchfig_dataset",
    "SketchFigSplit",
    "run_training",
]
