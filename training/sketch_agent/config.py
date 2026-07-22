"""Config for the two-stage sketch pipeline: build_sketch_dataset.py generates the
`sketch` column offline, sketch_choice_dataset.py picks clean-vs-sketch at train time."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SketchAugmentationConfig:
    # Offline: split between the two synthetic methods.
    ultrasketch_probability: float = 0.5
    displacement_alpha: float = 6.0
    displacement_sigma: float = 12.0
    resize_multiple: int = 16

    # Offline: whether every row gets a `sketch` (True), or some rows keep none (False,
    # restores the three-way clean/ultrasketch/displacement draw in one step).
    # Confirm this later
    sketch_always_populated: bool = True

    # Train-time: P(use the precomputed `sketch` column instead of `image`)
    train_time_sketch_probability: float = 0.5

    # Rows to offline-process per split. None = process the whole split.
    max_rows: Optional[int] = None

    hf_dataset_repo: str = "loss-boss/tikz-dataset"
    streaming: bool = True
    shuffle_buffer_size: int = 10000