"""Config for build_sketch_dataset.py: generates one method-dedicated split at a time for
`loss-boss/tikz-train` (see that module's docstring for the full pipeline)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SketchAugmentationConfig:
    displacement_alpha: float = 8.0
    displacement_sigma: float = 3.0

    max_rows: Optional[int] = None
