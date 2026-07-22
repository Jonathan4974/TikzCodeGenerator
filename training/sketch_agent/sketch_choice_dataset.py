"""Train-time clean/sketch selection. Runs inside the Structure/Main model/Text
Agent's training loop.

Wraps the image/sketch/code/description/source dataset from
build_sketch_dataset.py. On each access, draws whether to feed the clean `image` or the
precomputed `sketch`.
"""
from __future__ import annotations

import random
from typing import Any, Optional


class SketchChoiceDataset:
    """Wraps a dataset with image/sketch/code/description/source columns (a
    `datasets.Dataset`, or a plain list of dicts) and resolves the clean/sketch choice
    per access.
    """

    def __init__(
        self,
        dataset: Any,
        train_time_sketch_probability: float = 0.5,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.dataset = dataset
        self.train_time_sketch_probability = train_time_sketch_probability
        self._rng = rng or random.Random()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset[idx]
        use_sketch = self._rng.random() < self.train_time_sketch_probability
        input_image = row["sketch"] if use_sketch else row["image"]
        return {
            "input_image": input_image,
            "code": row["code"],
            "description": row.get("description"),
            "source": row["source"],
            "used_sketch": use_sketch, # not sure if will want this col
        }
