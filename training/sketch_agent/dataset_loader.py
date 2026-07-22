"""Loading `loss-boss/tikz-dataset` splits.

Dataset: cleaned image-TikZ code pairs merging DaTikZ-V4 and GeoTikz-Base
and collected samples. Four splits:
  - datikz_v4: cleaned samples from DaTikZ-V4
  - geotikz_bridge_base: cleaned, re-rendered samples from GeoTikz-Base
  - our_dataset_train: additionally collected training samples
  - our_dataset_benchmark: held-out samples for evaluation

Each split is loaded independently, never as one uniform multi-split load:
The dataset viewer errors with `FileFormatMismatchBetweenSplitsError`
across splits, meaning they don't share one file layout.
"""
from __future__ import annotations

import itertools
from typing import Iterator, Optional, Tuple

HF_REPO_ID = "loss-boss/tikz-dataset"

SPLITS = ["datikz_v4", "geotikz_bridge_base", "our_dataset_train", "our_dataset_benchmark"]

IMAGE_COLUMN = "image"

# Code column name per split. GeoTikz's code lives in "response": already cleaned of
# Markdown code fences upstream, so it's raw LaTeX/TikZ source, same as "code" elsewhere.
CODE_COLUMN = {
    "geotikz_bridge_base": "response",
}
DEFAULT_CODE_COLUMN = "code"


def code_column(split_name: str) -> str:
    return CODE_COLUMN.get(split_name, DEFAULT_CODE_COLUMN)


def load_split(
    split_name: str,
    streaming: bool = True,
    shuffle_seed: Optional[int] = None,
    shuffle_buffer_size: int = 10_000,
):
    """Loads one split of loss-boss/tikz-dataset.

    Tries the plain `datasets.load_dataset(repo, split)` form first, falls back to the
    raw `hf://datasets/<repo>/<split>_part-*.parquet` glob form (the shape
    `data/data-prep-tikz/tikz_clean_pipeline/storage.py` writes) if that errors.
    """
    from datasets import load_dataset

    if split_name not in SPLITS:
        raise ValueError(f"unknown split {split_name!r}, expected one of {SPLITS}")

    try:
        ds = load_dataset(HF_REPO_ID, split_name, streaming=streaming)
    except Exception:
        pattern = f"hf://datasets/{HF_REPO_ID}/{split_name}_part-*.parquet"
        ds = load_dataset(
            "parquet", data_files={split_name: pattern}, split=split_name, streaming=streaming
        )

    if streaming and shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)

    return ds


def iter_clean_images(
    split_name: str,
    num_samples: int,
    seed: Optional[int] = None,
    start_index: int = 0,
    shuffle_buffer_size: int = 10000,
) -> Iterator[Tuple[str, "Image.Image"]]:
    """Yields (id, clean_image) pairs from `split_name`."""
    ds = load_split(split_name, streaming=True, shuffle_seed=seed, shuffle_buffer_size=shuffle_buffer_size)
    window = itertools.islice(ds, start_index, start_index + num_samples)
    for row_index, row in enumerate(window):
        image = row[IMAGE_COLUMN]
        if not hasattr(image, "convert"):
            from PIL import Image as PILImage

            image = PILImage.open(image)
        row_id = row.get("file_id") or row.get("id") or f"{split_name}_{start_index + row_index}"
        yield row_id, image.convert("RGB")
