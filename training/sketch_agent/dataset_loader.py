"""Loading `loss-boss/tikz-train`'s `train` split.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

HF_REPO_ID = "loss-boss/tikz-train"
SPLIT_NAME = "train"

# variant name -> (image column, code column, description column)
VARIANT_COLUMNS = {
    "with_text": ("image_with_text", "code_with_text", "llm_description_with_text"),
    "without_text": ("image_without_text_full", "code_without_text_full", "llm_description_without_text_full"),
}


def _list_parquet_files() -> list:
    """Sorted (for reproducible row order across resumed runs) list of every `.parquet`
    file backing `HF_REPO_ID`, as `hf://` paths `load_dataset("parquet", ...)` can stream."""
    from huggingface_hub import HfApi

    api = HfApi()
    return sorted(
        f"hf://datasets/{HF_REPO_ID}/{path}"
        for path in api.list_repo_files(HF_REPO_ID, repo_type="dataset")
        if path.endswith(".parquet")
    )


def load_train_split(
    streaming: bool = True,
    shuffle_seed: Optional[int] = None,
    shuffle_buffer_size: int = 10_000,
):
    """Loads `loss-boss/tikz-train`'s single `train` split.

    Deliberately does NOT use `load_dataset(HF_REPO_ID, split=SPLIT_NAME)`: that enforces one
    repo-wide "canonical" Features schema, and a real run crashed with `CastError` because
    `datikz_v4/datikz_v4-simple_llm_description_part-00000.parquet` doesn't even have
    `llm_description_with_text` in its schema, unlike most of the dataset's other shards - a
    genuine upstream schema inconsistency (flagged to Jonas separately), not fixable by
    changing what columns we ask for. Loading the same underlying parquet files directly
    through the generic `"parquet"` builder instead makes `datasets` union each file's own
    schema and pad whatever a given shard is missing with `None`, rather than enforcing one
    schema and raising the moment a shard doesn't match it exactly - confirmed against the
    real failure shape with synthetic heterogeneous-schema shards before writing this."""
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=_list_parquet_files(), split="train", streaming=streaming)
    if streaming and shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
    return ds


def pick_variant(row: dict, seed: int) -> Tuple[str, Any, str, Optional[str]]:
    """Randomly (50/50) picks `"with_text"` or `"without_text"` for one row and returns the
    CORRECTLY PAIRED (image, code, description) for that variant - never the image from one
    variant with another variant's code. `description` may be `None`: not every row has one,
    especially for the with-text variant: don't assume it's populated.

    The without-text variant is only populated for a small minority of rows (~10k of 410k,
    the "no text" augmentation slice - most rows have `image_without_text_full`/
    `code_without_text_full` as `None`, not just a missing description). If the drawn
    variant's image isn't populated for this row, falls back to the other variant instead
    of returning a `None` image."""
    import numpy as np

    variant = "with_text" if np.random.default_rng(seed).random() < 0.5 else "without_text"
    image_col, code_col, description_col = VARIANT_COLUMNS[variant]
    image = row.get(image_col)

    if image is None:
        variant = "without_text" if variant == "with_text" else "with_text"
        image_col, code_col, description_col = VARIANT_COLUMNS[variant]
        image = row.get(image_col)

    if image is None:
        raise ValueError(f"row has no populated image for either variant: {sorted(row.keys())}")

    return variant, image, row[code_col], row.get(description_col)


def iter_clean_images(
    num_samples: int,
    seed: Optional[int] = None,
    start_index: int = 0,
    shuffle_buffer_size: int = 10000,
):
    """Yields (id, variant, clean_image) triples from the `train` split - one variant per row,
    picked the same way `pick_variant` does, for manual eyeballing scripts."""
    import itertools

    ds = load_train_split(streaming=True, shuffle_seed=seed, shuffle_buffer_size=shuffle_buffer_size)
    window = itertools.islice(ds, start_index, start_index + num_samples)
    for row_index, row in enumerate(window):
        variant, image, _code, _description = pick_variant(row, seed=(seed or 0) + start_index + row_index)
        if not hasattr(image, "convert"):
            from PIL import Image as PILImage

            image = PILImage.open(image)
        row_id = row.get("file_id") or row.get("id") or f"row_{start_index + row_index}"
        yield row_id, variant, image.convert("RGB")
