"""Manual check: runs both sketch-generation methods on a couple of real clean images
and saves side-by-side comparisons to check if everything alright.

Usage:
    python -m training.sketch_agent.check_sketch_generation --split datikz_v4 --num-samples 2
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _value = _line.partition("=")
        os.environ.setdefault(_key.strip(), _value.strip().strip('"'))

from PIL import Image, ImageDraw

from .config import SketchAugmentationConfig
from .dataset_loader import iter_clean_images
from .sketch_generation import (
    generate_synthetic_sketch,
    load_ultrasketch_pipeline,
    release_ultrasketch_pipeline,
)

OUTPUT_DIR = Path(__file__).parent / "output_check" / "sketch_generation"
LABEL_HEIGHT = 24

# Seed for "eyeball" checks only
DEFAULT_CHECK_SEED = 3407


def save_comparison(clean: Image.Image, ultrasketch: Image.Image, displacement: Image.Image, out_path: Path) -> None:
    w, h = clean.size
    canvas = Image.new("RGB", (w * 3 + 20, h + LABEL_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (img, label) in enumerate(
        [(clean, "clean (input)"), (ultrasketch, "ultrasketch"), (displacement, "displacement")]
    ):
        x = i * (w + 10)
        canvas.paste(img, (x, LABEL_HEIGHT))
        draw.text((x + 4, 4), label, fill="black")
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, help="a split name from dataset_loader.SPLITS")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = SketchAugmentationConfig()
    seed = args.seed if args.seed is not None else DEFAULT_CHECK_SEED

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pipe = load_ultrasketch_pipeline()
    try:
        import torch

        for i, (row_id, image) in enumerate(iter_clean_images(args.split, args.num_samples, seed=seed)):
            sample_seed = seed + i

            # Force each method deterministically for eyeballing (sketch_probability=1.0
            # "Training" draws use the default 50/25/25 split.
            ultrasketch_out, _ = generate_synthetic_sketch(
                image, sample_seed, sketch_probability=1.0, ultrasketch_probability=1.0, pipe=pipe
            )
            displacement_out, _ = generate_synthetic_sketch(
                image, sample_seed, sketch_probability=1.0, ultrasketch_probability=0.0,
                displacement_alpha=cfg.displacement_alpha, displacement_sigma=cfg.displacement_sigma,
            )

            comparison_path = OUTPUT_DIR / f"{row_id}_comparison.png"
            save_comparison(image.resize(ultrasketch_out.size), ultrasketch_out, displacement_out, comparison_path)
            print(f"{row_id}: saved {comparison_path}")

        if torch.cuda.is_available():
            print(f"peak VRAM allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            print(f"peak VRAM reserved:  {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
    finally:
        release_ultrasketch_pipeline(pipe)


if __name__ == "__main__":
    main()
