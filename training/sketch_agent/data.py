from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from .config import SketchAgentConfig
from .ultrasketch_methods import (
    ULTRASKETCH_PROMPT,
    load_ultrasketch_pipeline,
    random_displacement_field,
    resize_to_multiple,
    run_ultrasketch,
)


@dataclass
class SyntheticPair:
    input_path: Path
    target_path: Path
    source_name: str
    method: str


def sketch_to_canny(image: Image.Image, low_threshold: int, high_threshold: int) -> Image.Image:
    arr = cv2.Canny(np.array(image.convert("RGB")), low_threshold, high_threshold)
    return Image.fromarray(np.stack([arr] * 3, axis=-1))


_hed_detector = None


def sketch_to_scribble(image: Image.Image) -> Image.Image:
    """Matches xinsir/controlnet-scribble-sdxl-1.0's."""
    global _hed_detector
    if _hed_detector is None:
        from controlnet_aux import HEDdetector

        _hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
    size = image.size[0]
    return _hed_detector(image, scribble=True, detect_resolution=size, image_resolution=size).convert("RGB")


def prepare_conditioning_image(image: Image.Image, cfg: "SketchAgentConfig") -> Image.Image:
    """Single dispatch point to switch modes (canny or scribble)"""
    if cfg.conditioning_mode == "canny":
        return sketch_to_canny(image, cfg.canny_low_threshold, cfg.canny_high_threshold)
    return sketch_to_scribble(image)


def iter_datikz_renders(
    num_samples: int,
    seed: Optional[int] = None,
    dataset_name: str = "nllg/DaTikZ-V4",
    split: str = "train",
    streaming: bool = True,
    start_index: int = 0,
) -> Iterator[Tuple[str, Image.Image]]:
    """Source clean renders from DaTikZ-V4.

    Defaults to streaming + shuffle rather than a plain full-dataset
    load: DaTikZ-V4 has 400k+ rows and only a handful of renders are needed per
    batch of synthetic pairs. `start_index` skips that many rows of the
    shuffled stream, so topping up an existing dataset pulls
    unseen rows instead of re-pulling the same prefix.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    if seed is not None:
        ds = ds.shuffle(seed=seed, buffer_size=max(num_samples * 10, 1000))

    rows = itertools.islice(ds, start_index, start_index + num_samples)
    for row in rows:
        # DaTikZ-V4 columns: file_id, caption, vlm_description, tikz_code, source, png_image
        yield row["file_id"], row["png_image"].convert("RGB")


def generate_synthetic_pairs(
    output_dir: str | Path,
    num_samples: int,
    seed: int,
    ultrasketch_probability: float = 0.5,
    displacement_alpha: float = 6.0,
    displacement_sigma: float = 12.0,
    datikz_dataset_name: str = "nllg/DaTikZ-V4",
    datikz_split: str = "train",
    datikz_streaming: bool = True,
) -> List[SyntheticPair]:
    """Generate (sketch, clean_render) pairs, selecting ONE method (ultrasketch or displacement) per pair.

    Each pair is independently routed to either UltraSketch or the displacement-field
    warp via a single random draw.

    Incremental: if `output_dir` already contains at least `num_samples` pairs
    (per `load_synthetic_dataset`), those are returned unchanged; otherwise only the
    missing pairs are generated and appended.
    """
    base_dir = Path(output_dir)
    input_dir = base_dir / "inputs"
    target_dir = base_dir / "targets"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_synthetic_dataset(base_dir)
    if len(pairs) >= num_samples:
        return pairs[:num_samples]

    existing_count = len(pairs)
    num_new = num_samples - existing_count
    rng = np.random.default_rng(seed)
    manifest_path = base_dir / "manifest.jsonl"
    pipe = None

    renders = iter_datikz_renders(
        num_new,
        seed=seed,
        dataset_name=datikz_dataset_name,
        split=datikz_split,
        streaming=datikz_streaming,
        start_index=existing_count,
    )
    try:
        for offset in range(num_new):
            idx = existing_count + offset
            source_name, clean_image = next(renders)
            clean_image = resize_to_multiple(clean_image.convert("RGB"))

            if rng.random() < ultrasketch_probability:
                if pipe is None:
                    pipe = load_ultrasketch_pipeline()
                sketch_image = run_ultrasketch(pipe, clean_image, ULTRASKETCH_PROMPT)
                method = "ultrasketch"
            else:
                sketch_image = random_displacement_field(
                    clean_image, alpha=displacement_alpha, sigma=displacement_sigma, seed=seed + idx
                )
                method = "displacement"

            target_path = target_dir / f"sample_{idx:03d}_target.png"
            input_path = input_dir / f"sample_{idx:03d}_input.png"
            clean_image.save(target_path)
            sketch_image.save(input_path)

            pairs.append(SyntheticPair(input_path=input_path, target_path=target_path, source_name=source_name, method=method))

            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "idx": idx,
                            "source_name": source_name,
                            "method": method,
                            "input": str(input_path),
                            "target": str(target_path),
                        }
                    )
                    + "\n"
                )
    finally:
        # UltraSketch is itself a full SDXL-scale img2img pipeline;
        # release it before the caller loads the SketchAgent SDXL model in the same process
        if pipe is not None:
            del pipe
            torch.cuda.empty_cache()

    return pairs


def load_synthetic_dataset(output_dir: str | Path) -> List[SyntheticPair]:
    base_dir = Path(output_dir)
    input_dir = base_dir / "inputs"
    target_dir = base_dir / "targets"
    if not input_dir.exists() or not target_dir.exists():
        return []

    manifest_path = base_dir / "manifest.jsonl"
    methods_by_stem: dict[str, str] = {}
    sources_by_stem: dict[str, str] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            stem = Path(entry["input"]).stem.replace("_input", "")
            methods_by_stem[stem] = entry["method"]
            sources_by_stem[stem] = entry["source_name"]

    pairs: List[SyntheticPair] = []
    for input_path in sorted(input_dir.glob("*.png")):
        stem = input_path.stem.replace("_input", "")
        target_path = target_dir / f"{stem}_target.png"
        if target_path.exists():
            pairs.append(
                SyntheticPair(
                    input_path=input_path,
                    target_path=target_path,
                    source_name=sources_by_stem.get(stem, stem),
                    method=methods_by_stem.get(stem, "unknown"),
                )
            )
    return pairs


class SketchAgentDataset(torch.utils.data.Dataset):
    """(conditioning(sketch), clean_render) pairs for SDXL+ControlNet+LoRA training.

    Combines freshly-generated/cached synthetic pairs (UltraSketch or displacement,
    sourced from DaTikZ-V4) with any real SketchFig pairs passed in via `extra_pairs`.
    """

    def __init__(self, cfg: SketchAgentConfig, extra_pairs: Optional[List[SyntheticPair]] = None):
        self.cfg = cfg
        synthetic_pairs = (
            generate_synthetic_pairs(
                cfg.synthetic_dir,
                num_samples=cfg.synthetic_samples,
                seed=cfg.seed,
                ultrasketch_probability=cfg.ultrasketch_probability,
                displacement_alpha=cfg.displacement_alpha,
                displacement_sigma=cfg.displacement_sigma,
                datikz_dataset_name=cfg.datikz_dataset_name,
                datikz_split=cfg.datikz_split,
                datikz_streaming=cfg.datikz_streaming,
            )
            if cfg.use_synthetic_data
            else []
        )
        self.pairs = synthetic_pairs + (extra_pairs or [])
        if not self.pairs:
            raise ValueError(
                "SketchAgentDataset has no training pairs: enable use_synthetic_data and/or "
                "use_sketchfig with a non-empty sketchfig_train_fraction"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        size = (self.cfg.image_size, self.cfg.image_size)
        sketch = Image.open(pair.input_path).convert("RGB").resize(size)
        target = Image.open(pair.target_path).convert("RGB").resize(size)

        conditioning_rgb = prepare_conditioning_image(sketch, self.cfg)

        return {
            "pixel_values": to_tensor(target) * 2.0 - 1.0,
            "conditioning_pixel_values": to_tensor(conditioning_rgb),
            "original_size": torch.tensor([self.cfg.image_size, self.cfg.image_size]),
            "crop_top_left": torch.tensor([0, 0]),
            "target_size": torch.tensor([self.cfg.image_size, self.cfg.image_size]),
        }
