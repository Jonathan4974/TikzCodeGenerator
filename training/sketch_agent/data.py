from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .ultrasketch_methods import (
    DryRunUltraSketchPipe,
    ULTRASKETCH_PROMPT,
    load_ultrasketch_pipeline,
    random_displacement_field,
    resize_to_multiple,
    run_ultrasketch,
)

RenderSource = Callable[[int, int], Iterator[Tuple[str, Image.Image]]]

@dataclass
class SyntheticPair:
    input_path: Path
    target_path: Path
    source_name: str
    method: str

"""
Fake Renders for local test, deleting this later
"""
def _make_target_image(size: int, rng: np.random.Generator) -> Image.Image:
    image = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(image)
    for _ in range(4):
        x0 = int(rng.uniform(0, size * 0.4))
        y0 = int(rng.uniform(0, size * 0.8))
        x1 = int(rng.uniform(x0 + 4, size))
        y1 = int(rng.uniform(y0 + 4, size))
        draw.rectangle([x0, y0, x1, y1], outline=(40, 60, 80), width=3)

    for _ in range(3):
        cx = int(rng.uniform(size * 0.2, size * 0.8))
        cy = int(rng.uniform(size * 0.2, size * 0.8))
        radius = int(rng.uniform(size * 0.05, size * 0.12))
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(80, 90, 100), width=3)

    for _ in range(2):
        x0 = int(rng.uniform(0, size))
        y0 = int(rng.uniform(0, size))
        x1 = int(rng.uniform(0, size))
        y1 = int(rng.uniform(0, size))
        draw.line([x0, y0, x1, y1], fill=(100, 110, 120), width=2)

    return image


def iter_fake_renders(num_samples: int, image_size: int, seed: int, start_index: int = 0) -> Iterator[Tuple[str, Image.Image]]:
    """Simple render source for dry runs and tests.
    """
    for offset in range(num_samples):
        idx = start_index + offset
        rng = np.random.default_rng(seed + idx)
        yield f"synthetic_{idx:03d}", _make_target_image(image_size, rng)


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
    import itertools

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
    render_source: RenderSource,
    ultrasketch_probability: float = 0.5,
    ultrasketch_pipe_factory: Optional[Callable[[], Any]] = None,
    displacement_alpha: float = 6.0,
    displacement_sigma: float = 12.0,
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
    pipe: Optional[Any] = None

    renders = render_source(num_new, existing_count)
    for offset in range(num_new):
        idx = existing_count + offset
        source_name, clean_image = next(renders)
        clean_image = resize_to_multiple(clean_image.convert("RGB"))

        if rng.random() < ultrasketch_probability:
            if pipe is None:
                pipe = (ultrasketch_pipe_factory or load_ultrasketch_pipeline)()
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

    return pairs


def build_synthetic_dataset(output_dir: str | Path, num_samples: int = 8, image_size: int = 256, seed: int = 3407) -> List[SyntheticPair]:
    """wrapper: fake renders pair generation."""
    return generate_synthetic_pairs(
        output_dir,
        num_samples=num_samples,
        seed=seed,
        render_source=lambda n, start: iter_fake_renders(n, image_size, seed, start_index=start),
        ultrasketch_pipe_factory=lambda: DryRunUltraSketchPipe(),
    )


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
