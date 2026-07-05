from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def pixel_congruence_coefficient(img1: Image.Image, img2: Image.Image) -> float:
    """
    Simple pixel-cc implementation.
    """
    a = np.array(img1.convert("L")).astype(float).flatten()
    b = np.array(img2.convert("L")).astype(float).flatten()
    numerator = np.sum(a * b)
    denominator = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(numerator / denominator) if denominator != 0 else 0.0


SimilarityFn = Callable[..., float]


def _lazy_siglip_fn() -> SimilarityFn:
    try:
        from evaluation.promptfoo.pf_utils.clip_siglip_metric import image_cosine_similarity
    except ImportError as exc:
        raise RuntimeError("SigLIP eval requires transformers/torch - install and retry") from exc
    return image_cosine_similarity


def _lazy_dreamsim_fn() -> SimilarityFn:
    try:
        from evaluation.promptfoo.pf_utils.dreamsim_metric import compute_dreamsim_score
    except ImportError as exc:
        raise RuntimeError("DreamSim eval requires torch/dreamsim - install and retry") from exc
    return compute_dreamsim_score


def siglip_similarity(image_a_path: str | Path, image_b_path: str | Path, similarity_fn: Optional[SimilarityFn] = None) -> float:
    fn = similarity_fn or _lazy_siglip_fn()
    return float(fn(image_a_path, image_b_path, model_key="siglip"))


def dreamsim_similarity(image_a_path: str | Path, image_b_path: str | Path, similarity_fn: Optional[SimilarityFn] = None) -> float:
    fn = similarity_fn or _lazy_dreamsim_fn()
    return float(fn(str(image_a_path), str(image_b_path)))


def _collect_pred_target_pairs(output_dir: str | Path, reference_dir: str | Path) -> List[Tuple[Path, Path]]:
    output_root = Path(output_dir)
    reference_root = Path(reference_dir)

    pairs: List[Tuple[Path, Path]] = []
    for output_path in sorted(output_root.glob("*.png")):
        stem = output_path.stem
        target_path = reference_root / f"{stem.replace('_pred', '')}_target.png"
        if target_path.exists():
            pairs.append((output_path, target_path))
    return pairs


def _score_pixel_cc(output_path: Path, target_path: Path) -> float:
    output_image = Image.open(output_path).convert("RGB")
    target_image = Image.open(target_path).convert("RGB")
    return pixel_congruence_coefficient(output_image, target_image)


_METRIC_REGISTRY: dict[str, Callable[[Path, Path], float]] = {
    "pixel_cc": _score_pixel_cc,
    "siglip": siglip_similarity,
    "dreamsim": dreamsim_similarity,
}


def evaluate_generated_outputs(
    output_dir: str | Path,
    reference_dir: str | Path,
    metrics: Sequence[str] = ("pixel_cc",),
) -> dict:
    pairs = _collect_pred_target_pairs(output_dir, reference_dir)

    results: dict = {"count": len(pairs)}
    for metric_name in metrics:
        scorer = _METRIC_REGISTRY.get(metric_name)
        if scorer is None:
            raise ValueError(f"Unknown metric {metric_name!r}; expected one of {sorted(_METRIC_REGISTRY)}")

        if not pairs:
            results[metric_name] = {"status": "ok", "mean": 0.0}
            continue

        try:
            scores = [scorer(output_path, target_path) for output_path, target_path in pairs]
        except RuntimeError as exc:
            results[metric_name] = {"status": "unavailable", "mean": None, "reason": str(exc)}
            continue
        except Exception as exc:
            results[metric_name] = {"status": "error", "mean": None, "reason": str(exc)}
            continue

        results[metric_name] = {"status": "ok", "mean": float(np.mean(scores))}

    return results
