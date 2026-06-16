from pathlib import Path
import tempfile
import traceback
import shutil

from utils.tikz_rendering import render_tex_to_png
from utils.dists_similarity import (
    compute_dists_distance,
    dists_distance_to_similarity,
)


def as_bool(value, default=False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    return bool(value)


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    config = context.get("config", {})

    threshold = float(config.get("threshold", 0.75))
    reference_image = vars_.get("reference_image")
    debug_enabled = as_bool(config.get("debug", vars_.get("debug", False)))
    debug_dir = Path(config.get("debug_dir",vars_.get("debug_dir", "/app/debug_dists")))

    if not reference_image:
        return {
            "pass": False,
            "score": 0.0,
            "reason": "Missing vars.input_image or vars.reference_image",
        }

    reference_image = Path(reference_image)

    if not reference_image.exists():
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"Reference image does not exist: {reference_image}",
        }

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            generated_image = tmp_dir / "generated.png"

            render_tex_to_png(tex_code=output, output_path=generated_image)

            distance = compute_dists_distance(
                image_a=reference_image,
                image_b=generated_image,
            )

            similarity = dists_distance_to_similarity(distance)

            debug_generated = None

            if debug_enabled:
                debug_dir.mkdir(parents=True, exist_ok=True)

                test_id = reference_image.stem
                debug_generated = debug_dir / f"{test_id}_generated.png"
                debug_reference = debug_dir / f"{test_id}_reference.png"
                debug_output = debug_dir / f"{test_id}_output.tex"

                shutil.copyfile(generated_image, debug_generated)
                shutil.copyfile(reference_image, debug_reference)
                debug_output.write_text(output, encoding="utf-8")

        reason = (
            f"DISTS similarity={similarity:.4f}, "
            f"DISTS distance={distance:.4f}"
        )

        return {
            "pass": similarity >= threshold,
            "score": similarity,
            "reason": reason
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"DISTS failed: {e}\n{traceback.format_exc()}",
        }