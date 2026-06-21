from pathlib import Path
import tempfile
import traceback
import shutil
import os

from pf_utils.tikz_rendering import render_tex_to_png
from pf_utils.dists_metric import (
    compute_dists_distance,
    dists_distance_to_similarity,
)


BAD_DISTS_DISTANCE = 999.0


def as_bool(value, default=False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    return bool(value)


def failed(reason: str):
    return {
        "pass": False,
        "score": 0.0,
        "namedScores": {
            "dists_distance": BAD_DISTS_DISTANCE,
        },
        "reason": reason,
    }


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    config = context.get("config", {})

    threshold = float(config.get("threshold", 0.75))
    reference_image = vars_.get("reference_image")
    debug_enabled = as_bool(config.get("debug", vars_.get("debug", False)))
    generated_image_dir = Path(os.getenv("GENERATED_IMAGE_DIR", "none"))

    if not reference_image:
        return failed("Missing vars.input_image or vars.reference_image")

    reference_image = Path(reference_image)

    if not reference_image.exists():
        return failed(f"Reference image does not exist: {reference_image}")

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

            if debug_enabled:
                generated_image_dir.mkdir(parents=True, exist_ok=True)

                test_id = reference_image.stem
                debug_generated = generated_image_dir / f"{test_id}_generated.png"
                debug_reference = generated_image_dir / f"{test_id}_reference.png"
                debug_output = generated_image_dir / f"{test_id}_output.tex"

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
            "namedScores": {
                "dists_distance": distance,
            },
            "reason": reason,
        }

    except Exception as e:
        return failed(f"DISTS failed: {e}\n{traceback.format_exc()}")