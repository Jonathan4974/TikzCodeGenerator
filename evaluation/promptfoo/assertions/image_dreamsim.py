from pathlib import Path
import tempfile
import traceback

from utils.tikz_rendering import render_tex_to_png
from utils.dreamsim_similarity import (
    compute_dreamsim_distance,
    dreamsim_distance_to_similarity,
)


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    config = context.get("config", {}) or {}

    reference_image = vars_.get("reference_image")
    threshold = float(config.get("threshold", 0.75))

    if not reference_image:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"Missing vars.reference_image. Available vars: {list(vars_.keys())}",
        }

    reference_image = Path(reference_image)

    if not reference_image.exists():
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"Input image does not exist: {reference_image}",
        }

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            generated_image = Path(tmp_dir) / "generated.png"

            render_tex_to_png(
                tex_code=output,
                output_path=generated_image,
            )

            distance = compute_dreamsim_distance(
                image_a=reference_image,
                image_b=generated_image,
            )

            score = dreamsim_distance_to_similarity(distance)

        return {
            "pass": score >= threshold,
            "score": score,
            "reason": (
                f"DreamSim similarity={score:.4f}, "
                f"DreamSim distance={distance:.4f}"
            ),
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"DreamSim failed: {e}\n{traceback.format_exc()}",
        }