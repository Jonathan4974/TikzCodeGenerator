from pathlib import Path
import tempfile
import traceback

from pf_utils.tikz_rendering import render_tex_to_png
from pf_utils.lpips_metric import (
    compute_lpips_distance,
    lpips_distance_to_similarity,
)


BAD_LPIPS_DISTANCE = 999.0


def failed(reason: str):
    return {
        "pass": False,
        "score": 0.0,
        "namedScores": {
            "lpips_distance": BAD_LPIPS_DISTANCE,
        },
        "reason": reason,
    }


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    config = context.get("config", {}) or {}

    threshold = float(config.get("threshold", 0.75))
    reference_image = vars_.get("reference_image")
    net_type = str(config.get("net_type", vars_.get("lpips_net_type", "alex")))

    if not reference_image:
        return failed("Missing vars.input_image or vars.reference_image")

    reference_image = Path(reference_image)

    if not reference_image.exists():
        return failed(f"Reference image does not exist: {reference_image}")

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            generated_image = Path(tmp_dir) / "generated.png"

            render_tex_to_png(tex_code=output, output_path=generated_image)

            distance = compute_lpips_distance(
                image_a=reference_image,
                image_b=generated_image,
                net_type=net_type,
            )

            similarity = lpips_distance_to_similarity(distance)

        return {
            "pass": similarity >= threshold,
            "score": similarity,
            "namedScores": {
                "lpips_distance": distance,
            },
            "reason": (
                f"LPIPS similarity={similarity:.4f}, "
                f"LPIPS distance={distance:.4f}, "
                f"net_type={net_type}"
            ),
        }

    except Exception as e:
        return failed(f"LPIPS failed: {e}\n{traceback.format_exc()}")