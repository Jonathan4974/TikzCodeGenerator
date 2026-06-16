from pathlib import Path
import tempfile
import traceback

from utils.tikz_rendering import render_tex_to_png
from utils.semantic_similarity import image_cosine_similarity


def get_assert(output: str, context):
    vars_ = context.get("vars", {})

    reference_image = vars_.get("reference_image")

    config = context.get("config")
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

            render_tex_to_png(tex_code=output, output_path=generated_image)

            score = image_cosine_similarity(
                image_a=reference_image,
                image_b=generated_image,
                model_key="siglip",
            )

        return {
            "pass": score >= threshold,
            "score": score,
            "reason": f"SigLIP similarity={score:.4f}"
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"SigLIP similarity failed: {e}\n{traceback.format_exc()}",
        }