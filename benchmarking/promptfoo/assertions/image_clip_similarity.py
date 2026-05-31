from pathlib import Path
import tempfile
import traceback

from utils.tikz_rendering import render_tex_to_png
from utils.semantic_similarity import image_cosine_similarity


def get_assert(output: str, context):
    vars_ = context.get("vars", {})

    input_image = vars_.get("input_image")
    threshold = float(vars_.get("clip_threshold", 0.70))

    if not input_image:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"Missing vars.input_image. Available vars: {list(vars_.keys())}",
        }

    input_image = Path(input_image)

    if not input_image.exists():
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"Input image does not exist: {input_image}",
        }

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            generated_image = Path(tmp_dir) / "generated.png"

            render_tex_to_png(
                tex_code=output,
                output_path=generated_image,
            )

            score = image_cosine_similarity(
                image_a=input_image,
                image_b=generated_image,
                model_key="clip",
            )

        return {
            "pass": score >= threshold,
            "score": score,
            "reason": f"CLIP similarity={score:.4f}, threshold={threshold:.4f}",
            "namedScores": {
                "clip_similarity": score,
            },
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"CLIP similarity failed: {e}\n{traceback.format_exc()}",
        }