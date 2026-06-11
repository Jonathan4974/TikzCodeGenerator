from pathlib import Path
import tempfile
import traceback

from utils.tikz_rendering import render_tex_to_png
from utils.siglip_cc_similarity import compute_siglip_cc, siglip_cc_to_score

def get_assert(output: str, context):
    vars_ = context.get("vars", {})

    input_image = vars_.get("input_image")
    threshold = float(vars_.get("siglip_cc_threshold", 0.70))

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

            render_tex_to_png(tex_code=output, output_path=generated_image)

            cc = compute_siglip_cc(
                image_a=input_image,
                image_b=generated_image,
            )

            score = siglip_cc_to_score(cc)

        return {
            "pass": score >= threshold,
            "score": score,
            "reason": f"SigLIP-CC={score:.4f} (cc={cc:.4f}), threshold={threshold:.4f}"
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"SigLIP-CC failed: {e}\n{traceback.format_exc()}",
        }
