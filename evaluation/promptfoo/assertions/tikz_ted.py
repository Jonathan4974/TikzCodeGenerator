from pathlib import Path
import traceback

from utils.tex_edit_distance import compute_ted


REFERENCE_DIR = Path("/references")


def reference_path_from_input_image(input_image: str) -> Path:
    image_name = Path(input_image).name

    if not image_name.endswith(".png"):
        raise ValueError(f"Expected PNG input image, got: {image_name}")

    reference_name = image_name.removesuffix(".png") + ".txt"
    return REFERENCE_DIR / reference_name


def get_assert(output: str, context):
    vars_ = context.get("vars", {})

    input_image = vars_.get("input_image")

    config = context.get("config")
    threshold = float(config.get("threshold", 0.75))

    if not input_image:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"Missing vars.input_image. Available vars: {list(vars_.keys())}",
        }

    try:
        reference_path = reference_path_from_input_image(input_image)

        if not reference_path.exists():
            return {
                "pass": False,
                "score": 0.0,
                "reason": f"Reference file does not exist: {reference_path}",
            }

        reference_code = reference_path.read_text(encoding="utf-8")

        ted = compute_ted(
            generated_code=output,
            reference_code=reference_code,
        )

        ted_similarity = 1.0 / (1.0 + ted)

        return {
            "pass": ted_similarity >= threshold,
            "score": ted_similarity,
            "reason": (
                f"TED={ted_similarity:.4f}, "
                f"reference={reference_path.name}"
            )
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"TED failed: {e}\n{traceback.format_exc()}",
        }