from pathlib import Path
import traceback

from utils.crystalbleu_metric import compute_crystalbleu_score


REFERENCE_DIR = Path("/references")


def reference_path_from_reference_image(reference_image: str) -> Path:
    image_name = Path(reference_image).name

    if not image_name.endswith(".png"):
        raise ValueError(f"Expected PNG image path, got: {image_name}")

    reference_name = image_name.removesuffix(".png") + ".txt"
    return REFERENCE_DIR / reference_name


def get_assert(output: str, context):
    vars_ = context.get("vars", {})

    reference_image = vars_.get("reference_image")
    k = int(vars_.get("crystalbleu_k", 500))

    config = context.get("config")
    threshold = float(config.get("threshold", 0.75))

    if not reference_image:
        return {
            "pass": False,
            "score": 0.0,
            "reason": "Missing vars.reference_image",
        }

    try:
        reference_path = reference_path_from_reference_image(reference_image)

        if not reference_path.exists():
            return {
                "pass": False,
                "score": 0.0,
                "reason": f"Reference file does not exist: {reference_path}",
            }

        reference_code = reference_path.read_text(encoding="utf-8")

        score = compute_crystalbleu_score(
            reference_code=reference_code,
            generated_code=output,
            reference_dir=str(REFERENCE_DIR),
            k=k,
        )

        return {
            "pass": score >= threshold,
            "score": score,
            "reason": (
                f"CrystalBLEU={score:.4f}"
            )
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"CrystalBLEU failed: {e}\n{traceback.format_exc()}",
        }