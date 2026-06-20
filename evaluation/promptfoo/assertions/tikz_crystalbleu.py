from pathlib import Path
import traceback
import os
from utils.crystalbleu_metric import compute_crystalbleu_score



def get_assert(output: str, context):
    vars_ = context.get("vars", {})

    reference_code_path = vars_.get("reference_code")
    k = int(vars_.get("crystalbleu_k", 500))

    config = context.get("config")
    threshold = float(config.get("threshold", 0.75))
    reference_code_dir = str(os.getenv("REFERENCES_DIR", "none"))
    
    if not reference_code_path:
        return {
            "pass": False,
            "score": 0.0,
            "reason": "Missing vars.reference_code",
        }

    try:

        reference_code_path = Path(reference_code_path)
        reference_code = reference_code_path.read_text(encoding="utf-8")

        score = compute_crystalbleu_score(
            reference_code=reference_code,
            generated_code=output,
            reference_dir=reference_code_dir,
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