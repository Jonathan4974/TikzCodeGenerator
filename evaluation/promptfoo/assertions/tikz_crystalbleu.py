from pathlib import Path
import traceback
import os

from pf_utils.crystalbleu_metric import compute_crystalbleu_score


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    config = context.get("config", {}) or {}

    reference_code_path = vars_.get("reference_code")

    k = int(vars_.get("crystalbleu_k", 500))
    n = int(vars_.get("crystalbleu_n", 4))

    threshold = float(config.get("threshold", 0.75))
    crystalbleu_corpus = str(config.get("crystalbleu_corpus", "none"))

    if not reference_code_path:
        return {
            "pass": False,
            "score": 0.0,
            "reason": "Missing vars.reference_code",
        }

    if crystalbleu_corpus == "none":
        return {
            "pass": False,
            "score": 0.0,
            "reason": "Missing CRYSTALBLEU_CORPUS environment variable",
        }

    try:
        reference_code_path = Path(reference_code_path)
        reference_code = reference_code_path.read_text(encoding="utf-8")

        score = compute_crystalbleu_score(
            reference_code=reference_code,
            generated_code=output,
            corpus_dir=crystalbleu_corpus,
            k=k,
            n=n,
            use_cache=True,
        )

        return {
            "pass": score >= threshold,
            "score": score,
            "reason": f"CrystalBLEU={score:.4f}, k={k}, n={n}",
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"CrystalBLEU failed: {e}\n{traceback.format_exc()}",
        }