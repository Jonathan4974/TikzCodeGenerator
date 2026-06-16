import traceback

from utils.tikz_ted_similarity_from_scitikzer import (
    compute_ted_metrics,
    load_text_value,
)


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    config = context.get("config", {})

    threshold = float(config.get("threshold", 0.75))
    reference_code = vars_.get("reference_code")
    language = str(config.get("language", vars_.get("ted_language", "en")))
    alpha = float(config.get("alpha", vars_.get("ted_alpha", 2.0)))
    rho = float(config.get("rho", vars_.get("ted_rho", 0.3)))
    deletion = float(config.get("deletion", vars_.get("ted_deletion", 0.2)))
    insertion = float(config.get("insertion", vars_.get("ted_insertion", 1.0)))
    tau = float(config.get("tau", vars_.get("ted_tau", 0.4)))

    if not reference_code:
        return {
            "pass": False,
            "score": 0.0,
            "reason": "Missing vars.reference_code",
        }

    try:
        reference = load_text_value(reference_code)

        metrics = compute_ted_metrics(
            reference=reference,
            prediction=output,
            language=language,
            alpha=alpha,
            rho=rho,
            deletion=deletion,
            insertion=insertion,
            tau=tau,
        )

        ted_similarity = metrics["ted_similarity"]
        ted_dist = metrics["ted_dist"]
        ted_dist_norm = metrics["ted_dist_norm"]
        reference_token_count = metrics["reference_token_count"]

        return {
            "pass": ted_similarity >= threshold,
            "score": ted_similarity,
            "reason": (
                f"TED similarity Scitikzer={ted_similarity:.4f}, "
                f"ted_dist={ted_dist:.4f}, "
                f"ted_dist_norm={ted_dist_norm:.4f}, "
                f"reference_token_count={reference_token_count}, "
                f"language={language}, "
                f"tau={tau:.4f}"
            ),
            "namedScores": {
                "ted_dist_norm": ted_dist_norm,
            },
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"TED failed: {e}\n{traceback.format_exc()}",
        }