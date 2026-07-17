from assertions.common import code_assertion, setting
from config import PATHS
from pf_utils.crystalbleu_metric import compute_crystalbleu_score


def score(reference: str, generated: str, config: dict):
    k = int(setting("crystalbleu", "k", config))
    n = int(setting("crystalbleu", "n", config))
    use_cache = bool(setting("crystalbleu", "use_cache", config))
    value = compute_crystalbleu_score(reference, generated, PATHS.crystalbleu_corpus, k, n, use_cache)
    return value, {}, f"k={k}; n={n}"


def get_assert(output: str, context: dict):
    return code_assertion(output, context, "crystalbleu", score)
