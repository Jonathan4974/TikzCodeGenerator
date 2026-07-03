from dataclasses import dataclass

from pf_utils.crystalbleu_metric import compute_crystalbleu_score
from pf_utils.ted_metric import compute_ted


@dataclass
class CodeRewardResult:
    score: float
    ted: float
    crystalbleu: float
    reason: str


def code_reward_func(generated_code: str, reference_code: str, cfg) -> CodeRewardResult:
    crystalbleu = compute_crystalbleu_score(
        reference_code=reference_code,
        generated_code=generated_code,
        corpus_dir=cfg.crystalbleu_corpus_dir,
        k=cfg.crystalbleu_k,
        n=cfg.crystalbleu_n,
        use_cache=cfg.crystalbleu_use_cache,
    )

    ted = compute_ted(
        generated_code=generated_code,
        reference_code=reference_code,
    )

    ted_penalty = min(ted / cfg.ted_scale, 1.0)

    score = (
        cfg.crystalbleu_weight * crystalbleu
        - cfg.ted_weight * ted_penalty
    )

    return CodeRewardResult(
        score=float(score),
        ted=float(ted),
        crystalbleu=float(crystalbleu),
        reason=(
            f"code_score={score:.3f}; "
            f"crystalbleu={crystalbleu:.3f}; "
            f"ted_distance={ted:.3f}; "
            f"ted_penalty={ted_penalty:.3f}"
        ),
    )