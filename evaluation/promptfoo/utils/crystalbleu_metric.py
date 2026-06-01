from collections import Counter
from functools import lru_cache
from pathlib import Path
import re

from nltk.util import ngrams
from crystalbleu import corpus_bleu


def tokenize_latex_code(code: str) -> list[str]:
    return re.findall(
        r"\\[a-zA-Z]+|\\.|[a-zA-Z_][a-zA-Z0-9_]*|\d+(?:\.\d+)?|[^\s]",
        code,
    )


@lru_cache(maxsize=1)
def load_trivially_shared_ngrams(
    reference_dir: str,
    max_files: int = 5000,
    k: int = 500,
) -> dict:
    reference_dir = Path(reference_dir)

    frequencies = Counter()
    reference_files = sorted(reference_dir.glob("*.txt"))[:max_files]

    for reference_file in reference_files:
        code = reference_file.read_text(encoding="utf-8", errors="replace")
        tokens = tokenize_latex_code(code)

        for n in range(1, 5):
            frequencies.update(ngrams(tokens, n))

    return dict(frequencies.most_common(k))


def compute_crystalbleu_score(
    reference_code: str,
    generated_code: str,
    reference_dir: str,
    k: int = 500,
) -> float:
    ignoring = load_trivially_shared_ngrams(
        reference_dir=reference_dir,
        k=k,
    )

    reference_tokens = tokenize_latex_code(reference_code)
    generated_tokens = tokenize_latex_code(generated_code)

    references = [[reference_tokens]]
    candidates = [generated_tokens]

    score = corpus_bleu(
        references,
        candidates,
        ignoring=ignoring,
    )

    return float(score)