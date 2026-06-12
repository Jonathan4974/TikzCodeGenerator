from pathlib import Path
import traceback

from utils.crystalbleu_similarity_from_scitikzer import compute_crystalbleu_score


def as_bool(value, default=False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    return bool(value)


def read_text_value(value: str) -> str:
    if value.startswith("file://"):
        path = Path(value.removeprefix("file://"))
        return path.read_text(encoding="utf-8")

    return value


def load_corpus_from_dir(corpus_dir: str | Path) -> list[str]:
    corpus_dir = Path(corpus_dir)

    if not corpus_dir.exists():
        return []

    corpus = []

    for path in sorted(corpus_dir.glob("*.txt")):
        try:
            corpus.append(path.read_text(encoding="utf-8"))
        except Exception:
            continue

    return corpus


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    config = context.get("config", {})

    threshold = float(config.get("threshold", 0.75))

    reference_text = vars_.get("reference_text")
    k = int(config.get("k", vars_.get("crystalbleu_k", 500)))
    n = int(config.get("n", vars_.get("crystalbleu_n", 4)))
    use_cache = as_bool(config.get("use_cache",vars_.get("crystalbleu_use_cache", True)), default=True)
    cache_dir = config.get("cache_dir", vars_.get("crystalbleu_cache_dir", "/app/.cache/crystalbleu_latex"))
    corpus_dir = config.get("corpus_dir", vars_.get("crystalbleu_corpus_dir", "/references"))

    if not reference_text:
        return {
            "pass": False,
            "score": 0.0,
            "reason": "Missing vars.reference_text",
        }

    try:
        reference = read_text_value(reference_text)

        corpus = load_corpus_from_dir(corpus_dir)
        if not corpus:
            corpus = [reference]

        score = compute_crystalbleu_score(
            reference=reference,
            prediction=output,
            corpus=corpus,
            k=k,
            n=n,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )

        return {
            "pass": score >= threshold,
            "score": score,
            "reason": (
                f"CrystalBLEU Scitikzer ={score:.4f}, "
                f"corpus_size={len(corpus)}"
            )
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"CrystalBLEU failed: {e}\n{traceback.format_exc()}",
        }