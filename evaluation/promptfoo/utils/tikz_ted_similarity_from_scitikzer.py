from __future__ import annotations

from pathlib import Path
from typing import List
import math
import re

from pygments.lexers.markup import TexLexer
from pygments.token import Comment, Text

from torchmetrics.text import ExtendedEditDistance
from torchmetrics.functional.text.eed import (
    _compute_sentence_statistics,
    _preprocess_en,
    _preprocess_ja,
)
from torchmetrics.functional.text.helper import _validate_inputs


def extract_document_body(tex: str) -> str:
    match = re.search(
        r"\\begin\{document\}(.*)\\end\{document\}",
        tex,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1) if match else tex


def strip_latex_comments(tex: str) -> str:

    out_lines = []

    for line in tex.splitlines():
        cut = None

        for i, char in enumerate(line):
            if char == "%":
                if i > 0 and line[i - 1] == "\\":
                    continue

                cut = i
                break

        if cut is not None:
            line = line[:cut]

        out_lines.append(line)

    return "\n".join(out_lines)


def normalize_tex(tex: str) -> str:
    tex = tex.replace("\r\n", "\n").replace("\r", "\n")
    tex = strip_latex_comments(tex)
    tex = re.sub(r"[ \t]+", " ", tex)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    return tex.strip()


def normalize_tex_body(tex: str) -> str:
    return normalize_tex(extract_document_body(tex))


class TokenEditDistance(ExtendedEditDistance):
    """
    Extended Edit Distance over TeX/Pygments-Tokens.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lexer = TexLexer()

    @staticmethod
    def _is_comment(tokentype) -> bool:
        return tokentype in Comment

    @staticmethod
    def _is_text(tokentype) -> bool:
        return tokentype in Text

    def tokenize_to_tokens(self, text: str, language: str) -> List[str]:
        norm = normalize_tex(text)
        tokens: List[str] = []

        for tokentype, value in self.lexer.get_tokens(norm):
            if not value or not value.strip():
                continue

            if self._is_comment(tokentype):
                continue

            value = value.strip()

            if not value:
                continue

            if self._is_text(tokentype):
                if language == "en":
                    preprocess_function = _preprocess_en
                elif language == "ja":
                    preprocess_function = _preprocess_ja
                else:
                    raise ValueError(f"language must be en/ja, got {language}")

                tokens.extend(preprocess_function(value).split())
            else:
                tokens.extend(value.split())

        return tokens

    def _preprocess_sentences(self, preds, target, language):
        target, preds = _validate_inputs(
            hypothesis_corpus=preds,
            ref_corpus=target,
        )

        def to_eed_string(text: str) -> str:
            tokens = self.tokenize_to_tokens(text, language=language)
            return " " + " ".join(tokens) + " "

        preds = [to_eed_string(pred) for pred in preds]
        target = [
            [to_eed_string(ref) for ref in reference]
            for reference in target
        ]

        return preds, target

    def update(self, preds, target):
        preds, target = self._preprocess_sentences(
            preds,
            target,
            self.language,
        )

        if self.sentence_eed is None:
            self.sentence_eed = []

        if 0 in (len(preds), len(target[0])):
            return self.sentence_eed

        for hypothesis, target_words in zip(preds, target):
            score = _compute_sentence_statistics(
                hypothesis,
                target_words,
                self.alpha,
                self.rho,
                self.deletion,
                self.insertion,
            )
            self.sentence_eed.append(score)

        return self.sentence_eed

    def compute(self, *args, **kwargs):
        return super().compute(*args, **kwargs).item()


def ted_distance_to_similarity(
    ted_dist_norm: float,
    tau: float = 0.4,
) -> float:

    if not math.isfinite(ted_dist_norm):
        return 0.0

    if tau <= 0:
        return 0.0

    distance = max(0.0, float(ted_dist_norm))

    return float(math.exp(-distance / tau))


def compute_ted_metrics(
    reference: str,
    prediction: str,
    language: str = "en",
    alpha: float = 2.0,
    rho: float = 0.3,
    deletion: float = 0.2,
    insertion: float = 1.0,
    tau: float = 0.4,
) -> dict:

    reference_body = normalize_tex_body(reference)
    prediction_body = normalize_tex_body(prediction)

    metric = TokenEditDistance(
        language=language,
        alpha=alpha,
        rho=rho,
        deletion=deletion,
        insertion=insertion,
    )

    metric.reset()

    ted_dist_norm = float(
        metric(
            preds=[prediction_body],
            target=[[reference_body]],
        )
    )

    if not math.isfinite(ted_dist_norm):
        ted_dist_norm = float("nan")

    try:
        ref_tokens = metric.tokenize_to_tokens(
            reference_body,
            language=language,
        )
        ref_len = max(len(ref_tokens), 1)
    except Exception:
        ref_len = 1

    ted_dist = (
        float(ted_dist_norm) * float(ref_len)
        if math.isfinite(ted_dist_norm)
        else float("nan")
    )

    ted_similarity = ted_distance_to_similarity(
        ted_dist_norm=ted_dist_norm,
        tau=tau,
    )

    return {
        "ted_dist": ted_dist,
        "ted_dist_norm": ted_dist_norm,
        "ted_similarity": ted_similarity,
        "reference_token_count": ref_len,
    }


def load_text_value(value: str) -> str:

    if value.startswith("file://"):
        path = Path(value.removeprefix("file://"))
        return path.read_text(encoding="utf-8")

    path = Path(value)

    if path.exists():
        return path.read_text(encoding="utf-8")

    return value