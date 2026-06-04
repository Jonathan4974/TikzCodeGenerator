from __future__ import annotations

from collections import Counter
from functools import cached_property
from hashlib import md5
from itertools import chain, tee
from pathlib import Path
from pickle import dump, load
from typing import Dict, List, Optional, Tuple
import os
import re

from crystalbleu import corpus_bleu
from pygments.lexers.markup import TexLexer
from pygments.token import Comment, Name, Text
from sacremoses import MosesTokenizer


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


def pad_sequence(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    sequence = iter(sequence)

    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)

    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))

    return sequence


def ngrams(sequence, n, **kwargs):
    sequence = pad_sequence(sequence, n, **kwargs)
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)

    return zip(*iterables)


class CrystalBLEULatex:
    def __init__(
        self,
        corpus: List[str],
        k: int = 500,
        n: int = 4,
        use_cache: bool = True,
        cache_dir: Optional[str | Path] = None,
    ):
        self.lexer = TexLexer()
        self.tokenizer = MosesTokenizer()

        self.corpus = list(corpus)
        self.k = int(k)
        self.n = int(n)
        self.use_cache = bool(use_cache)

        if cache_dir is None:
            cache_dir = Path(os.path.expanduser("~/.cache/crystalbleu_latex"))

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _is_comment_token(self, tokentype) -> bool:
        return tokentype in Comment

    def _is_text_like_token(self, tokentype) -> bool:
        if tokentype in Text:
            return True

        if tokentype in Name.Attribute or tokentype in Name.Builtin:
            return True

        return False

    def tokenize(self, tex: str) -> List[str]:
        tokens: List[str] = []
        tex = normalize_tex(tex)

        for tokentype, value in self.lexer.get_tokens(tex):
            if not value or not value.strip():
                continue

            if self._is_comment_token(tokentype):
                continue

            value = value.strip()

            if not value:
                continue

            if self._is_text_like_token(tokentype):
                tokens.extend(self.tokenizer.tokenize(value))
            else:
                tokens.append(value)

        return tokens

    def _corpus_fingerprint(self) -> str:
        h = md5()
        h.update(f"k={self.k};n={self.n};len={len(self.corpus)}".encode("utf-8"))

        for tex in self.corpus:
            normalized = normalize_tex(tex)
            data = normalized.encode("utf-8", errors="ignore")

            h.update(len(data).to_bytes(8, "little", signed=False))
            h.update(data[:4096])
            h.update(md5(data).digest())

        return h.hexdigest()

    @cached_property
    def trivially_shared_ngrams(self) -> Dict[Tuple[str, ...], int]:
        cache_file = self.cache_dir / f"trivial_{self._corpus_fingerprint()}.pkl"

        if self.use_cache and cache_file.is_file():
            with open(cache_file, "rb") as f:
                return load(f)

        freq: Counter = Counter()

        for tex in self.corpus:
            tokens = self.tokenize(tex)

            if not tokens:
                continue

            for order in range(1, self.n + 1):
                freq.update(ngrams(tokens, order))

        trivial = dict(freq.most_common(self.k))

        if self.use_cache:
            with open(cache_file, "wb") as f:
                dump(trivial, f)

        return trivial

    def score_sentence(self, reference: str, prediction: str) -> float:
        reference_tokens = self.tokenize(reference)
        prediction_tokens = self.tokenize(prediction)

        if not reference_tokens or not prediction_tokens:
            return 0.0

        return float(
            corpus_bleu(
                list_of_references=[[reference_tokens]],
                hypotheses=[prediction_tokens],
                ignoring=self.trivially_shared_ngrams,
            )
        )

    def score_corpus(self, references: List[str], predictions: List[str]) -> float:
        if len(references) != len(predictions):
            raise ValueError("references and predictions must have the same length")

        list_of_references = [[self.tokenize(ref)] for ref in references]
        hypotheses = [self.tokenize(pred) for pred in predictions]

        return float(
            corpus_bleu(
                list_of_references=list_of_references,
                hypotheses=hypotheses,
                ignoring=self.trivially_shared_ngrams,
            )
        )


def compute_crystalbleu_score(
    reference: str,
    prediction: str,
    corpus: List[str],
    k: int = 500,
    n: int = 4,
    use_cache: bool = True,
    cache_dir: Optional[str | Path] = None,
) -> float:
    metric = CrystalBLEULatex(
        corpus=[normalize_tex_body(item) for item in corpus],
        k=k,
        n=n,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )

    return metric.score_sentence(
        reference=normalize_tex_body(reference),
        prediction=normalize_tex_body(prediction),
    )