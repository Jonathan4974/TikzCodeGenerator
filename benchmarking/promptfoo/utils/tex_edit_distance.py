from latexcodec.lexer import LatexIncrementalLexer
from torchmetrics.text import ExtendedEditDistance


def latexcodec_tokens(code: str) -> list[str]:
    lexer = LatexIncrementalLexer()
    raw_tokens = list(lexer.get_tokens(code, final=True))

    tokens = []
    buffer = ""

    for tok in raw_tokens:
        if tok.name == "space":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            continue

        if tok.name == "chars":
            buffer += tok.text
        else:
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append(tok.text)

    if buffer:
        tokens.append(buffer)

    return tokens


def compute_ted(generated_code: str, reference_code: str) -> float:
    metric = ExtendedEditDistance()

    generated_tokens = " ".join(latexcodec_tokens(generated_code))
    reference_tokens = " ".join(latexcodec_tokens(reference_code))

    return float(metric([generated_tokens], [reference_tokens]))