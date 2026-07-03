import re
import difflib


class TikZRewards:
    BAD_PHRASES = [
        "here is",
        "the code",
        "explanation",
        "i will",
        "sure",
        "dieser code",
    ]

    IMPORTANT_TOKENS = [
        "\\draw",
        "\\node",
        "\\path",
        "\\fill",
        "\\coordinate",
        "\\matrix",
        "\\usetikzlibrary",
        "\\begin{axis}",
        "\\addplot",
    ]

    @staticmethod
    def completion_to_text(completion):
        if isinstance(completion, str):
            return completion

        if isinstance(completion, list):
            return "\n".join(TikZRewards.completion_to_text(x) for x in completion)

        if isinstance(completion, dict):
            return TikZRewards.completion_to_text(completion.get("content", ""))

        return str(completion)

    @staticmethod
    def clean_code(x):
        text = TikZRewards.completion_to_text(x).strip()

        text = re.sub(r"```(?:latex|tex)?", "", text)
        text = text.replace("```", "")

        return text.strip()

    @staticmethod
    def formatting_reward_func(completions, **kwargs):
        scores = []

        for completion in completions:
            raw = TikZRewards.completion_to_text(completion)
            code = TikZRewards.clean_code(raw)

            score = 0.0

            if "\\documentclass" in code:
                score += 0.4
            if "\\begin{document}" in code:
                score += 0.3
            if "\\end{document}" in code:
                score += 0.3

            if "\\begin{tikzpicture}" in code:
                score += 0.5
            if "\\end{tikzpicture}" in code:
                score += 0.5

            if "```" not in raw:
                score += 0.3

            if not any(word in code.lower()[:300] for word in TikZRewards.BAD_PHRASES):
                score += 0.3

            if len(code) > 200:
                score += 0.2

            if code.count("{") == code.count("}"):
                score += 0.3

            scores.append(score)

        return scores

    @staticmethod
    def correctness_reward_func(prompts, completions, answer, **kwargs):
        scores = []

        for completion, ref in zip(completions, answer):
            pred = TikZRewards.clean_code(completion)
            ref = TikZRewards.clean_code(ref)

            similarity = difflib.SequenceMatcher(
                None,
                pred[:5000],
                ref[:5000],
            ).ratio()

            score = similarity * 1.5

            hits = 0
            possible = 0

            for token in TikZRewards.IMPORTANT_TOKENS:
                if token in ref:
                    possible += 1
                    if token in pred:
                        hits += 1

            if possible > 0:
                score += hits / possible

            scores.append(score)

        return scores