import re
from collections import OrderedDict

from reward_functions.render_reward import is_renderable
from reward_functions.diagnostic_reward import diagnostic_reward_func
from reward_functions.visual_reward import visual_reward_func
from reward_functions.code_reward import code_reward_func


def clean_code(completion) -> str:
    if isinstance(completion, list):
        completion = completion[0].get("content", "") if completion else ""

    if isinstance(completion, dict):
        completion = completion.get("content", "")

    code = str(completion).strip()

    code = code.replace("```latex", "")
    code = code.replace("```tex", "")
    code = code.replace("```", "")
    code = code.strip()

    end = r"\end{document}"
    if end in code:
        code = code[: code.index(end) + len(end)]

    start = r"\documentclass"
    if start in code:
        code = code[code.index(start):]

    return code.strip()


def pick(values, idx):
    if values is None:
        return None
    if isinstance(values, list):
        if len(values) == 0:
            return None
        return values[idx % len(values)]
    return values


class TikZReward:
    def __init__(self, cfg):
        self.cfg = cfg
        self.__name__ = "tikz_reward"
        self.last_metrics = {}
        self.last_examples = []

    @staticmethod
    def _mean(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        return float(sum(values) / len(values))

    def _summarize_records(self, records):
        metrics = {
            "reward/render_ok_rate": self._mean([r["render_ok"] for r in records]),
            "reward/avg_total_score": self._mean([r["total_score"] for r in records]),

            "reward/avg_latex_errors": self._mean([r["latex_errors"] for r in records]),
            "reward/avg_latex_warnings": self._mean([r["latex_warnings"] for r in records]),
            "reward/avg_latex_badboxes": self._mean([r["latex_badboxes"] for r in records]),

            "reward/avg_diagnostic_score": self._mean([r["diagnostic_score"] for r in records]),
            "reward/avg_visual_score": self._mean([r["visual_score"] for r in records]),
            "reward/avg_siglip": self._mean([r["siglip"] for r in records]),
            "reward/avg_lpips": self._mean([r["lpips"] for r in records]),
            "reward/avg_dreamsim": self._mean([r["dreamsim"] for r in records]),

            "reward/code_eval_rate": self._mean([r["code_evaluated"] for r in records]),
            "reward/avg_code_score": self._mean([r["code_score"] for r in records]),
            "reward/avg_crystalbleu": self._mean([r["crystalbleu"] for r in records]),
            "reward/avg_ted": self._mean([r["ted"] for r in records]),
        }

        return {k: v for k, v in metrics.items() if v is not None}

    def __call__(self, completions, answer=None, image=None, images=None, **kwargs):
        scores = []
        records = []
        examples = []

        input_images = image or images or kwargs.get("image") or kwargs.get("images")
        reference_codes = answer or kwargs.get("answer")

        for i, completion in enumerate(completions):
            gen_code = clean_code(completion)
            ref_code = pick(reference_codes, i)
            input_image = pick(input_images, i)

            record = {
                "render_ok": 0.0,
                "total_score": None,

                "latex_errors": None,
                "latex_warnings": None,
                "latex_badboxes": None,

                "diagnostic_score": None,
                "visual_score": None,
                "siglip": None,
                "lpips": None,
                "dreamsim": None,

                "code_evaluated": 0.0,
                "code_score": None,
                "crystalbleu": None,
                "ted": None,
            }

            render = is_renderable(gen_code)

            record["latex_errors"] = float(render.errors)
            record["latex_warnings"] = float(render.warnings)
            record["latex_badboxes"] = float(render.badboxes)

            if not render.ok:
                score = float(self.cfg.not_renderable_score)
                record["total_score"] = score

                if len(examples) < self.cfg.log_examples_max:
                    examples.append({
                        "input_image": input_image,
                        "rendered_image": None,
                        "generated_code": gen_code,
                        "reference_code": ref_code,
                        "score": score,
                        "render_ok": False,
                        "render_reason": render.reason,
                    })

                scores.append(score)
                records.append(record)
                continue

            record["render_ok"] = 1.0

            diag = diagnostic_reward_func(
                cfg=self.cfg,
                errors=render.errors,
                warnings=render.warnings,
                badboxes=render.badboxes,
            )
            record["diagnostic_score"] = float(diag.score)

            try:
                visual = visual_reward_func(
                    cfg=self.cfg,
                    input_image=input_image,
                    rendered_image=render.image,
                )

                visual_score = float(visual.score)

                record["visual_score"] = visual_score
                record["siglip"] = float(visual.siglip)
                record["lpips"] = float(visual.lpips)
                record["dreamsim"] = float(visual.dreamsim)

            except Exception:
                visual_score = 0.0
                record["visual_score"] = 0.0

            code_score = 0.0

            try:
                code = code_reward_func(
                    generated_code=gen_code,
                    reference_code=ref_code,
                    cfg=self.cfg,
                )

                code_score = float(code.score)

                record["code_evaluated"] = 1.0
                record["code_score"] = code_score
                record["crystalbleu"] = float(code.crystalbleu)
                record["ted"] = float(code.ted)

            except Exception:
                record["code_evaluated"] = 0.0
                record["code_score"] = 0.0

            score = self.cfg.renderable_score
            score += diag.score
            score += self.cfg.code_reward_multiplier * code_score
            score += self.cfg.visual_reward_multiplier * visual_score

            score = float(score)
            record["total_score"] = score


            if len(examples) < self.cfg.log_examples_max:
                examples.append({
                    "input_image": input_image,
                    "rendered_image": render.image,
                    "generated_code": gen_code,
                    "reference_code": ref_code,
                    "score": float(score) if "score" in locals() else None,
                    "render_ok": bool(render.ok),
                })


            scores.append(score)
            records.append(record)

        self.last_metrics = self._summarize_records(records)
        self.last_examples = examples

        return scores