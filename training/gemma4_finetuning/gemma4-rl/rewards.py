import re
from collections import OrderedDict

from reward_functions.render_reward import is_renderable
from reward_functions.diagnostic_reward import diagnostic_reward_func
from reward_functions.visual_reward import visual_reward_func
from reward_functions.code_reward import code_reward_func


def clean_code(completion) -> str:
    if isinstance(completion, list):
        completion = completion[0]["content"] if completion else ""

    if isinstance(completion, dict):
        completion = completion.get("content", "")

    code = str(completion).strip()

    return code.strip()


def pick(values, idx):
    if values is None:
        return None
    if isinstance(values, list):
        return values[idx % len(values)]
    return values


class TikZReward:
    def __init__(self, cfg):
        self.cfg = cfg
        self.__name__ = "tikz_reward"

    def __call__(self, completions, answer=None, image=None, images=None, **kwargs):
        scores = []

        input_images = image or images or kwargs.get("image") or kwargs.get("images")
        reference_codes = answer or kwargs.get("answer")

        for i, completion in enumerate(completions):
            gen_code = clean_code(completion)
            ref_code = pick(reference_codes, i)
            input_image = pick(input_images, i)

            render = is_renderable(gen_code)

            if not render.ok:
                score = float(self.cfg.not_renderable_score)
                scores.append(score)
                continue

            diag = diagnostic_reward_func(
                cfg=self.cfg,
                errors=render.errors,
                warnings=render.warnings,
                badboxes=render.badboxes,
            )

            visual = visual_reward_func(
                cfg=self.cfg,
                input_image=input_image,
                rendered_image=render.image,
            )

            score = self.cfg.renderable_score
            score += diag.score
            score += self.cfg.visual_reward_multiplier * visual.score

            if visual.score >= self.cfg.visual_threshold:
                code = code_reward_func(
                    generated_code=gen_code,
                    reference_code=ref_code,
                    cfg=self.cfg,
                )
                score += code.score

            score = float(score)


            scores.append(score)

        return scores