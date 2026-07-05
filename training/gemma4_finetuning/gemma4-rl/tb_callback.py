import numpy as np
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback


def pil_to_hwc(img: Image.Image):
    return np.asarray(img.convert("RGB"))


def make_side_by_side(left: Image.Image, right: Image.Image | None):
    left = left.convert("RGB")

    if right is None:
        right = Image.new("RGB", left.size, "white")
        draw = ImageDraw.Draw(right)
        draw.text((10, 10), "not renderable", fill=(0, 0, 0))
    else:
        right = right.convert("RGB").resize(left.size)

    canvas = Image.new("RGB", (left.width + right.width, left.height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))

    return canvas


def truncate_text(text, max_chars=8000):
    text = str(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n\n...[truncated]..."


class TensorBoardNumberCallback(TrainerCallback):
    def __init__(self, log_dir, reward_fn=None):
        self.writer = SummaryWriter(log_dir)
        self.reward_fn = reward_fn

    def on_log(self, args, state, control, logs=None, **kwargs):
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return

        step = state.global_step
        logs = logs or {}

        for key, value in logs.items():
            try:
                self.writer.add_scalar(key, float(value), step)
            except (TypeError, ValueError):
                continue

        if self.reward_fn is not None:
            self._log_reward_metrics(step)
            self._log_reward_examples(step)

        self.writer.flush()

    def _log_reward_metrics(self, step):
        reward_metrics = getattr(self.reward_fn, "last_metrics", {})

        for key, value in reward_metrics.items():
            try:
                self.writer.add_scalar(key, float(value), step)
            except (TypeError, ValueError):
                continue

    def _log_reward_examples(self, step):
        cfg = getattr(self.reward_fn, "cfg", None)
        if cfg is None:
            return

        log_every = getattr(cfg, "log_examples_every", 10)
        max_chars = getattr(cfg, "max_logged_code_chars", 8000)

        if log_every <= 0:
            return

        if step % log_every != 0:
            return

        examples = getattr(self.reward_fn, "last_examples", [])

        for i, ex in enumerate(examples):
            input_image = ex.get("input_image")
            rendered_image = ex.get("rendered_image")

            if input_image is not None:
                self.writer.add_image(
                    f"samples/{i}/original",
                    pil_to_hwc(input_image),
                    step,
                    dataformats="HWC",
                )

            if rendered_image is not None:
                self.writer.add_image(
                    f"samples/{i}/generated_render",
                    pil_to_hwc(rendered_image),
                    step,
                    dataformats="HWC",
                )

            if input_image is not None:
                side_by_side = make_side_by_side(input_image, rendered_image)
                self.writer.add_image(
                    f"samples/{i}/original_vs_generated",
                    pil_to_hwc(side_by_side),
                    step,
                    dataformats="HWC",
                )

            reason = truncate_text(ex.get("render_reason", ""), 2000)
            generated = truncate_text(ex.get("generated_code", ""), max_chars)
            reference = truncate_text(ex.get("reference_code", ""), max_chars)

            text = (
                f"Score: {ex.get('score')}\n"
                f"Render OK: {ex.get('render_ok')}\n\n"
                f"Render reason:\n{reason}\n\n"
                f"Generated code:\n{generated}\n\n"
                f"Reference code:\n{reference}"
            )

            self.writer.add_text(
                f"samples/step_{step:06d}/sample_{i}/code",
                text,
                step,
            )