import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback


def pil_to_np(img):
    return np.asarray(img.convert("RGB"))


class RewardTensorBoardCallback(TrainerCallback):
    def __init__(self, reward_fn, log_dir, every_n_steps=10):
        self.reward_fn = reward_fn
        self.writer = SummaryWriter(log_dir)
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps != 0:
            return

        sample = getattr(self.reward_fn, "last_sample", None)
        if sample is None:
            return

        step = state.global_step

        self.writer.add_scalar("reward/sample_score", sample["score"], step)

        self.writer.add_image(
            "reward/input_image",
            pil_to_np(sample["input_image"]),
            step,
            dataformats="HWC",
        )

        self.writer.add_image(
            "reward/generated_image",
            pil_to_np(sample["generated_image"]),
            step,
            dataformats="HWC",
        )

        self.writer.add_text(
            "reward/generated_code",
            "```latex\n" + sample["generated_code"][:3000] + "\n```",
            step,
        )

        self.writer.add_text(
            "reward/reference_code",
            "```latex\n" + sample["reference_code"][:3000] + "\n```",
            step,
        )

        self.writer.flush()