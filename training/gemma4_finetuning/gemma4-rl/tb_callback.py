import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
from unsloth import FastVisionModel

from rewards import clean_code
from reward_functions.render_reward import is_renderable


def pil_to_np(img):
    return np.asarray(img.convert("RGB"))


class TensorBoardNumberCallback(TrainerCallback):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}

        for key, value in logs.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            self.writer.add_scalar(key, value, state.global_step)

        self.writer.flush()