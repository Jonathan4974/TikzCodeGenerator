"""VRAM check: load the real model bundle and run a few forward/backward
steps on random tensors of the right shape, before committing to a real training run.

Uses random tensors rather than the real dataset.
"""
from __future__ import annotations

import argparse

import torch

from .config import build_training_config
from .model_loader import SketchAgentModelLoader

NUM_STEPS = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    cfg = build_training_config({"image_size": args.image_size})
    torch.cuda.reset_peak_memory_stats()

    models = SketchAgentModelLoader(cfg).load()
    optimizer = torch.optim.AdamW([p for p in models.unet.parameters() if p.requires_grad], lr=cfg.learning_rate)

    latent_size = cfg.image_size // 8
    for step in range(NUM_STEPS):
        latents = torch.randn(cfg.batch_size, 4, latent_size, latent_size, device="cuda", dtype=models.unet.dtype)
        control_image = torch.rand(
            cfg.batch_size, 3, cfg.image_size, cfg.image_size, device="cuda", dtype=models.controlnet.dtype
        )
        timesteps = torch.randint(
            0, models.noise_scheduler.config.num_train_timesteps, (cfg.batch_size,), device="cuda"
        ).long()
        add_time_ids = torch.tensor(
            [[cfg.image_size, cfg.image_size, 0, 0, cfg.image_size, cfg.image_size]] * cfg.batch_size,
            device="cuda",
            dtype=latents.dtype,
        )
        prompt_embeds = models.prompt_embeds.expand(cfg.batch_size, -1, -1)
        added_cond_kwargs = {
            "text_embeds": models.pooled_prompt_embeds.expand(cfg.batch_size, -1),
            "time_ids": add_time_ids,
        }

        down_block_res_samples, mid_block_res_sample = models.controlnet(
            latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=control_image,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        model_pred = models.unet(
            latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        model_pred.float().pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()

        allocated = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"step {step}: allocated={allocated:.2f}GB peak={peak:.2f}GB")

    print(f"FINAL PEAK VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
