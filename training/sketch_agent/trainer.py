from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import EulerDiscreteScheduler, StableDiffusionXLControlNetPipeline
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from evaluation.promptfoo.pf_utils.clip_siglip_metric import image_cosine_similarity
from evaluation.promptfoo.pf_utils.dreamsim_metric import compute_dreamsim_score

from .checkpoint_utils import clear_resumable_state, resolve_latest_dir, resolve_run_name, write_latest_manifest
from .config import SketchAgentConfig
from .data import SyntheticPair, prepare_conditioning_image
from .eval import pixel_congruence_coefficient
from .model_loader import SketchAgentModels


def _resolve_self_resubmit_command(cfg: SketchAgentConfig, is_main_process: bool, job_id: Optional[str]) -> Optional[List[str]]:
    """Decision logic for whether/how to self-resubmit"""
    if not cfg.self_resubmit or not is_main_process or not job_id:
        return None
    script = cfg.sbatch_script or str(Path(__file__).with_name("train.sbatch"))
    return ["sbatch", "--dependency=afterany:" + job_id, script]


class SketchAgentTrainer:
    def __init__(
        self,
        cfg: SketchAgentConfig,
        models: SketchAgentModels,
        train_dataset: torch.utils.data.Dataset,
        eval_pairs: Optional[List[SyntheticPair]] = None,
    ):
        self.cfg = cfg
        self.eval_pairs = eval_pairs or []

        self.accelerator = Accelerator(
            mixed_precision=cfg.mixed_precision,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        )

        dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader_num_workers,
            drop_last=True,
        )
        params = [p for p in models.unet.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)
        # AcceleratedScheduler only advances the wrapped scheduler on true gradient sync steps (once per gradient_accumulation_steps)
        real_steps = max(1, cfg.max_steps // cfg.gradient_accumulation_steps)
        lr_scheduler = get_scheduler(
            cfg.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(real_steps * cfg.lr_warmup_ratio),
            num_training_steps=real_steps,
        )
        self.unet, self.optimizer, self.lr_scheduler, self.dataloader = self.accelerator.prepare(
            models.unet, optimizer, lr_scheduler, dataloader
        )

        self.controlnet = models.controlnet
        self.vae = models.vae
        self.noise_scheduler = models.noise_scheduler
        self.prompt_embeds = models.prompt_embeds.to(self.accelerator.device)
        self.pooled_prompt_embeds = models.pooled_prompt_embeds.to(self.accelerator.device)

        if self.accelerator.is_main_process:
            self.run_name = resolve_run_name(Path(cfg.output_dir), Path(cfg.checkpoint_dir), cfg.run_name)
            self.writer = SummaryWriter(f"{cfg.output_dir}/tensorboard/{self.run_name}")
        else:
            self.run_name = None
            self.writer = None

        self.eval_pipeline = StableDiffusionXLControlNetPipeline(
            vae=self.vae,
            text_encoder=models.text_encoder,
            text_encoder_2=models.text_encoder_2,
            tokenizer=models.tokenizer,
            tokenizer_2=models.tokenizer_2,
            unet=self.accelerator.unwrap_model(self.unet),
            controlnet=self.controlnet,
            scheduler=EulerDiscreteScheduler.from_config(self.noise_scheduler.config),
        ).to(self.accelerator.device)

        self.start_time = time.monotonic()

    def _training_step(self, batch: dict) -> torch.Tensor:
        pixel_values = batch["pixel_values"].to(dtype=self.vae.dtype)
        control_image = batch["conditioning_pixel_values"].to(dtype=self.controlnet.dtype)

        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample() * self.vae.config.scaling_factor
        latents = latents.to(dtype=self.unet.dtype)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        add_time_ids = torch.cat(
            [batch["original_size"], batch["crop_top_left"], batch["target_size"]], dim=1
        ).to(latents.dtype)
        prompt_embeds = self.prompt_embeds.expand(bsz, -1, -1)
        added_cond_kwargs = {
            "text_embeds": self.pooled_prompt_embeds.expand(bsz, -1),
            "time_ids": add_time_ids,
        }

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=control_image,
            conditioning_scale=self.cfg.controlnet_conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"unsupported prediction_type: {prediction_type}")

        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    def _save_checkpoint(self, step: int) -> None:
        if not self.cfg.save_checkpoints:
            return
        checkpoint_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir = checkpoint_dir / f"step_{step:06d}"
        self.accelerator.save_state(str(ckpt_dir))
        if self.accelerator.is_main_process:
            write_latest_manifest(checkpoint_dir, "latest_checkpoint.json", step, ckpt_dir)

    def _load_checkpoint(self) -> int:
        checkpoint_dir = Path(self.cfg.checkpoint_dir)
        if not checkpoint_dir.exists():
            return 0
        resolved = resolve_latest_dir(checkpoint_dir, "latest_checkpoint.json")
        if resolved is None:
            return 0
        ckpt_dir, step = resolved
        self.accelerator.load_state(str(ckpt_dir))
        print(f"Resuming from checkpoint at step {step}")
        return step

    def _save_lora_export(self, step: int) -> None:
        if not self.accelerator.is_main_process:
            return
        lora_dir = Path(self.cfg.lora_output_dir) / self.run_name
        step_dir = lora_dir / f"step_{step:06d}"
        self.accelerator.unwrap_model(self.unet).save_lora_adapter(str(step_dir))
        write_latest_manifest(lora_dir, "latest_lora.json", step, step_dir)
        print(f"Saved LoRA export at step {step} -> {step_dir}")

    def _run_eval(self, step: int) -> None:
        if not self.accelerator.is_main_process or not self.eval_pairs:
            return

        self.unet.eval()
        preview_dir = Path(self.cfg.output_dir) / "eval_previews" / self.run_name / f"step_{step:06d}"
        preview_dir.mkdir(parents=True, exist_ok=True)

        size = (self.cfg.image_size, self.cfg.image_size)
        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.cfg.seed)
        scores: dict[str, list[float]] = {metric: [] for metric in self.cfg.eval_metrics}
        for i, pair in enumerate(self.eval_pairs[: self.cfg.eval_sample_size]):
            sketch = Image.open(pair.input_path).convert("RGB").resize(size)
            target = Image.open(pair.target_path).convert("RGB").resize(size)
            with self.accelerator.autocast():
                generated = self.eval_pipeline(
                    prompt=self.cfg.positive_prompt,
                    negative_prompt=self.cfg.negative_prompt,
                    image=prepare_conditioning_image(sketch, self.cfg),
                    controlnet_conditioning_scale=self.cfg.controlnet_conditioning_scale,
                    guidance_scale=self.cfg.guidance_scale,
                    generator=generator,
                    num_inference_steps=20,
                    height=self.cfg.image_size,
                    width=self.cfg.image_size,
                ).images[0]
            pred_path = preview_dir / f"{i}.png"
            generated.save(pred_path)

            if "pixel_cc" in scores:
                scores["pixel_cc"].append(pixel_congruence_coefficient(generated, target))
            if "siglip" in scores:
                scores["siglip"].append(image_cosine_similarity(pred_path, pair.target_path, model_key="siglip"))
            if "dreamsim" in scores:
                scores["dreamsim"].append(compute_dreamsim_score(generated, target))

        for metric, values in scores.items():
            if values and self.writer is not None:
                self.writer.add_scalar(f"eval/{metric}", float(np.mean(values)), step)

        self.unet.train()

    def _time_limit_reached(self) -> bool:
        return (time.monotonic() - self.start_time) / 3600 >= self.cfg.time_limit_hours

    def _maybe_self_resubmit(self) -> Optional[str]:
        command = _resolve_self_resubmit_command(
            self.cfg, self.accelerator.is_main_process, os.environ.get("SLURM_JOB_ID")
        )
        if command is None:
            return None
        subprocess.run(command, check=False)
        return " ".join(command)

    def _cleanup_after_completion(self) -> None:
        if not self.accelerator.is_main_process:
            return
        run_name_file = Path(self.cfg.output_dir) / "current_run_name.txt"
        clear_resumable_state(Path(self.cfg.checkpoint_dir), run_name_file)

    def train(self) -> None:
        step = self._load_checkpoint()
        data_iter = iter(self.dataloader)

        while step < self.cfg.max_steps:
            batch = next(data_iter, None)
            if batch is None:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            with self.accelerator.accumulate(self.unet):
                loss = self._training_step(batch)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            step += 1
            if self.writer is not None:
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/lr", self.lr_scheduler.get_last_lr()[0], step)

            if step % self.cfg.checkpoint_interval_steps == 0 or step == self.cfg.max_steps:
                self._save_checkpoint(step)
                self._save_lora_export(step)
                self._run_eval(step)

            if self._time_limit_reached():
                self._save_checkpoint(step)
                self._save_lora_export(step)
                self._maybe_self_resubmit()
                return

        self._cleanup_after_completion()
