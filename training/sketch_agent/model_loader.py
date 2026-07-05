from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from .config import SketchAgentConfig

LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]


@dataclass
class SketchAgentModels:
    unet: UNet2DConditionModel
    controlnet: ControlNetModel
    vae: AutoencoderKL
    text_encoder: CLIPTextModel
    text_encoder_2: CLIPTextModelWithProjection
    tokenizer: CLIPTokenizer
    tokenizer_2: CLIPTokenizer
    noise_scheduler: DDPMScheduler
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor


class SketchAgentModelLoader:
    """Loads SDXL UNet + text encoders + VAE + the canny ControlNet, freezes everything
    except a LoRA adapter on the UNet's attention projections."""

    def __init__(self, cfg: SketchAgentConfig):
        self.cfg = cfg

    def load(self) -> SketchAgentModels:
        cfg = self.cfg
        dtype = torch.bfloat16

        tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer_2")
        text_encoder = CLIPTextModel.from_pretrained(cfg.base_model, subfolder="text_encoder", torch_dtype=dtype).to("cuda")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            cfg.base_model, subfolder="text_encoder_2", torch_dtype=dtype
        ).to("cuda")
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)

        prompt_embeds, pooled_prompt_embeds = self._encode_fixed_prompt(
            tokenizer, tokenizer_2, text_encoder, text_encoder_2, cfg.training_prompt
        )

        vae = AutoencoderKL.from_pretrained(cfg.vae_model, torch_dtype=dtype).to("cuda")
        vae.requires_grad_(False)

        unet = UNet2DConditionModel.from_pretrained(cfg.base_model, subfolder="unet", torch_dtype=dtype).to("cuda")
        unet.requires_grad_(False)
        unet.add_adapter(
            LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                init_lora_weights="gaussian",
                target_modules=LORA_TARGET_MODULES,
            )
        )
        cast_training_params(unet, dtype=torch.float32)

        controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model, torch_dtype=dtype).to("cuda")
        controlnet.requires_grad_(False)

        noise_scheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")

        return SketchAgentModels(
            unet=unet,
            controlnet=controlnet,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            noise_scheduler=noise_scheduler,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

    @staticmethod
    def _encode_fixed_prompt(tokenizer, tokenizer_2, text_encoder, text_encoder_2, prompt: str):
        """SDXL's dual-text-encoder prompt embedding: hidden_states[-2] from each encoder
        concatenated for cross-attention, plus text_encoder_2's pooled output."""
        embeds = []
        pooled_prompt_embeds = None
        for tokenizer_, text_encoder_ in ((tokenizer, text_encoder), (tokenizer_2, text_encoder_2)):
            input_ids = tokenizer_(
                prompt,
                padding="max_length",
                max_length=tokenizer_.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to("cuda")
            with torch.no_grad():
                output = text_encoder_(input_ids, output_hidden_states=True)
            pooled_prompt_embeds = output[0]
            embeds.append(output.hidden_states[-2])
        prompt_embeds = torch.concat(embeds, dim=-1)
        return prompt_embeds, pooled_prompt_embeds
