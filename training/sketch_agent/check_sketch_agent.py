"""Manual check: run the trained LoRA weights on a few real held-out SketchFig
examples and print pixel-CC/SigLIP/DreamSim scores. This is more for eyeballing whether the trained model actually works
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image

from evaluation.promptfoo.pf_utils.clip_siglip_metric import image_cosine_similarity
from evaluation.promptfoo.pf_utils.dreamsim_metric import compute_dreamsim_score

from .config import SketchAgentConfig
from .eval import pixel_congruence_coefficient

cfg = SketchAgentConfig()

EVAL_DIR = Path(cfg.sketchfig_cache_dir) / "eval"
TEST_PAIRS = [
    (path, EVAL_DIR / path.name.replace("_input", "_target"))
    for path in sorted(EVAL_DIR.glob("*_input.png"))[:3]
]


def to_canny(image: Image.Image) -> Image.Image:
    arr = cv2.Canny(np.array(image.convert("RGB")), cfg.canny_low_threshold, cfg.canny_high_threshold)
    return Image.fromarray(np.stack([arr] * 3, axis=-1))


def main() -> None:
    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model, torch_dtype=torch.bfloat16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        cfg.base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights(cfg.lora_output_dir)
    pipe.to("cuda")

    for sketch_path, target_path in TEST_PAIRS:
        sketch = Image.open(sketch_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        generated = pipe(prompt=cfg.training_prompt, image=to_canny(sketch), num_inference_steps=30).images[0]
        pred_path = Path(f"/tmp/sketch_agent_pred_{sketch_path.stem}.png")
        generated.save(pred_path)

        cc = pixel_congruence_coefficient(generated, target)
        siglip = image_cosine_similarity(pred_path, target_path, model_key="siglip")
        dreamsim = compute_dreamsim_score(generated, target)
        print(f"{sketch_path.stem}: pixel_cc={cc:.4f} siglip={siglip:.4f} dreamsim={dreamsim:.4f} -> {pred_path}")


if __name__ == "__main__":
    main()
