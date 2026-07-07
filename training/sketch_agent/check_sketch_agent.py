"""Manual check: run the trained LoRA weights on a few real held-out SketchFig
examples and print pixel-CC/SigLIP/DreamSim scores. This is more for eyeballing whether the trained model actually works
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _value = _line.partition("=")
        os.environ.setdefault(_key.strip(), _value.strip().strip('"'))

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image, ImageDraw

from evaluation.promptfoo.pf_utils.clip_siglip_metric import image_cosine_similarity
from evaluation.promptfoo.pf_utils.dreamsim_metric import compute_dreamsim_score

from .config import SketchAgentConfig
from .eval import pixel_congruence_coefficient

cfg = SketchAgentConfig()

EVAL_DIR = Path(cfg.sketchfig_cache_dir) / "eval"
CHECK_PREVIEW_DIR = Path(cfg.output_dir) / "check_previews"
TEST_PAIRS = [
    (path, EVAL_DIR / path.name.replace("_input", "_target"))
    for path in sorted(EVAL_DIR.glob("*_input.png"))[:3]
]


def to_canny(image: Image.Image) -> Image.Image:
    arr = cv2.Canny(np.array(image.convert("RGB")), cfg.canny_low_threshold, cfg.canny_high_threshold)
    return Image.fromarray(np.stack([arr] * 3, axis=-1))


LABEL_HEIGHT = 24

def save_comparison(sketch: Image.Image, generated: Image.Image, target: Image.Image, out_path: Path) -> None:
    w, h = sketch.size
    canvas = Image.new("RGB", (w * 3 + 20, h + LABEL_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (img, label) in enumerate([(sketch, "sketch (input)"), (generated, "prediction"), (target, "target")]):
        x = i * (w + 10)
        canvas.paste(img, (x, LABEL_HEIGHT))
        draw.text((x + 4, 4), label, fill="black")
    canvas.save(out_path)


def main() -> None:
    CHECK_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model, torch_dtype=torch.bfloat16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        cfg.base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
    )
    pipe.unet.load_lora_adapter(cfg.lora_output_dir, prefix=None, use_safetensors=True)
    pipe.to("cuda")

    size = (cfg.image_size, cfg.image_size)
    for sketch_path, target_path in TEST_PAIRS:
        sketch = Image.open(sketch_path).convert("RGB").resize(size)
        target = Image.open(target_path).convert("RGB").resize(size)

        generated = pipe(
            prompt=cfg.training_prompt,
            image=to_canny(sketch),
            num_inference_steps=30,
            height=cfg.image_size,
            width=cfg.image_size,
        ).images[0]
        pred_path = CHECK_PREVIEW_DIR / f"{sketch_path.stem}_pred.png"
        generated.save(pred_path)
        sketch.save(CHECK_PREVIEW_DIR / f"{sketch_path.stem}_sketch.png")
        target.save(CHECK_PREVIEW_DIR / f"{sketch_path.stem}_target.png")
        comparison_path = CHECK_PREVIEW_DIR / f"{sketch_path.stem}_comparison.png"
        save_comparison(sketch, generated, target, comparison_path)

        cc = pixel_congruence_coefficient(generated, target)
        siglip = image_cosine_similarity(pred_path, target_path, model_key="siglip")
        dreamsim = compute_dreamsim_score(generated, target)
        print(
            f"{sketch_path.stem}: pixel_cc={cc:.4f} siglip={siglip:.4f} dreamsim={dreamsim:.4f} "
            f"-> {pred_path} ({comparison_path.name})"
        )


if __name__ == "__main__":
    main()
