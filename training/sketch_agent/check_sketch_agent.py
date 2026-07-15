"""Manual check: run the trained LoRA weights on a few real held-out SketchFig
examples and print pixel-CC/SigLIP/DreamSim scores. This is more for eyeballing whether the trained model actually works
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _value = _line.partition("=")
        os.environ.setdefault(_key.strip(), _value.strip().strip('"'))

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image, ImageDraw

from evaluation.promptfoo.pf_utils.clip_siglip_metric import image_cosine_similarity
from evaluation.promptfoo.pf_utils.dreamsim_metric import compute_dreamsim_score

from .checkpoint_utils import resolve_latest_dir
from .config import SketchAgentConfig
from .data import prepare_conditioning_image
from .eval import pixel_congruence_coefficient

cfg = SketchAgentConfig()

EVAL_DIR = Path(cfg.sketchfig_cache_dir) / "eval"
TEST_PAIRS = [
    (path, EVAL_DIR / path.name.replace("_input", "_target"))
    for path in sorted(EVAL_DIR.glob("*_input.png"))[:10]
]


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


def _pick_latest_run_dir(lora_root: Path) -> Path:
    """No --run-name given: fall back to whichever run subdir was touched most recently"""
    run_dirs = [p for p in lora_root.iterdir() if p.is_dir()] if lora_root.exists() else []
    if not run_dirs:
        raise FileNotFoundError(f"no run directories found under {lora_root}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _resolve_lora_dir(lora_dir_arg: Optional[str], run_name_arg: Optional[str]) -> tuple[Path, str]:
    """Returns (step_dir, run_name) - run_name is used to namespace check_previews & output"""
    if lora_dir_arg is not None:
        step_dir = Path(lora_dir_arg)
        return step_dir, step_dir.parent.name

    lora_root = Path(cfg.lora_output_dir)
    run_dir = (lora_root / run_name_arg) if run_name_arg else _pick_latest_run_dir(lora_root)
    resolved = resolve_latest_dir(run_dir, "latest_lora.json")
    if resolved is None:
        raise FileNotFoundError(f"no step_* LoRA export found under {run_dir}")
    return resolved[0], run_dir.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora-dir",
        type=str,
        default=None,
        help="dir with pytorch_lora_weights.safetensors; defaults to the latest step_* export in the resolved run (see --run-name)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="which run's export to use under; defaults to the most recently modified run dir",
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="skip loading any LoRA adapter; run the (frozen) base SDXL+ControlNet pipeline as a baseline",
    )
    parser.add_argument("--tag", type=str, default=None, help="label for the leaf subfolder under check_previews/<run>/; defaults to the step name (or --image-size for --zero-shot)")
    parser.add_argument("--image-size", type=int, default=None, help="overrides cfg.image_size for this run")
    args = parser.parse_args()

    image_size = args.image_size or cfg.image_size
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.zero_shot:
        lora_dir = None
        run_label, step_label = "zero_shot", args.tag or str(image_size)
    else:
        lora_dir, run_name = _resolve_lora_dir(args.lora_dir, args.run_name)
        run_label, step_label = run_name, args.tag or lora_dir.name
    check_preview_dir = Path(cfg.output_dir) / "check_previews" / run_label / step_label
    check_preview_dir.mkdir(parents=True, exist_ok=True)

    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model, torch_dtype=torch.bfloat16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        cfg.base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
    )
    if not args.zero_shot:
        pipe.unet.load_lora_adapter(str(lora_dir), prefix=None, use_safetensors=True)
    pipe.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(cfg.seed)

    size = (image_size, image_size)
    for sketch_path, target_path in TEST_PAIRS:
        sketch = Image.open(sketch_path).convert("RGB").resize(size)
        target = Image.open(target_path).convert("RGB").resize(size)

        generated = pipe(
            prompt=cfg.positive_prompt,
            negative_prompt=cfg.negative_prompt,
            image=prepare_conditioning_image(sketch, cfg),
            controlnet_conditioning_scale=cfg.controlnet_conditioning_scale,
            guidance_scale=cfg.guidance_scale,
            generator=generator,
            num_inference_steps=30,
            height=image_size,
            width=image_size,
        ).images[0]
        pred_path = check_preview_dir / f"{sketch_path.stem}_pred.png"
        generated.save(pred_path)
        sketch.save(check_preview_dir / f"{sketch_path.stem}_sketch.png")
        target.save(check_preview_dir / f"{sketch_path.stem}_target.png")
        comparison_path = check_preview_dir / f"{sketch_path.stem}_comparison.png"
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
