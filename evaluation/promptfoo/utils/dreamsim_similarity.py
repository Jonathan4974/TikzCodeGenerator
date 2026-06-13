from pathlib import Path
from inspect import signature

import torch
import os
from PIL import Image, ImageChops
import sys


try:
    from dreamsim import dreamsim as dreamsim_fn
except ImportError:  # pragma: no cover
    dreamsim_fn = None


_dreamsim_model = None
_dreamsim_proc = None
_dreamsim_device = None
_dreamsim_dtype = None


def _patch_dino_utils_shadowing():

    utils_module = sys.modules.get("utils")

    if utils_module is not None and not hasattr(utils_module, "trunc_normal_"):
        utils_module.trunc_normal_ = torch.nn.init.trunc_normal_

def _safe_open_rgb(path: str | Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _trim_white_border(
    img: Image.Image,
    *,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    pad: int = 2,
) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")

    bg = Image.new("RGB", img.size, bg_color)
    diff = ImageChops.difference(img, bg).convert("L")
    bbox = diff.getbbox()

    if bbox is None:
        return img

    left, upper, right, lower = bbox

    left = max(0, left - pad)
    upper = max(0, upper - pad)
    right = min(img.width, right + pad)
    lower = min(img.height, lower + pad)

    return img.crop((left, upper, right, lower))


def _pad_to_size(
    img: Image.Image,
    target_w: int,
    target_h: int,
    *,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")

    if img.width == target_w and img.height == target_h:
        return img

    out = Image.new("RGB", (target_w, target_h), fill)
    x = (target_w - img.width) // 2
    y = (target_h - img.height) // 2
    out.paste(img, (x, y))

    return out


def _expand_for_dreamsim(
    img: Image.Image,
    *,
    trim_pad: int = 2,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = _trim_white_border(img, bg_color=bg_color, pad=trim_pad)
    side = max(img.size)

    return _pad_to_size(img, side, side, fill=bg_color)


def _to_chw(t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(type(t))

    if t.dim() == 4 and t.shape[0] == 1:
        return t.squeeze(0)

    if t.dim() == 3:
        return t

    raise ValueError(f"Unexpected dreamsim_proc output shape: {tuple(t.shape)}")


def _filter_supported_kwargs(fn, kwargs: dict) -> dict:
    params = signature(fn).parameters

    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kwargs

    return {k: v for k, v in kwargs.items() if k in params}


def load_dreamsim_model(
    dreamsim_model_name: str = "ensemble",
    pretrained: bool = True,
    normalize: bool = True,
    preprocess: bool = True,
    device: str | None = None,
    cache_dir: str | None = None,
):
    global _dreamsim_model, _dreamsim_proc, _dreamsim_device, _dreamsim_dtype

    if _dreamsim_model is not None:
        return _dreamsim_model, _dreamsim_proc, _dreamsim_device

    if dreamsim_fn is None:
        raise ImportError(
            "The dreamsim package is required to use dreamsim_similarity. "
            "Install it with `pip install dreamsim`."
        )

    _dreamsim_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if _dreamsim_device.type == "cuda":
        try:
            bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        except Exception:
            bf16_ok = False

        _dreamsim_dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        _dreamsim_dtype = torch.float32

    if cache_dir is None:
        cache_dir = os.getenv("DREAMSIM_CACHE_DIR", "/root/.cache/dreamsim")

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    kwargs = {
        "dreamsim_type": dreamsim_model_name,
        "pretrained": pretrained,
        "normalize_embeds": normalize,
        "preprocess": preprocess,
        "device": str(_dreamsim_device),
        "cache_dir": str(cache_dir),
    }

    kwargs = _filter_supported_kwargs(dreamsim_fn, kwargs)

    _patch_dino_utils_shadowing()

    model, proc = dreamsim_fn(**kwargs)

    if _dreamsim_device.type == "cuda":
        try:
            for extractor in getattr(model, "extractor_list", []):
                if hasattr(extractor, "model"):
                    extractor.model = extractor.model.to(_dreamsim_dtype)
                if hasattr(extractor, "proj"):
                    extractor.proj = extractor.proj.to(_dreamsim_dtype)

            model = model.to(_dreamsim_dtype)
        except Exception:
            pass

    try:
        model = model.to(_dreamsim_device)
    except Exception:
        pass

    try:
        model.eval()
    except Exception:
        pass

    _dreamsim_model = model
    _dreamsim_proc = proc

    return _dreamsim_model, _dreamsim_proc, _dreamsim_device


def _preprocess_single_image(proc, img: Image.Image) -> torch.Tensor:
    try:
        out = proc(img)
    except Exception:
        out = proc([img])

    if isinstance(out, dict):
        # Häufige Keys bei Transformern/Processor-ähnlichen APIs
        for key in ("pixel_values", "image", "images"):
            if key in out:
                out = out[key]
                break

    if isinstance(out, torch.Tensor):
        if out.dim() == 5 and out.shape[1] == 1:
            out = out.squeeze(1)

        if out.dim() == 4:
            return out

        if out.dim() == 3:
            return out.unsqueeze(0)

    raise ValueError(
        f"Unexpected DreamSim processor output: type={type(out)}, "
        f"shape={getattr(out, 'shape', None)}"
    )


def compute_dreamsim_distance(
    image_a: str | Path,
    image_b: str | Path,
    trim_pad: int = 2,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> float:
    model, proc, device = load_dreamsim_model()

    img_a = _expand_for_dreamsim(
        _safe_open_rgb(image_a),
        trim_pad=trim_pad,
        bg_color=bg_color,
    )
    img_b = _expand_for_dreamsim(
        _safe_open_rgb(image_b),
        trim_pad=trim_pad,
        bg_color=bg_color,
    )

    gt_t = _preprocess_single_image(proc, img_a)
    pr_t = _preprocess_single_image(proc, img_b)

    if not (isinstance(gt_t, torch.Tensor) and gt_t.dim() == 4):
        raise ValueError(
            f"Unexpected batched gt_t shape/type: {type(gt_t)} "
            f"{getattr(gt_t, 'shape', None)}"
        )

    if not (isinstance(pr_t, torch.Tensor) and pr_t.dim() == 4):
        raise ValueError(
            f"Unexpected batched pr_t shape/type: {type(pr_t)} "
            f"{getattr(pr_t, 'shape', None)}"
        )

    dtype = _dreamsim_dtype if device.type == "cuda" else torch.float32

    gt_t = gt_t.to(device, dtype=dtype)
    pr_t = pr_t.to(device, dtype=dtype)

    with torch.inference_mode():
        dist = model(gt_t, pr_t)

    if isinstance(dist, torch.Tensor):
        distance = dist.detach().float().view(-1).cpu().tolist()
    else:
        distance = [float(dist)]

    return float(distance[0])


def dreamsim_distance_to_similarity(distance: float) -> float:
    similarity = 1.0 - float(distance)

    if similarity < 0.0:
        similarity = 0.0
    elif similarity > 1.0:
        similarity = 1.0

    return similarity