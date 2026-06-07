from pathlib import Path

import torch
from PIL import Image, ImageChops

try:
    from dreamsim import dreamsim as dreamsim_fn
except ImportError:  # pragma: no cover
    dreamsim_fn = None

_dreamsim_model = None
_dreamsim_proc = None
_dreamsim_device = None
_dreamsim_dtype = None


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


def load_dreamsim_model(
    dreamsim_model_name: str = "ensemble",
    pretrained: bool = True,
    normalize: bool = True,
    preprocess: bool = True,
    device: str | None = None,
    cache_dir: str | None = None,
):
    """Load and cache the DreamSim model and processor."""
    global _dreamsim_model, _dreamsim_proc, _dreamsim_device, _dreamsim_dtype

    if _dreamsim_model is None:
        if dreamsim_fn is None:
            raise ImportError(
                "The dreamsim package is required to use dreamsim_similarity. "
                "Install it with `pip install dreamsim`."
            )

        _dreamsim_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        if _dreamsim_device.type == "cuda":
            try:
                bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            except Exception:
                bf16_ok = False
            _dreamsim_dtype = torch.bfloat16 if bf16_ok else torch.float16
        else:
            _dreamsim_dtype = torch.float32

        model, proc = dreamsim_fn(
            dreamsim_type=dreamsim_model_name,
            pretrained=pretrained,
            normalize_embeds=normalize,
            preprocess=preprocess,
            device=str(_dreamsim_device),
            cache_dir=cache_dir,
        )

        try:
            for extractor in getattr(model, "extractor_list", []):
                if hasattr(extractor, "model"):
                    extractor.model = extractor.model.to(_dreamsim_dtype)
                if hasattr(extractor, "proj"):
                    extractor.proj = extractor.proj.to(_dreamsim_dtype)
        except Exception:
            pass

        _dreamsim_model = model.to(_dreamsim_dtype)
        _dreamsim_proc = proc
        try:
            _dreamsim_model.eval()
        except Exception:
            pass

    return _dreamsim_model, _dreamsim_proc, _dreamsim_device


def compute_dreamsim_distance(
    image_a: str | Path,
    image_b: str | Path,
    trim_pad: int = 2,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> float:
    """Compute the DreamSim distance between two images."""
    model, proc, device = load_dreamsim_model()

    img_a = _expand_for_dreamsim(_safe_open_rgb(image_a), trim_pad=trim_pad, bg_color=bg_color)
    img_b = _expand_for_dreamsim(_safe_open_rgb(image_b), trim_pad=trim_pad, bg_color=bg_color)

    try:
        gt_list = [proc(img_a)]
        pr_list = [proc(img_b)]

        if isinstance(gt_list[0], dict) or isinstance(pr_list[0], dict):
            raise TypeError("dreamsim_proc returned dict; use batched mode")

        gt_t = torch.stack([_to_chw(t) for t in gt_list], dim=0)
        pr_t = torch.stack([_to_chw(t) for t in pr_list], dim=0)
    except Exception:
        gt_t = proc([img_a])
        pr_t = proc([img_b])

        if isinstance(gt_t, torch.Tensor) and gt_t.dim() == 5 and gt_t.shape[1] == 1:
            gt_t = gt_t.squeeze(1)
        if isinstance(pr_t, torch.Tensor) and pr_t.dim() == 5 and pr_t.shape[1] == 1:
            pr_t = pr_t.squeeze(1)

    if not (isinstance(gt_t, torch.Tensor) and gt_t.dim() == 4):
        raise ValueError(f"Unexpected batched gt_t shape/type: {type(gt_t)} {getattr(gt_t, 'shape', None)}")
    if not (isinstance(pr_t, torch.Tensor) and pr_t.dim() == 4):
        raise ValueError(f"Unexpected batched pr_t shape/type: {type(pr_t)} {getattr(pr_t, 'shape', None)}")

    gt_t = gt_t.to(device, dtype=_dreamsim_dtype)
    pr_t = pr_t.to(device, dtype=_dreamsim_dtype)

    with torch.inference_mode():
        dist = model(gt_t, pr_t)

    if isinstance(dist, torch.Tensor):
        distance = dist.detach().float().view(-1).cpu().tolist()
    else:
        distance = [float(dist)]

    return float(distance[0])


def dreamsim_distance_to_similarity(distance: float) -> float:
    """Convert DreamSim distance to a 0-1 similarity score."""
    similarity = 1.0 - distance
    if similarity < 0.0:
        similarity = 0.0
    elif similarity > 1.0:
        similarity = 1.0
    return similarity
