from functools import cached_property
from typing import List

from PIL import Image
from dreamsim import dreamsim
from huggingface_hub import cached_assets_path
import torch
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
from torchmetrics import Metric

from torch.cuda import is_available as is_torch_cuda_available
from transformers.utils import is_torch_npu_available, is_torch_xpu_available

from base64 import b64decode
from codecs import encode
from io import BytesIO
from os.path import isfile

from PIL import Image, ImageChops, ImageOps
import pymupdf
import requests

#######https://github.com/potamides/DeTikZify/blob/main/detikzify/util/torch.py#########################

# https://github.com/huggingface/peft/blob/c4cf9e7d3b2948e71ec65a19e6cd1ff230781d13/src/peft/utils/other.py#L60-L71
def infer_device():
    if is_torch_cuda_available():
        torch_device = "cuda"
    elif is_torch_xpu_available():
        torch_device = "xpu"
    elif is_torch_npu_available():
        torch_device = "npu"
    else:
        torch_device = "cpu"
    return torch_device

#######https://github.com/potamides/DeTikZify/blob/main/detikzify/util/image.py#########################

DUMMY_IMAGE = Image.new("RGB", (24, 24), color="white")

def convert(image, filetype):
    image.save(imgbytes:=BytesIO(), format=filetype)
    return Image.open(imgbytes)

def remove_alpha(image, bg):
    # https://stackoverflow.com/a/62414364
    background = Image.new('RGBA', image.size, bg)
    alpha_composite = Image.alpha_composite(background, image.convert("RGBA"))
    return alpha_composite.convert("RGB")

# https://stackoverflow.com/a/10616717
def trim(image, bg="white"):
    bg = Image.new(image.mode, image.size, bg)
    diff = ImageChops.difference(image, bg)
    #diff = ImageChops.add(diff, diff, 2.0, -10)
    return image.crop(bbox) if (bbox:=diff.getbbox()) else image

def expand(image, size, do_trim=False, bg="white"):
    """Expand image to a square of size {size}. Optionally trims borders first."""
    image = trim(image, bg=bg) if do_trim else image
    return ImageOps.pad(image, (size, size), color=bg, method=Image.Resampling.LANCZOS)

#  based on transformers/image_utils.py (added support for rgba images)
def load(image: Image.Image | str | bytes, bg="white", timeout=None):
    if isinstance(image, bytes):
        # assume image bytes and open
        image = Image.open(BytesIO(image))
    elif isinstance(image, str):
        if isfile(image):
            image = Image.open(image)
        else:
            try:
                image.removeprefix("data:image/")
                image = Image.open(BytesIO(b64decode(image)))
            except Exception as e:
                raise ValueError(
                    "Incorrect image source. "
                    "Must be a valid URL starting with `http://` or `https://`, "
                    "a valid path to an image file, bytes, or a base64 encoded "
                    f"string. Got {image}. Failed with {e}"
                )

    image = ImageOps.exif_transpose(image) # type: ignore
    return  remove_alpha(image, bg=bg)

def redact(doc, rot_13=False):
    for page in (copy:=pymupdf.open("pdf", doc.tobytes())):
        for word in page.get_text("words", clip=pymupdf.INFINITE_RECT()): # type: ignore
            text = encode(word[4], "rot13") if rot_13 else None
            page.add_redact_annot(word[:4], text=text, fill=False) # type: ignore
        page.apply_redactions(  # type: ignore
            images=pymupdf.PDF_REDACT_IMAGE_NONE, # type: ignore
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE # type: ignore
        )
    return copy


#######https://github.com/potamides/DeTikZify/blob/main/detikzify/evaluate/dreamsim.py#########################


class DreamSim(Metric):
    """Perceptual image similarity using DreamSim"""

    higher_is_better = True

    def __init__(
        self,
        model_name: str = "ensemble",
        pretrained: bool = True,
        normalize: bool = True,
        preprocess: bool = True,
        device: str = infer_device(),
        dtype=torch.bfloat16 if is_cuda_available() and is_bf16_supported() else torch.float16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.pretrained = pretrained
        self.normalize = normalize
        self._device = device
        self.set_dtype(dtype)
        self.preprocess = preprocess

        self.add_state("score", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def __str__(self):
        return self.__class__.__name__

    @cached_property
    def dreamsim(self):
        model, processor = dreamsim(
            dreamsim_type=self.model_name,
            pretrained = self.pretrained,
            normalize_embeds=self.normalize,
            device=str(self.device),
            cache_dir=str(cached_assets_path(library_name="evaluate", namespace=self.__class__.__name__.lower()))
        )
        for extractor in model.extractor_list:
            extractor.model = extractor.model.to(self.dtype)
            extractor.proj = extractor.proj.to(self.dtype)
        return dict(
            model=model.to(self.dtype),
            processor=processor
        )

    @property
    def model(self):
        return self.dreamsim['model']

    @property
    def processor(self):
        return self.dreamsim['processor']

    def update(
        self,
        img1: Image.Image | str | List[Image.Image | str],
        img2: Image.Image | str | List[Image.Image | str],
    ):
        if isinstance(img1, List) or isinstance(img2, List):
            assert type(img1) == type(img2) and len(img1) == len(img2) # type: ignore
        else:
            img1, img2 = [img1], [img2]

        for i1, i2 in zip(img1, img2): # type: ignore
            i1, i2 = load(i1), load(i2)
            if self.preprocess:
                i1 = expand(load(i1), max(i1.size), do_trim=True)
                i2 = expand(load(i2), max(i2.size), do_trim=True)
            i1 = self.processor(i1).to(self.device, self.dtype)
            i2 = self.processor(i2).to(self.device, self.dtype)
            with torch.inference_mode():
                self.score += 1 - self.model(i1, i2).item() # type: ignore
            self.n_samples += 1

    def compute(self):
        return (self.score / self.n_samples).item()



def compute_dreamsim_score(
    image_a,
    image_b,
    model_name: str = "ensemble",
    pretrained: bool = True,
    normalize: bool = True,
    preprocess: bool = True,
):
    metric = DreamSim(
        model_name=model_name,
        pretrained=pretrained,
        normalize=normalize,
        preprocess=preprocess,
    )

    metric.update(
        img1=image_a,
        img2=image_b,
    )

    return float(metric.compute())