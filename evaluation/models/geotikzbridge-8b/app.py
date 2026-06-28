from io import BytesIO
from typing import Optional
import os
import re

import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "NONE").strip('"')
USE_8BIT = os.getenv("USE_8BIT", "false").lower() == "true"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "0.95"))

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "448"))

app = FastAPI(title="GeoTikzBridge API")

model = None
tokenizer = None
device: Optional[torch.device] = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_quantization_config():
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=[
            "vision_tower",
            "vision_model",
            "visual",
            "multi_modal_projector",
            "mm_projector",
            "projector",
            "image_projector",
            "connector",
        ],
    )


def image_to_pixel_values(image: Image.Image):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return torch.stack([transform(image)])


def extract_tikz_code(text: str) -> str:
    text = text.strip()

    preferred_block = re.search(
        r"```(?:tikz|tex|latex)\s*\n?(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if preferred_block:
        return preferred_block.group(1).strip()

    any_block = re.search(
        r"```\s*\n?(.*?)```",
        text,
        flags=re.DOTALL,
    )
    if any_block:
        return any_block.group(1).strip()

    return text


@app.on_event("startup")
def load_model():
    global model, tokenizer, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_8bit = USE_8BIT and device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CHECKPOINT,
        trust_remote_code=True,
        use_fast=False,
    )

    model = AutoModel.from_pretrained(
        MODEL_CHECKPOINT,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if use_8bit else None,
        quantization_config=build_quantization_config() if use_8bit else None,
    ).eval()

    if not use_8bit:
        model = model.to(device)


@app.post("/geotikzbridge")
async def geotikzbridge(
    image: UploadFile = File(...),
    prompt: str = Form(""),
):
    if model is None or tokenizer is None or device is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        pixel_values = image_to_pixel_values(pil_image)
        pixel_values = pixel_values.to(dtype=dtype, device=device)

        generation_config = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "do_sample": False,
        }

        question = prompt.strip()

        if not question:
            question = "Generate the TikZ code for this geometry image."

        if "<image>" not in question:
            question = "<image>\n" + question

        if DEBUG:
            print(question, flush=True)

        with torch.no_grad():
            raw_output = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
            )

        tikz_code = extract_tikz_code(raw_output)

        return JSONResponse({
            "model": "geotikzbridge",
            "model_checkpoint": MODEL_CHECKPOINT,
            "use_8bit": USE_8BIT and torch.cuda.is_available(),
            "tikz": tikz_code,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "geotikzbridge",
        "model_checkpoint": MODEL_CHECKPOINT,
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "use_8bit": USE_8BIT and torch.cuda.is_available(),
    }