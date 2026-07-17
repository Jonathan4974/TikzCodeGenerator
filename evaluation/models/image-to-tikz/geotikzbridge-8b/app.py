from io import BytesIO
import os
import re

import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "NONE").strip('"')
USE_8BIT = os.getenv("USE_8BIT", "false").lower() == "true"
DEBUG_DEFAULT = os.getenv("DEBUG", "false").lower() == "true"

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "0.95"))

DEFAULT_PROMPT = "Generate the TikZ code for this geometry image."
DESCRIPTION_PREFIX = (
    "Additionally here is a description of the image with some creation hints:"
)

app = FastAPI(title="GeoTikzBridge API")

model = None
tokenizer = None
device = None

def image_to_pixel_values(image: Image.Image) -> torch.Tensor:
    image_size = (
        model.config.force_image_size
        or model.config.vision_config.image_size
    )

    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    return transform(image).unsqueeze(0)


def quantization_config() -> BitsAndBytesConfig:
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


def build_prompt(prompt: str, description: str, use_description: bool) -> str:
    final_prompt = prompt.strip() or DEFAULT_PROMPT

    if use_description and description.strip():
        final_prompt += f"\n\n{DESCRIPTION_PREFIX}\n{description.strip()}"

    if "<image>" not in final_prompt:
        final_prompt = f"<image>\n{final_prompt}"

    return final_prompt


def extract_tikz_code(text: str) -> str:
    text = text.strip()

    match = re.search(
        r"```(?:tikz|tex|latex)\s*\n?(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ) or re.search(r"```\s*\n?(.*?)```", text, flags=re.DOTALL)

    return match.group(1).strip() if match else text


@app.on_event("startup")
def load_model() -> None:
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
        quantization_config=quantization_config() if use_8bit else None,
    ).eval()

    if not use_8bit:
        model = model.to(device)


@app.post("/geotikzbridge")
async def geotikzbridge(
    image: UploadFile = File(...),
    llm_description: UploadFile = File(...),
    prompt: str = Form(""),
    use_llm_description: bool = Form(True),
    debug: bool = Form(DEBUG_DEFAULT),
):
    if model is None or tokenizer is None or device is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        pil_image = Image.open(BytesIO(await image.read())).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid image: {error}") from error

    try:
        description = (await llm_description.read()).decode("utf-8")
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="llm_description must be UTF-8 text",
        ) from error

    final_prompt = build_prompt(prompt, description, use_llm_description)

    if debug:
        print(
            f"\n===== FINAL PROMPT =====\n{final_prompt}\n========================\n",
            flush=True,
        )

    try:
        pixel_values = image_to_pixel_values(pil_image)

        vision_parameter = next(model.vision_model.parameters())
        pixel_values = pixel_values.to(
            device=vision_parameter.device,
            dtype=vision_parameter.dtype,
        )

        if debug:
            actual_tokens = int(
                (pixel_values.shape[-1] // model.patch_size) ** 2
                * model.downsample_ratio**2
            )

            print(
                "\n===== IMAGE DEBUG =====\n"
                f"pixel_values: {tuple(pixel_values.shape)}\n"
                f"model image size: {model.config.force_image_size}\n"
                f"model image tokens: {model.num_image_token}\n"
                f"actual image tokens: {actual_tokens}\n"
                "=======================\n",
                flush=True,
            )

        generation_config = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "do_sample": False,
        }

        with torch.no_grad():
            output = model.chat(
                tokenizer,
                pixel_values,
                final_prompt,
                generation_config,
            )

        return {
            "model": "geotikzbridge",
            "model_checkpoint": MODEL_CHECKPOINT,
            "use_8bit": USE_8BIT and torch.cuda.is_available(),
            "tikz": extract_tikz_code(str(output)),
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "geotikzbridge",
        "model_checkpoint": MODEL_CHECKPOINT,
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
        "use_8bit": USE_8BIT and torch.cuda.is_available(),
    }