from io import BytesIO
from typing import Optional
import os

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "SJY-1995/GeoTikzBridge-Base-8B")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "0.95"))

app = FastAPI(title="GeoTikzBridge API")

model: Optional[AutoModelForCausalLM] = None
processor: Optional[AutoProcessor] = None
device: Optional[torch.device] = None


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


@app.on_event("startup")
def load_model():
    global model, processor, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_8bit = device.type == "cuda"
    device_map = "auto" if use_8bit else None
    quantization_config = build_quantization_config() if use_8bit else None

    processor = AutoProcessor.from_pretrained(
        MODEL_CHECKPOINT,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CHECKPOINT,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if use_8bit else torch.float32,
        device_map=device_map,
        load_in_8bit=use_8bit,
        quantization_config=quantization_config,
    ).eval()


@app.post("/geotikzbridge")
async def geotikzbridge(
    image: UploadFile = File(...),
):
    if model is None or processor is None or device is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        inputs = processor(
            text="",
            images=pil_image,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=False,
            )

        tikz_code = processor.decode(output[0], skip_special_tokens=True)

        return JSONResponse({
            "model": "geotikzbridge",
            "model_checkpoint": MODEL_CHECKPOINT,
            "tikz": tikz_code,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "geotikzbridge",
        "model_checkpoint": MODEL_CHECKPOINT,
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
