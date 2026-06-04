from operator import itemgetter
from io import BytesIO
from typing import Optional
import os
import logging

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from transformers import BitsAndBytesConfig
from detikzify.model import load
from detikzify.infer import DetikzifyPipeline


logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("detikzify-api")


MODEL_PATH = os.getenv("MODEL_PATH", "/models/detikzify_2_5_8b")
QUANTIZATION = os.getenv("QUANTIZATION", "8bit").lower()

app = FastAPI(title="DeTikZify API")

pipeline: Optional[DetikzifyPipeline] = None


def build_quantization_config():
    if QUANTIZATION in ("none", "false", "0", "no"):
        return None

    if QUANTIZATION == "8bit":
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

    raise ValueError(
        f"Invalid QUANTIZATION={QUANTIZATION!r}. Use '8bit' or 'none'."
    )


@app.on_event("startup")
def load_model():
    global pipeline

    try:
        quantization_config = build_quantization_config()

        load_kwargs = {
            "model_name_or_path": MODEL_PATH,
            "device_map": "auto",
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        model, processor = load(**load_kwargs)
        pipeline = DetikzifyPipeline(model, processor)

    except Exception:
        logger.exception("Failed to load model")
        raise


@app.post("/detikzify")
async def detikzify(
    image: UploadFile = File(...),
    timeout: int = Form(60),
):
    if pipeline is None:
        logger.error("Model is not loaded yet")
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.exception("Invalid image")
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        figs = []

        for score, fig in pipeline.simulate(image=pil_image, timeout=timeout):
            figs.append((score, fig))

        if not figs:
            logger.error("No TikZ figure generated")
            raise HTTPException(status_code=500, detail="No TikZ figure generated")

        best_score, best_fig = max(figs, key=itemgetter(0))

        output_path = "/tmp/fig.tex"
        best_fig.save(output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            tikz_code = f.read()

        return JSONResponse({
            "model_path": MODEL_PATH,
            "quantization": QUANTIZATION,
            "score": float(best_score),
            "tikz": tikz_code,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("DeTikZify request failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "quantization": QUANTIZATION,
        "model_loaded": pipeline is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }