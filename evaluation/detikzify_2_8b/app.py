from operator import itemgetter
from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from transformers import BitsAndBytesConfig
from detikzify.model import load
from detikzify.infer import DetikzifyPipeline


MODEL_PATH = "/models/detikzify_2_8b"

app = FastAPI(title="DeTikZify v2-8B API")

pipeline: Optional[DetikzifyPipeline] = None


@app.on_event("startup")
def load_model():
    global pipeline

    bnb_config = BitsAndBytesConfig(
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

    model, processor = load(
        model_name_or_path=MODEL_PATH,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    pipeline = DetikzifyPipeline(model, processor)


@app.post("/detikzify")
async def detikzify(
    image: UploadFile = File(...),
    timeout: int = Form(60),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        figs = []

        for score, candidate_fig in pipeline.simulate(
            image=pil_image,
            timeout=timeout,
        ):
            figs.append((score, candidate_fig))

        if not figs:
            raise HTTPException(
                status_code=500,
                detail="pipeline.simulate() hat keine Kandidaten erzeugt.",
            )

        best_score, best_fig = max(figs, key=itemgetter(0))

        output_path = "/tmp/fig.tex"
        best_fig.save(output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            tikz_code = f.read()

        return JSONResponse({
            "model": "detikzify-v2-8b",
            "model_path": MODEL_PATH,
            "score": float(best_score),
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
        "model": "detikzify-v2-8b",
        "model_path": MODEL_PATH,
        "model_loaded": pipeline is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }