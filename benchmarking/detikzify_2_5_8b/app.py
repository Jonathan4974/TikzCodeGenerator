from operator import itemgetter
from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from detikzify.model import load
from detikzify.infer import DetikzifyPipeline


MODEL_PATH = "/models/detikzify_2_5_8b"

app = FastAPI(title="DeTikZify API")

pipeline: Optional[DetikzifyPipeline] = None


@app.on_event("startup")
def load_model():
    global pipeline

    pipeline = DetikzifyPipeline(*load(
        model_name_or_path=MODEL_PATH,
        device_map="auto",
        torch_dtype="bfloat16",
    ))


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
        figs = set()

        for score, fig in pipeline.simulate(image=pil_image, timeout=timeout):
            figs.add((score, fig))

        if not figs:
            raise HTTPException(status_code=500, detail="No TikZ figure generated")

        best_score, best_fig = sorted(figs, key=itemgetter(0))[-1]

        # best_fig als .tex speichern/lesen
        output_path = "/tmp/fig.tex"
        best_fig.save(output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            tikz_code = f.read()

        return JSONResponse({
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
        "model_loaded": pipeline is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }