import os
import gc
import asyncio
import logging
from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from unsloth import FastVisionModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemma4-tikz-api")


MODEL_PATH = os.getenv("MODEL_PATH", "/models/gemma4-rl/gemma4_grpo_lora")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))

app = FastAPI(title="Gemma4 TikZ API")

model = None
tokenizer = None
inference_lock = asyncio.Lock()


INSTRUCTION = """Take this image and write the LaTeX/TikZ code for it.

VLM description:
{vlm_description}

Return only complete compilable LaTeX code.
Do not explain anything.
Do not use Markdown.
Stop immediately after \\end{{document}}.
"""


def clean_code(text: str) -> str:
    text = str(text).strip()

    text = text.replace("```latex", "")
    text = text.replace("```tex", "")
    text = text.replace("```", "")
    text = text.strip()

    start = r"\documentclass"
    if start in text:
        text = text[text.index(start):]

    end = r"\end{document}"
    if end in text:
        text = text[: text.index(end) + len(end)]

    return text.strip()


@app.on_event("startup")
def load_model():
    global model, tokenizer

    logger.info("Loading model from %s", MODEL_PATH)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_PATH,
        load_in_4bit=True,
        fast_inference=True,
    )

    FastVisionModel.for_inference(model)

    logger.info("Model loaded")


@app.post("/tikz")
async def generate_tikz(
    image: UploadFile = File(...),
    vlm_description: str = Form(""),
    temperature: float = Form(0.2),
    top_p: float = Form(0.9),
):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    prompt = INSTRUCTION.format(vlm_description=vlm_description)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        async with inference_lock:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(
                pil_image,
                text,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    use_cache=True,
                )

            output_text = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=False,
            )

            tikz_code = clean_code(output_text)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return JSONResponse({
            "model_path": MODEL_PATH,
            "tikz": tikz_code,
        })

    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))