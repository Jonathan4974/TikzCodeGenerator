import os, sys
from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
import json
import asyncio

import config

from prompt import COMMAND_TIKZ

from tikzcodegenerator.demo.demo_wo_descrip_gen.backend.ollama_client_backup import (
    generate_tikz,
    OllamaTimeoutError,
    OllamaConnectionError,
    OllamaModelError,
    OllamaResponseError,
    OllamaOutputError,
)

from tex_compiler import (
    tex_cleaning,
    compile_tikz,
    clean_code,
)

from utils import(
    render_instruction,
    normalize_canvas,
)


config.apply_environment()

BASE_DIR = Path(__file__).resolve().parent.parent 
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="TikZ Generator Demo")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")

@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/shutdown")
async def shutdown():
    # force the demo to end
    os._exit(0)

@app.post("/generate")
async def generate(image: UploadFile = File(...), description: str = Form(""), num_samples: int = Form(1)):

    # save uploaded image
    image_path = config.UPLOAD_DIR / image.filename
    with image_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)

    normalize_canvas(image_path)

    # build prompt
    prompt_text = render_instruction(COMMAND_TIKZ, description)

    # call ollama
    async def event_stream():
        for i in range(num_samples):
            # dynamically set temperature（0.1 ~ 0.9）
            if num_samples > 1:
                temp = 0.1 + (i / (num_samples - 1)) * 0.8
            else:
                temp = 0.5
            options = {"temperature": temp, "seed": 42 + i}
            
            try:
                # asynchronously call the generation function
                raw_tex = await asyncio.to_thread(
                    generate_tikz,
                    sketch=image_path,
                    prompt_text=prompt_text,
                    options=options
                )
                tex = clean_code(raw_tex)
                
                png_path = config.OUTPUT_DIR / f"result_{i}.png"
                compile_result = compile_tikz(tex, png_path)   

                result_data = {
                    "index": i,
                    "tikz": tex,
                    "success": compile_result["success"],
                    "png": f"/outputs/result_{i}.png" if compile_result["success"] and not compile_result.get("overall_blank") else None,
                    "error": None if compile_result["success"] else compile_result.get("message"),
                    "attempts": compile_result.get("attempts"),
                }
            except Exception as e:
                result_data = {
                    "index": i,
                    "tikz": None,
                    "success": False,
                    "png": None,
                    "error": str(e),
                    "attempts": None,
                }
            yield f"data: {json.dumps(result_data)}\n\n"
            await asyncio.sleep(0.05)

        yield "event: done\ndata: \n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")