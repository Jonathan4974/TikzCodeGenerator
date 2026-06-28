from pathlib import Path
import base64
import os

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b-it-qat")
IMAGE_BASE_PATH = Path("/usr/prakt/s0030/projects/data/benchmark_data/images")

app = FastAPI()


class RequestBody(BaseModel):
    prompt: str
    image: str
    model: str | None = None


def image_to_base64(path: str) -> str:
    path = Path(path)
    filename = path.name
    image_path = IMAGE_BASE_PATH / filename

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


@app.post("/generate", response_class=PlainTextResponse)
def generate(body: RequestBody):
    try:
        model = body.model or DEFAULT_OLLAMA_MODEL
        image_b64 = image_to_base64(body.image)

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": body.prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 8192,
            },
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=900)
        response.raise_for_status()

        data = response.json()
        output = data.get("message", {}).get("content", "")

        if not isinstance(output, str):
            output = str(output)

        output = output.strip()

        if not output:
            raise HTTPException(status_code=502, detail="Ollama returned empty output")

        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))