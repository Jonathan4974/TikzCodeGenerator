from pathlib import Path
import base64
import os

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/chat")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b-it-qat")

app = FastAPI()


class RequestBody(BaseModel):
    prompt: str
    image: str
    model: str | None = None


def image_to_base64(path: str) -> str:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    return base64.b64encode(path.read_bytes()).decode("utf-8")


@app.post("/generate")
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

        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        response.raise_for_status()

        data = response.json()
        return {
            "output": data["message"]["content"],
            "model": model,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
