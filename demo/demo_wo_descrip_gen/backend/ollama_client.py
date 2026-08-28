"""
Ollama client for the TikZ demo.

Responsibilities
----------------
1. Encode uploaded image
2. Build Ollama request
3. Send request to local Ollama server
4. Return generated LaTeX code
"""

import base64
from pathlib import Path

import requests

import config



def encode_image(image_path: str | Path) -> str:
    """
    Encode an image to base64.

    Parameters:
        image_path : str | Path

    Returns:
        base64 encoded image : str
    """

    image_path = Path(image_path)

    with image_path.open("rb") as f:
        return base64.b64encode(
            f.read()
        ).decode("utf-8")


def build_payload(prompt_text: str, image_path: str | Path, options: dict):
    """
    Build request payload for Ollama /api/chat.
    """

    image_base64 = encode_image(image_path)

    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
                "images": [image_base64]
            }
        ],
        "stream": False,
        "think": False,
        "options": options,
    }

    return payload


# ============================================================
# Custom Exceptions
# ============================================================

class OllamaError(Exception):
    """Base exception for Ollama client."""
    pass


class OllamaTimeoutError(OllamaError):
    """Model inference timeout."""
    pass


class OllamaConnectionError(OllamaError):
    """Cannot connect to Ollama server."""
    pass


class OllamaModelError(OllamaError):
    """Requested model is not available."""
    pass


class OllamaResponseError(OllamaError):
    """Ollama returned an invalid response."""
    pass


class OllamaOutputError(OllamaError):
    """Model returned no usable output."""
    pass


# ============================================================
# Main API
# ============================================================

def generate_tikz(sketch: str | Path, prompt_text: str, options: dict = {}):
    """
    Generate TikZ code with Ollama.

    Parameters:
        sketch: Uploaded image.
        prompt_text: Prompt including optional figure description.
        options: ollama model options (e.g. temperature, top_p, seed, num_predict, etc.)

    Returns:
        Raw response from the model : str
    """

    payload = build_payload(prompt_text,sketch,options)

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        response.raise_for_status()

    except requests.exceptions.Timeout:
        raise OllamaTimeoutError("Model replying timeout.")

    except requests.exceptions.ConnectionError as e:
        raise OllamaConnectionError(f"Cannot connect to Ollama.\n{e}")

    except requests.exceptions.HTTPError as e:
        response_text = ""
        if e.response is not None:
            response_text = e.response.text.lower()

        if "model" in response_text and "not found" in response_text:
            raise OllamaModelError(response_text)

        raise OllamaResponseError(f"HTTP Error: {e}")

    except requests.exceptions.RequestException as e:
        raise OllamaResponseError(str(e))

    try:
        data = response.json()
    except ValueError:
        raise OllamaResponseError("Ollama did not return valid JSON.")

    if "message" not in data:
        raise OllamaResponseError("Missing 'message' field.")

    if "content" not in data["message"]:
        raise OllamaOutputError("Model returned no content.")

    return data["message"]["content"]