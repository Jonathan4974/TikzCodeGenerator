import base64
import os
import re

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/chat",
)

app = FastAPI()


LATEX_BLOCK_PATTERN = re.compile(
    r"```(?:latex|tex)[ \t]*\r?\n?(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)

GENERIC_BLOCK_PATTERN = re.compile(
    r"```[a-zA-Z0-9_-]*[ \t]*\r?\n?(.*?)```",
    flags=re.DOTALL,
)


def clean_tex(code: str) -> str:
    
    if not code:
        return ""

    code = code.strip().lstrip("\ufeff")

    # Bevorzugt explizite ```latex- oder ```tex-Blöcke.
    match = LATEX_BLOCK_PATTERN.search(code)

    # Fallback für generische Markdown-Codeblöcke.
    if not match:
        match = GENERIC_BLOCK_PATTERN.search(code)

    if match:
        code = match.group(1).strip()

    # Vollständiges LaTeX-Dokument extrahieren.
    document_start = code.find(r"\documentclass")
    document_end = code.rfind(r"\end{document}")

    if document_start >= 0 and document_end >= 0:
        document_end += len(r"\end{document}")
        return code[document_start:document_end].strip()

    # Fallback für reine TikZ-Fragmente.
    tikz_start = code.find(r"\begin{tikzpicture}")
    tikz_end = code.rfind(r"\end{tikzpicture}")

    if tikz_start >= 0 and tikz_end >= 0:
        tikz_end += len(r"\end{tikzpicture}")
        return code[tikz_start:tikz_end].strip()

    # Übrig gebliebene Markdown-Fences entfernen.
    code = re.sub(
        r"^\s*```[a-zA-Z0-9_-]*\s*",
        "",
        code,
        flags=re.IGNORECASE,
    )
    code = re.sub(r"\s*```\s*$", "", code)

    return code.strip()


@app.post("/generate", response_class=PlainTextResponse)
def generate(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    llm_description: UploadFile = File(...),
    use_llm_description: bool = Form(True),
    debug: bool = Form(False),
    model: str = Form(...),
):
    try:
        image_b64 = base64.b64encode(
            image.file.read()
        ).decode("utf-8")

        description = (
            llm_description.file.read()
            .decode("utf-8")
            .strip()
        )

        final_prompt = prompt.strip()

        if use_llm_description and description:
            final_prompt += (
                "\n\nAdditionally, here is a description of the image "
                "with some creation hints:\n"
                f"{description}"
            )

        if debug:
            print(
                "\n===== FINAL PROMPT =====\n"
                f"{final_prompt}\n"
                "========================\n",
                flush=True,
            )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": final_prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 8192,
            },
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=900,
        )
        response.raise_for_status()

        raw_output = (
            response.json()
            .get("message", {})
            .get("content", "")
        )

        if debug:
            print(
                "\n===== raw_output =====\n"
                f"{raw_output}\n"
                "========================\n",
                flush=True,
            )

        output = clean_tex(raw_output)

        if not output:
            raise HTTPException(
                status_code=502,
                detail="Ollama returned no usable LaTeX output",
            )

        if debug:
            print(
                "\n===== CLEANED LATEX =====\n"
                f"{output}\n"
                "=========================\n",
                flush=True,
            )

        return output

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error