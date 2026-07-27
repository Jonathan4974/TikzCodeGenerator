import asyncio
import re
from contextlib import asynccontextmanager
from io import BytesIO

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from unsloth import FastModel


MODEL_NAME = "/home/jonas/models/gemma-4-31B-it"
MAX_SEQ_LENGTH = 9216
MODEL_CONCURRENCY = 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, processor = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    FastModel.for_inference(model)

    app.state.model = model
    app.state.processor = processor
    app.state.semaphore = asyncio.Semaphore(MODEL_CONCURRENCY)

    yield

    del app.state.model
    del app.state.processor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


def clean_tex(code: str) -> str:
    if not code:
        return ""

    code = code.strip().lstrip("\ufeff")

    match = re.search(
        r"```[a-zA-Z0-9_-]*[ \t]*\r?\n?(.*?)```",
        code,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        code = match.group(1).strip()

    start = code.find(r"\documentclass")
    end = code.rfind(r"\end{document}")
    if start >= 0 and end >= start:
        return code[start : end + len(r"\end{document}")].strip()

    start = code.find(r"\begin{tikzpicture}")
    end = code.rfind(r"\end{tikzpicture}")
    if start >= 0 and end >= start:
        return code[start : end + len(r"\end{tikzpicture}")].strip()

    return code.strip()


def split_thinking(text: str) -> tuple[str, str]:
    final_marker = "<|channel>final"
    thought_marker = "<|channel>thought"

    if final_marker not in text:
        return "", text.strip()

    thinking, answer = text.split(final_marker, maxsplit=1)
    thinking = thinking.replace(thought_marker, "").strip()

    return thinking, answer.strip()


def run_inference(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    think: bool,
    max_new_tokens: int,
) -> tuple[str, str]:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=think,
    ).to(model.device)

    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    generated_ids = output_ids[:, prompt_length:]

    raw_output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
    )[0].strip()

    return split_thinking(raw_output)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


@app.post("/generate")
async def generate(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile = File(...),
    llm_description: UploadFile | None = File(None),
    use_llm_description: bool = Form(True),
    think: bool = Form(False),
    num_predict: int = Form(8192, ge=1),
    thinking_token_multiplier: int = Form(2, ge=1),
    debug: bool = Form(False),
) -> dict:
    image_data = await image.read()

    if not image_data:
        raise HTTPException(
            status_code=400,
            detail="Image is empty",
        )

    try:
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid image",
        ) from error

    description = ""

    if llm_description is not None:
        try:
            description = (
                await llm_description.read()
            ).decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise HTTPException(
                status_code=400,
                detail="LLM description must be UTF-8",
            ) from error

    final_prompt = prompt.strip()

    if use_llm_description and description:
        final_prompt += (
            "\n\nAdditionally, here is a description of the image "
            "with some creation hints:\n"
            f"{description}"
        )

    max_new_tokens = num_predict

    if think:
        max_new_tokens *= thinking_token_multiplier

    if debug:
        print(
            "\n===== REQUEST =====\n"
            f"model: {MODEL_NAME}\n"
            f"think: {think}\n"
            f"max_new_tokens: {max_new_tokens}\n"
            f"\n{final_prompt}\n"
            "===================\n",
            flush=True,
        )

    try:
        async with request.app.state.semaphore:
            thinking_output, answer_output = await asyncio.to_thread(
                run_inference,
                request.app.state.model,
                request.app.state.processor,
                pil_image,
                final_prompt,
                think,
                max_new_tokens,
            )
    except torch.cuda.OutOfMemoryError as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=503,
            detail="CUDA out of memory",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {error}",
        ) from error

    output = clean_tex(answer_output)

    if debug:
        print(
            "\n===== MODEL RESPONSE =====\n"
            f"thinking_length: {len(thinking_output)}\n"
            f"content_length: {len(answer_output)}\n"
            "\n===== RAW ANSWER =====\n"
            f"{answer_output}\n"
            "==========================\n",
            flush=True,
        )

    if not output:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Model returned no usable LaTeX output",
                "thinking_length": len(thinking_output),
            },
        )

    return {
        "output": output,
        "thinking_length": len(thinking_output),
    }