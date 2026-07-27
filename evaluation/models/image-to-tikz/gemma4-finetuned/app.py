import re
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from peft import PeftModel
from unsloth import FastModel
from io import BytesIO

BASE_MODEL = "/home/jonas/models/gemma-4-31B-it"
SFT_ADAPTER = "/home/jonas/models/sft-finetuned-checkpoint-1100"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, processor = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=9216,
        load_in_4bit=True,
    )

    model = PeftModel.from_pretrained(
        model,
        SFT_ADAPTER,
        is_trainable=False,
    )

    FastModel.for_inference(model)

    app.state.model = model
    app.state.processor = processor

    yield


app = FastAPI(lifespan=lifespan)


def clean_tex(code: str) -> str:
    code = code.strip().lstrip("\ufeff")

    match = re.search(
        r"```[a-zA-Z0-9_-]*[ \t]*\r?\n?(.*?)```",
        code,
        flags=re.DOTALL,
    )
    if match:
        code = match.group(1).strip()

    start = code.find(r"\documentclass")
    end = code.rfind(r"\end{document}")
    if start >= 0 and end >= start:
        return code[start:end + len(r"\end{document}")].strip()

    start = code.find(r"\begin{tikzpicture}")
    end = code.rfind(r"\end{tikzpicture}")
    if start >= 0 and end >= start:
        return code[start:end + len(r"\end{tikzpicture}")].strip()

    return code.strip()

def split_thinking(text: str) -> tuple[str, str]:
    thought_marker = "<|channel>thought"
    final_marker = "<|channel>final"

    if final_marker in text:
        thinking, answer = text.split(final_marker, maxsplit=1)
        thinking = thinking.replace(
            thought_marker,
            "",
        ).strip()
        return thinking, answer.strip()

    return "", text.strip()


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    llm_description: UploadFile = File(...),
    use_llm_description: bool = Form(True),
    think: bool = Form(False),
    num_predict: int = Form(8192, ge=1),
    thinking_token_multiplier: int = Form(2, ge=1),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "Image is empty")

    try:
        description = (
            await llm_description.read()
        ).decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise HTTPException(
            400,
            "LLM description must be UTF-8",
        ) from error

    final_prompt = prompt.strip()
    if use_llm_description and description:
        final_prompt += (
            "\n\nAdditionally, here is a description of the image "
            "with some creation hints:\n"
            f"{description}"
        )

    try:
        pil_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")
    except Exception as error:
        raise HTTPException(400, "Invalid image") from error

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                },
                {
                    "type": "text",
                    "text": final_prompt,
                },
            ],
        }
    ]

    model = app.state.model
    processor = app.state.processor

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=think,
    ).to(model.device)

    max_new_tokens = (
        num_predict * thinking_token_multiplier
        if think
        else num_predict
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    generated_ids = output_ids[
        :,
        inputs["input_ids"].shape[1]:,
    ]

    raw_output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
    )[0].strip()

    thinking_output, answer_output = split_thinking(raw_output)

    output = clean_tex(answer_output)
    if not output:
        raise HTTPException(
            502,
            {
                "message": "Model returned no usable LaTeX output",
                "raw_output": raw_output,
            },
        )

    return {
        "output": output,
        "thinking_length": len(thinking_output),
    }