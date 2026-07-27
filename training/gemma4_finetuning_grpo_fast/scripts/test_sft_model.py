import torch
from pathlib import Path

from peft import PeftModel
from PIL import Image
from unsloth import FastModel


BASE_MODEL = "/root/projects/models/gemma-4-31B-it"
SFT_ADAPTER = "/root/projects/models/checkpoints-normal/checkpoint-1100"

IMAGE_PATH = "/root/projects/00000004.png"
OUTPUT_PATH = "/root/projects/generated.tex"

PROMPT = """As a LaTeX graphics expert, translate the image into TikZ code suitable for academic publications.
Focus on recreating geometric precision, typographic elements, and color schemes.
The code must be compilable, efficient, and maintain the original image's visual fidelity for professional document integration.
Return only the full LaTeX document.
Do not use markdown fences.
Do not add explanations."""

full_prompt = (
    PROMPT
)

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

image = Image.open(IMAGE_PATH).convert("RGB")

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
                "text": full_prompt,
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
).to(model.device)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=8192,
        do_sample=False,
        use_cache=True,
    )

generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

latex_code = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0].strip()

Path(OUTPUT_PATH).write_text(
    latex_code,
    encoding="utf-8",
)

print(latex_code)