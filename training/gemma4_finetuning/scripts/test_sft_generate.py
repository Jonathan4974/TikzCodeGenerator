import argparse
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from unsloth import FastVisionModel
from transformers import StoppingCriteria, StoppingCriteriaList


def clean_code(text: str) -> str:
    text = str(text)

    text = text.replace("```latex", "").replace("```tex", "").replace("```", "")

    start = r"\documentclass"
    if start in text:
        text = text[text.index(start):]

    end = r"\end{document}"
    if end in text:
        text = text[: text.index(end) + len(end)]

    return text.strip()


class StopAtEndDocument(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[0][self.prompt_len:]
        text = self.tokenizer.decode(gen, skip_special_tokens=False)
        return r"\end{document}" in text


def build_prompt(desc: str) -> str:
    return f"""Take this image and generate a complete LaTeX TikZ document.

Visual description:
{desc}

Return only LaTeX.
Start with \\documentclass.
Stop after \\end{{document}}.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="/models/gemma4-sft/gemma4_sft_lora")
    parser.add_argument("--dataset_path", default="/data")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--vlm_description", default="")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=5000)
    parser.add_argument("--output_file", default="/app/generated_sft_test.tex")
    args = parser.parse_args()

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
    )
    FastVisionModel.for_inference(model)

    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
        desc = args.vlm_description
        reference = ""
    else:
        df = pd.read_csv(Path(args.dataset_path) / "manifest.csv")
        row = df.iloc[args.index]

        image = Image.open(Path(args.dataset_path) / row["image_path"]).convert("RGB")
        desc = Path(args.dataset_path, row["vlm_description_path"]).read_text()
        reference = clean_code(Path(args.dataset_path, row["code_path"]).read_text())

    prompt = build_prompt(desc)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ).rstrip() + "\n<|turn>model\n"

    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    prompt_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([
                StopAtEndDocument(tokenizer, prompt_len)
            ]),
        )

    # Nur neue Tokens decodieren, nicht Prompt + Antwort
    new_tokens = outputs[0][prompt_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
    code = clean_code(raw)

    Path(args.output_file).write_text(code, encoding="utf-8")

    print("\nRAW GENERATED:\n")
    print(raw)

    print("\nCLEAN CODE:\n")
    print(code)

    print("\nREFERENCE START:\n")
    print(reference[:1000] if reference else "No reference")

    print("\nSUMMARY:")
    print("saved:", args.output_file)
    print("contains documentclass:", "\\documentclass" in code)
    print("contains end document:", "\\end{document}" in code)


if __name__ == "__main__":
    main()