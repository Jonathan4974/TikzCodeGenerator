import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import csv
import heapq
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from unsloth import FastVisionModel


@dataclass
class LossSelectionConfig:
    model_name: str = "/models/gemma4-sft/gemma4_sft_lora"
    dataset_path: str = "/data"
    manifest: str = "manifest_train.csv"
    output: str = "/data/hardest_manifest.csv"

    top_k: int = 100
    candidate_limit: int | None = 1000
    seed: int = 3407
    max_seq_length: int = 8192

    image_column: str = "image_path"
    code_column: str = "code_path"
    vlm_description_column: str = "vlm_description_path"


def clean_code(text: str) -> str:
    text = str(text).strip()

    start = r"\documentclass"
    if start in text:
        text = text[text.index(start):]

    end = r"\end{document}"
    if end in text:
        text = text[: text.index(end) + len(end)]

    return text.strip()


def build_prompt(vlm_description: str) -> str:
    vlm_description = str(vlm_description).strip()

    if vlm_description:
        vlm_description_block = f"VLM description:\n{vlm_description}"
    else:
        vlm_description_block = ""

    return f"""Take this image and write the LaTeX/TikZ code for it.

{vlm_description_block}

Return only complete compilable LaTeX code.
Do not explain anything.
Do not use Markdown.
Stop immediately after \\end{{document}}.
"""


def resolve(root: Path, p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else root / p


@torch.inference_mode()
def sample_loss(model, tokenizer, image, prompt: str, answer: str, device: str) -> tuple[float, int]:
    prompt_text = (
        "<bos><|turn>user\n"
        "<|image|>"
        f"{prompt}"
        "<turn|>\n"
        "<|turn>model\n"
    )

    full_text = prompt_text + answer.strip() + "<turn|>"

    prompt_inputs = tokenizer(
        image,
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    prompt_len = prompt_inputs["input_ids"].shape[-1]
    del prompt_inputs

    full_inputs = tokenizer(
        image,
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100

    outputs = model(
        **full_inputs,
        labels=labels,
        use_cache=False,
    )

    loss = outputs.loss.item()
    n_tokens = (labels[:, 1:] != -100).sum().item()

    del full_inputs, labels, outputs

    return loss, n_tokens


def main():
    cfg = LossSelectionConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(cfg.dataset_path)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
    )

    FastVisionModel.for_inference(model)
    model.eval()
    model.config.use_cache = False

    manifest_path = root / cfg.manifest
    output_path = Path(cfg.output)

    heap = []
    counter = 0

    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if cfg.candidate_limit is not None and len(rows) > cfg.candidate_limit:
        random.seed(cfg.seed)
        rows = random.sample(rows, cfg.candidate_limit)

    progress = tqdm(rows, total=len(rows), desc="Scoring samples")

    for i, row in enumerate(progress):
        image_path = resolve(root, row[cfg.image_column])
        code_path = resolve(root, row[cfg.code_column])
        desc_path = resolve(root, row[cfg.vlm_description_column])

        image = Image.open(image_path).convert("RGB")
        desc = desc_path.read_text(encoding="utf-8").strip()
        answer = clean_code(code_path.read_text(encoding="utf-8"))

        prompt = build_prompt(desc)

        try:
            loss, n_tokens = sample_loss(
                model=model,
                tokenizer=tokenizer,
                image=image,
                prompt=prompt,
                answer=answer,
                device=device,
            )

        except RuntimeError as e:
            progress.write(f"[skip] row={i} error={e}")

            if "out of memory" in str(e).lower() and device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            continue

        except Exception as e:
            progress.write(f"[skip] row={i} error={e}")
            continue

        row_out = dict(row)
        row_out["sft_loss"] = loss
        row_out["sft_answer_tokens"] = n_tokens

        item = (loss, counter, row_out)
        counter += 1

        if len(heap) < cfg.top_k:
            heapq.heappush(heap, item)
        else:
            heapq.heappushpop(heap, item)

        progress.set_postfix(
            loss=f"{loss:.4f}",
            tokens=n_tokens,
            kept=len(heap),
        )

        del image

    hardest = sorted(heap, key=lambda x: x[0], reverse=True)

    out_fields = list(fieldnames) + ["sft_loss", "sft_answer_tokens"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()

        for _, _, row in hardest:
            writer.writerow(row)

    print(f"Saved hardest manifest: {output_path}")
    print(f"Rows written: {len(hardest)}")


if __name__ == "__main__":
    main()