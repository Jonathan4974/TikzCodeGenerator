from __future__ import annotations

import argparse
from pathlib import Path

from unsloth import FastVisionModel

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "grpo" / "lora"))
    parser.add_argument("--output", default=str(ROOT / "models" / "gguf"))
    parser.add_argument("--quantization", default="q4_k_m")
    args = parser.parse_args()

    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model,
        max_seq_length=8192,
        load_in_4bit=True,
    )
    model.save_pretrained_gguf(
        args.output,
        processor,
        quantization_method=args.quantization,
    )


if __name__ == "__main__":
    main()
