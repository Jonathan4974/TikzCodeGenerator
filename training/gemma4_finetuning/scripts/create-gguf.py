import os
import torch
from peft import PeftModel
from unsloth import FastVisionModel

BASE_PATH = "/models/huggingface/gemma4-base-16bit"  # <- hier dein echtes 16-bit Base-Modell
LORA_PATH = "/models/gemma4-rl/gemma4_grpo_lora"

MERGED_PATH = "/models/gemma4-rl/gemma4_merged_16bit"
GGUF_PATH = "/models/gemma4-rl/gemma4_gguf"

os.makedirs(MERGED_PATH, exist_ok=True)
os.makedirs(GGUF_PATH, exist_ok=True)

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=BASE_PATH,
    max_seq_length=4096,
    load_in_4bit=False,
    dtype=torch.bfloat16,
    fast_inference=False,
)

model = PeftModel.from_pretrained(
    model,
    LORA_PATH,
)

model = model.merge_and_unload()

model.save_pretrained(
    MERGED_PATH,
    safe_serialization=True,
)

tokenizer.save_pretrained(MERGED_PATH)

print("Merged model saved:", MERGED_PATH)

# Danach mit Unsloth oder llama.cpp aus MERGED_PATH nach GGUF konvertieren.