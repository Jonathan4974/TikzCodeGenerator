from unsloth import FastModel
from peft import PeftModel


BASE_MODEL = "/root/projects/models/gemma-4-31B-it"
SFT_ADAPTER = "/root/projects/models/checkpoints-normal/checkpoint-1100"
OUTPUT_PATH = "/root/projects/models/gemma4-31B-it-tikz-q4_k_m"

model, processor = FastModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=9216,
    load_in_4bit=True,
)

model = PeftModel.from_pretrained(
    model,
    SFT_ADAPTER,
)

model.save_pretrained_gguf(
    OUTPUT_PATH,
    processor,
    quantization_method="q4_k_m",
)