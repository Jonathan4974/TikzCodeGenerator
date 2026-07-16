"""Central configuration for the TikZ dataset pipeline."""

from pathlib import Path
import os


# Dictionary order is split order; list order is mode order.
# Every mode is an independent run and writes its own Parquet shards.
# For the same split and RANDOM_SEED, all modes use the same sample order.
#
# num          = samples that receive the selected mode processing.
# absolute_num = total selected samples written for this mode.
# Remaining rows contain only the original LaTeX code; mode outputs are None.
SPLITS: dict[str, list[dict]] = {
    "our_dataset_benchmark": [
        {
            "type": "full_cleaning",
            "num": 550,
            "absolute_num": 30_000,
        },
        {
            "type": "simple_vlm_description",
            "num": 550,
            "absolute_num": 30_000,
        },
        {
            "type": "deterministic_cleaning",
            "num": 550,
            "absolute_num": 30_000,
        },
    ],
}

HF_REPO_ID = "loss-boss/tikz-dataset"
HF_CACHE_DIR = Path("../hf_cache")
TOKENIZER_CACHE_DIR = Path("../tokenizer_cache")
OUTPUT_DIR = Path("../data/clean_parquets")
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

ROWS_PER_PARQUET = 50_000
ARROW_WRITE_BATCH_SIZE = 128
RANDOM_SEED = 42
OVERWRITE_MODE_OUTPUT = True

# The model supports 8192 tokens; 8000 leaves a small safety buffer.
TOKENIZER_MODEL = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
MAX_LATEX_TOKENS = 8_000

# Input / output columns
IMAGE_WITH_TEXT_COL = "image_with_text"
CODE_WITH_TEXT_COL = "code_with_text"
DESCRIPTION_WITH_TEXT_COL = "llm_description_with_text"

IMAGE_WITHOUT_TEXT_FULL_COL = "image_without_text_full"
CODE_WITHOUT_TEXT_FULL_COL = "code_without_text_full"
DESCRIPTION_WITHOUT_TEXT_FULL_COL = "llm_description_without_text_full"

IMAGE_WITHOUT_TEXT_DETERMINISTIC_COL = "image_without_text_deterministic"
CODE_WITHOUT_TEXT_DETERMINISTIC_COL = "code_without_text_deterministic"
DESCRIPTION_WITHOUT_TEXT_DETERMINISTIC_COL = (
    "llm_description_without_text_deterministic"
)

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3-coder:30b-a3b-q4_K_M"
OLLAMA_TIMEOUT_SECONDS = 1_800
OLLAMA_RETRIES = 2
OLLAMA_KEEP_ALIVE = "10m"
OLLAMA_NUM_CTX = 32_768
OLLAMA_CLEAN_NUM_PREDICT = 32_768
OLLAMA_DESCRIPTION_NUM_PREDICT = 512
CLEANING_PROMPT_PATH = PROMPT_DIR / "llm_cleaning.txt"
DESCRIPTION_PROMPT_PATH = PROMPT_DIR / "llm_description_with_code.txt"

# Rendering / blank-image validation
LATEX_BIN_DIR = "/usr/local/texlive/2026/bin/x86_64-linux"
LATEX_ENGINES = "pdflatex,lualatex,xelatex"
LATEX_TIMEOUT_SECONDS = 45
LATEX_DPI = 400
REFERENCE_IMAGE_SIZE = 512
WHITE_PIXEL_THRESHOLD = 250
MIN_INK_FRACTION = 0.002


def apply_environment() -> None:
    """Apply cache and renderer settings before importing dependencies."""

    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")

    if LATEX_BIN_DIR:
        current_path = os.environ.get("PATH", "")
        if LATEX_BIN_DIR not in current_path.split(":"):
            os.environ["PATH"] = f"{LATEX_BIN_DIR}:{current_path}"

    os.environ["LATEX_ENGINES"] = LATEX_ENGINES
    os.environ["LATEX_TIMEOUT"] = str(LATEX_TIMEOUT_SECONDS)
    os.environ["LATEX_DPI"] = str(LATEX_DPI)
    os.environ["REF_IMAGE_SIZE"] = str(REFERENCE_IMAGE_SIZE)
    os.environ["LATEX_DISABLE_PAGE_NUMBERS"] = "true"
