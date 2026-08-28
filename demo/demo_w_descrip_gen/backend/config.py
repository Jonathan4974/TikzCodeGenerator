"""Configuration for the TikZ Demo."""

from pathlib import Path
import os


# ============================================================
# Project directories
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

UPLOAD_DIR = PROJECT_ROOT / "uploads"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

TEMPLATE_DIR = PROJECT_ROOT / "templates"


# ============================================================
# Ollama
# ============================================================

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# OLLAMA_MODEL = "gemma4:31b-it-q4_K_M"
OLLAMA_MODEL = "gemma4:e2b-it-q4_K_M"

OLLAMA_TIMEOUT = 600


# ============================================================
# LaTeX
# ============================================================

LATEX_BIN_DIR = "/usr/prakt/s0042/projects/full_latex/texlive/2026/bin/x86_64-linux"

LATEX_ENGINES = ["pdflatex", "lualatex", "xelatex",]

LATEX_TIMEOUT = 45

LATEX_DPI = 400


# ============================================================
# Image rendering
# ============================================================

OUTPUT_IMAGE_SIZE = 512

# ============================================================
# Blank image detection
# ============================================================

WHITE_PIXEL_THRESHOLD = 250

MIN_INK_FRACTION = 0.002


# ============================================================
# Apply environment
# ============================================================

def apply_environment():

    """
    Configure environment variables used by the renderer.
    """

    if LATEX_BIN_DIR:
        current = os.environ.get("PATH", "")

        if LATEX_BIN_DIR not in current.split(":"):
            os.environ["PATH"] = LATEX_BIN_DIR + ":" + current

    os.environ["LATEX_TIMEOUT"] = str(LATEX_TIMEOUT)

    os.environ["LATEX_DPI"] = str(LATEX_DPI)

    os.environ["LATEX_ENGINES"] = ",".join(LATEX_ENGINES)

    os.environ["REF_IMAGE_SIZE"] = str(OUTPUT_IMAGE_SIZE)

# ============================================================
# Create directories automatically
# ============================================================

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# other potential config
# ==============================

MAX_UPLOAD_SIZE_MB = 10

DEFAULT_OUTPUT_NAME = "result"

SAVE_HISTORY = False

DEBUG = True