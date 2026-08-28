"""Compile TikZ code into PNG image. Used by TikZ demo backend."""

import os
import re
import shutil
import subprocess
import tempfile

from pathlib import Path
from PIL import Image, ImageChops
import numpy as np

import config

# ============================================================
# Exceptions
# ============================================================
class TikzCompileError(RuntimeError):
    def __init__(self, message, metrics=None):
        super().__init__(message)
        self.metrics = metrics or {}


# ============================================================
# Run command
# ============================================================
def run_command(cmd, cwd):
    timeout = int(os.getenv("LATEX_TIMEOUT", "45"))

    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            cmd,
            returncode=124,
            stdout="",
            stderr="LaTeX timeout"
        )


# ============================================================
# Latex log analysis
# ============================================================
def count_latex_issues(log):
    errors = 0
    warnings = 0
    badboxes = 0

    for line in log.splitlines():
        line = line.strip()

        if line.startswith("!"):
            errors += 1

        elif re.match(r"^\.?/?.*\.tex:\d+:",line):
            errors += 1

        if "Warning:" in line and not line.startswith("Package rerunfilecheck Warning:"):
            warnings += 1

        if line.startswith("Overfull \\") or line.startswith("Underfull \\"):
            badboxes += 1


    return {
        "latex_errors": errors,
        "latex_warnings": warnings,
        "latex_badboxes": badboxes,
    }

def get_pdf_page_count(pdf_path: Path, cwd: Path) -> int | None:
    if not shutil.which("pdfinfo"):
        return None

    result = run_command(["pdfinfo", str(pdf_path)], cwd)

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None

    return None

# ============================================================
# Tex cleaning
# ============================================================
def clean_markdown_tex(tex_code: str) -> str:
    if not tex_code:
        return ""

    tex_code = re.sub(
        r"^\s*```[a-zA-Z]*\s*\n?",
        "",
        tex_code,
        flags=re.IGNORECASE,
    )
    tex_code = re.sub(r"\n?\s*```\s*$", "", tex_code)

    return tex_code.strip()


def clean_code(text: str) -> str:
    code = str(text).strip()
    start, end = r"\documentclass", r"\end{document}"
    if start in code:
        code = code[code.index(start):]
    if end in code:
        code = code[:code.index(end) + len(end)]
    return code.strip()


def tex_cleaning(tex_code):
    tex_code = clean_markdown_tex(tex_code)

    return clean_code(tex_code)


# ============================================================
# Image processing
# ============================================================
def check_blank(path):
    img = Image.open(path).convert("L")
    pixels = list(img.getdata())
    ink = sum(p < config.WHITE_PIXEL_THRESHOLD for p in pixels)
    ratio = ink / len(pixels)

    return ratio


# ============================================================
# Main compiler
# ============================================================
def compile_tikz(tex_code: str, output_path: str):
    output_path = Path(output_path)
    attempts = []
    overall_blank = False
    halt_modes = [True, False]

    for engine in config.LATEX_ENGINES:
        if shutil.which(engine) is None:
            continue

        for halt in halt_modes:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                tex = tmp / "figure.tex"
                pdf = tmp / "figure.pdf"
                png = tmp / "figure.png"
                log = tmp / "figure.log"

                tex.write_text(tex_code,encoding="utf-8")

                cmd = [engine, "-interaction=nonstopmode", "-file-line-error"]
                if halt:
                    cmd.append("-halt-on-error")
                cmd.append(tex.name)
                result = run_command(cmd, tmp)

                log_text = ""
                if log.exists():
                    log_text = log.read_text(encoding="utf-8", errors="replace")

                attempt = {
                    "engine": engine,
                    "halt_on_error": halt,
                    "returncode": result.returncode,
                    "pdf_created": pdf.exists(),
                    "png_created": False,
                    "blank_image": False,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "log": log_text,
                }

                if not pdf.exists():
                    attempts.append(attempt)
                    continue

                # --------------------------------------------------
                # PDF -> PNG
                # --------------------------------------------------
                convert = run_command(
                    [
                        "pdftoppm",
                        "-png",
                        "-singlefile",
                        "-r",
                        str(config.LATEX_DPI),
                        str(pdf),
                        str(tmp / "figure")
                    ],
                    tmp
                )

                attempt["stdout"] += "\n" + convert.stdout
                attempt["stderr"] += "\n" + convert.stderr

                if convert.returncode != 0 or not png.exists():
                    attempts.append(attempt)
                    continue
                attempt["png_created"] = True

                # --------------------------------------------------
                # Blank image check
                # --------------------------------------------------
                ink_fraction = check_blank(png)
                attempt["ink_fraction"] = ink_fraction
                if ink_fraction < config.MIN_INK_FRACTION:
                    attempt["blank_image"] = True
                    overall_blank = True
                    attempts.append(attempt)
                    continue

                # --------------------------------------------------
                # SUCCESS
                # --------------------------------------------------
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(png, output_path)

                attempts.append(attempt)
                return {
                    "success": True,
                    "png_path": str(output_path),
                    "overall_blank": False,
                    "successful_engine": engine,
                    "attempts": attempts,
                }

    # ==========================================================
    # if all failed
    # ==========================================================
    return {
        "success": False,
        "png_path": None,
        "overall_blank": overall_blank,
        "successful_engine": None,
        "attempts": attempts,
    }