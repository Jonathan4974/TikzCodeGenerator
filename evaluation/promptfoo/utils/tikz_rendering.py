import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops


class TikzRenderError(RuntimeError):
    pass


def extract_tikz_libraries(tex_code: str) -> list[str]:
    libraries: list[str] = []

    pattern = re.compile(r"\\usetikzlibrary\s*\{([^}]*)\}")

    for match in pattern.finditer(tex_code):
        libs = match.group(1)
        for lib in libs.split(","):
            lib = lib.strip()
            if lib and lib not in libraries:
                libraries.append(lib)

    return libraries


def extract_first_tikzpicture(tex_code: str) -> str:
    pattern = re.compile(
        r"\\begin\s*\{tikzpicture\}.*?\\end\s*\{tikzpicture\}",
        re.DOTALL,
    )

    match = pattern.search(tex_code)

    if not match:
        raise TikzRenderError("No tikzpicture environment found in tex_code")

    return match.group(0)


def build_standalone_tikz_document(
    tex_code: str,
    border: str = "2pt",
) -> str:
    tikzpicture = extract_first_tikzpicture(tex_code)
    libraries = extract_tikz_libraries(tex_code)

    library_line = ""
    if libraries:
        library_line = r"\usetikzlibrary{" + ",".join(libraries) + "}"

    return rf"""
\documentclass[tikz,border={border}]{{standalone}}

\usepackage{{tikz}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{amsfonts}}
\usepackage{{mathrsfs}}
\usepackage{{eufrak}}
\usepackage{{textcomp}}
\usepackage{{xcolor}}

{library_line}

\pagestyle{{empty}}

\begin{{document}}
{tikzpicture}
\end{{document}}
"""


def crop_png_to_content(
    image_path: str | Path,
    output_path: str | Path | None = None,
    background: str = "white",
    padding: int = 4,
    tolerance: int = 10,
) -> Path:
    image_path = Path(image_path)
    output_path = Path(output_path) if output_path is not None else image_path

    img = Image.open(image_path).convert("RGB")
    bg = Image.new("RGB", img.size, background)

    diff = ImageChops.difference(img, bg)
    diff = diff.point(lambda p: 255 if p > tolerance else 0)

    bbox = diff.getbbox()

    if bbox is None:
        img.save(output_path)
        return output_path

    left, top, right, bottom = bbox

    left = max(left - padding, 0)
    top = max(top - padding, 0)
    right = min(right + padding, img.width)
    bottom = min(bottom + padding, img.height)

    cropped = img.crop((left, top, right, bottom))
    cropped.save(output_path)

    return output_path


def compile_tex_to_png(
    tex_code: str,
    output_path: str | Path,
    dpi: int = 200,
    crop_png: bool = True,
    png_padding: int = 4,
    png_tolerance: int = 10,
) -> Path:
    output_path = Path(output_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        tex_path = tmp_dir / "figure.tex"
        pdf_path = tmp_dir / "figure.pdf"
        png_prefix = tmp_dir / "figure"

        tex_path.write_text(tex_code, encoding="utf-8")

        latex_result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=tmp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if latex_result.returncode != 0:
            log_path = tmp_dir / "figure.log"
            log_text = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.exists()
                else ""
            )

            raise TikzRenderError(
                "pdflatex failed.\n\n"
                f"STDOUT:\n{latex_result.stdout}\n\n"
                f"STDERR:\n{latex_result.stderr}\n\n"
                f"LOG:\n{log_text[-4000:]}"
            )

        if not pdf_path.exists():
            raise TikzRenderError("pdflatex did not create figure.pdf")

        render_result = subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-singlefile",
                "-r",
                str(dpi),
                str(pdf_path),
                str(png_prefix),
            ],
            cwd=tmp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if render_result.returncode != 0:
            raise TikzRenderError(
                "pdftoppm failed.\n\n"
                f"STDOUT:\n{render_result.stdout}\n\n"
                f"STDERR:\n{render_result.stderr}"
            )

        generated_png = tmp_dir / "figure.png"

        if not generated_png.exists():
            raise TikzRenderError("pdftoppm did not create figure.png")

        if crop_png:
            crop_png_to_content(
                image_path=generated_png,
                output_path=generated_png,
                padding=png_padding,
                tolerance=png_tolerance,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(generated_png.read_bytes())

    return output_path


def render_tex_to_png(
    tex_code: str,
    output_path: str | Path,
    dpi: int = 200,
    extract_tikz: bool = True,
    border: str = "2pt",
    crop_png: bool = True,
    png_padding: int = 4,
    png_tolerance: int = 10,
) -> Path:

    if extract_tikz:
        tex_code = build_standalone_tikz_document(
            tex_code=tex_code,
            border=border,
        )

    return compile_tex_to_png(
        tex_code=tex_code,
        output_path=output_path,
        dpi=dpi,
        crop_png=crop_png,
        png_padding=png_padding,
        png_tolerance=png_tolerance,
    )