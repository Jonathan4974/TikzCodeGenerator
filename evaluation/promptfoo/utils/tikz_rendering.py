import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops


class TikzRenderError(RuntimeError):
    pass


def normalize_png_canvas(
    image_path: str | Path,
    target_size: tuple[int, int],
    output_path: str | Path | None = None,
    background: str = "white",
    upscale: bool = True,
) -> Path:
    image_path = Path(image_path)
    output_path = Path(output_path) if output_path is not None else image_path

    img = Image.open(image_path).convert("RGBA")

    target_w, target_h = target_size
    img_w, img_h = img.size

    scale = min(target_w / img_w, target_h / img_h)

    if not upscale:
        scale = min(scale, 1.0)

    new_w = max(1, int(round(img_w * scale)))
    new_h = max(1, int(round(img_h * scale)))

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", target_size, background)

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2

    canvas.alpha_composite(img, dest=(x, y))
    canvas.convert("RGB").save(output_path)

    return output_path


def extract_tikz_libraries(tex_code: str) -> list[str]:
    libraries = []

    for match in re.finditer(r"\\usetikzlibrary\s*\{([^}]*)\}", tex_code):
        for lib in match.group(1).split(","):
            lib = lib.strip()
            if lib and lib not in libraries:
                libraries.append(lib)

    return libraries


def extract_tikzpicture_blocks(tex_code: str) -> list[str]:
    pattern = re.compile(
        r"\\begin\s*\{tikzpicture\}.*?\\end\s*\{tikzpicture\}",
        re.DOTALL,
    )

    return [match.group(0) for match in pattern.finditer(tex_code)]


def remove_document_wrapper(tex_code: str) -> str:
    match = re.search(
        r"\\begin\s*\{document\}(.*)\\end\s*\{document\}",
        tex_code,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return match.group(1).strip() if match else tex_code.strip()


def patch_common_color_names(tex_code: str) -> str:
    color_defs = r"""
\definecolor{pdfcitecolor}{rgb}{0,0,1}
\definecolor{pdflinkcolor}{rgb}{0,0,1}
\definecolor{pdfanchorcolor}{rgb}{0,0,1}
\definecolor{pdfurlcolor}{rgb}{0,0,1}
"""

    needed = any(
        name in tex_code
        for name in [
            "pdfcitecolor",
            "pdflinkcolor",
            "pdfanchorcolor",
            "pdfurlcolor",
        ]
    )

    already_defined = r"\definecolor{pdfcitecolor}" in tex_code

    if needed and not already_defined and r"\begin{document}" in tex_code:
        tex_code = tex_code.replace(
            r"\begin{document}",
            color_defs + "\n" + r"\begin{document}",
            1,
        )

    return tex_code


def disable_page_numbers(tex_code: str) -> str:
    if r"\begin{document}" not in tex_code:
        return tex_code

    if r"\pagestyle{empty}" not in tex_code:
        tex_code = tex_code.replace(
            r"\begin{document}",
            r"\pagestyle{empty}" + "\n" + r"\begin{document}",
            1,
        )

    if r"\thispagestyle{empty}" not in tex_code:
        tex_code = tex_code.replace(
            r"\begin{document}",
            r"\begin{document}" + "\n" + r"\thispagestyle{empty}",
            1,
        )

    return tex_code


def build_standalone_document(
    tex_code: str,
    border: str = "2pt",
    mode: str = "tikz",
) -> str:
    libraries = extract_tikz_libraries(tex_code)

    if mode == "tikz":
        blocks = extract_tikzpicture_blocks(tex_code)
        if blocks:
            body = "\n\n".join(blocks)
        else:
            body = remove_document_wrapper(tex_code)

    elif mode == "body":
        body = remove_document_wrapper(tex_code)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    library_line = ""
    if libraries:
        library_line = r"\usetikzlibrary{" + ",".join(libraries) + "}"

    return rf"""
\documentclass[tikz,border={border}]{{standalone}}

\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{amsfonts}}
\usepackage{{amsthm}}
\usepackage{{mathrsfs}}
\usepackage{{eufrak}}
\usepackage{{textcomp}}
\usepackage{{xcolor}}
\usepackage{{color}}
\usepackage{{tikz}}
\usepackage{{pgfplots}}
\usepackage{{pgfplotstable}}
\usepackage{{colortbl}}
\usepackage{{array}}
\usepackage{{graphicx}}
\usepackage{{ifthen}}

\pgfplotsset{{compat=1.18}}

{library_line}

\pagestyle{{empty}}

\begin{{document}}
{body}
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

    img.crop((left, top, right, bottom)).save(output_path)

    return output_path


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def compile_candidate_to_png(
    tex_code: str,
    output_path: str | Path,
    target_size: tuple[int, int],
    dpi: int = 200,
    engine: str = "pdflatex",
    crop_pdf: bool = True,
    crop_png: bool = True,
    pdf_margin: int = 0,
    png_padding: int = 4,
    png_tolerance: int = 10,
    normalize_canvas: bool = True,
    upscale_canvas: bool = True,
) -> Path:
    output_path = Path(output_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        tex_path = tmp_dir / "figure.tex"
        pdf_path = tmp_dir / "figure.pdf"
        cropped_pdf_path = tmp_dir / "figure-crop.pdf"
        png_prefix = tmp_dir / "figure"

        tex_path.write_text(tex_code, encoding="utf-8")

        latex_result = run_command(
            [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=tmp_dir,
        )

        if latex_result.returncode != 0:
            log_path = tmp_dir / "figure.log"
            log_text = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.exists()
                else ""
            )

            raise TikzRenderError(
                f"{engine} failed.\n\n"
                f"STDOUT:\n{latex_result.stdout[-3000:]}\n\n"
                f"STDERR:\n{latex_result.stderr[-3000:]}\n\n"
                f"LOG:\n{log_text[-5000:]}"
            )

        if not pdf_path.exists():
            raise TikzRenderError(f"{engine} did not create figure.pdf")

        pdf_to_render = pdf_path

        if crop_pdf and shutil.which("pdfcrop"):
            crop_result = run_command(
                [
                    "pdfcrop",
                    "--margins",
                    str(pdf_margin),
                    str(pdf_path),
                    str(cropped_pdf_path),
                ],
                cwd=tmp_dir,
            )

            if crop_result.returncode == 0 and cropped_pdf_path.exists():
                pdf_to_render = cropped_pdf_path

        render_result = run_command(
            [
                "pdftoppm",
                "-png",
                "-singlefile",
                "-r",
                str(dpi),
                str(pdf_to_render),
                str(png_prefix),
            ],
            cwd=tmp_dir,
        )

        if render_result.returncode != 0:
            raise TikzRenderError(
                "pdftoppm failed.\n\n"
                f"STDOUT:\n{render_result.stdout[-3000:]}\n\n"
                f"STDERR:\n{render_result.stderr[-3000:]}"
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

        if normalize_canvas:
            normalize_png_canvas(
                image_path=generated_png,
                target_size=target_size,
                output_path=generated_png,
                upscale=upscale_canvas,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(generated_png.read_bytes())

    return output_path


def make_render_candidates(
    tex_code: str,
    border: str = "2pt",
) -> list[tuple[str, str]]:
    candidates = []

    tikz_blocks = extract_tikzpicture_blocks(tex_code)

    if tikz_blocks:
        standalone_tikz = build_standalone_document(
            tex_code=tex_code,
            border=border,
            mode="tikz",
        )
        candidates.append(("standalone_tikzpicture_only", standalone_tikz))

    standalone_body = build_standalone_document(
        tex_code=tex_code,
        border=border,
        mode="body",
    )
    candidates.append(("standalone_document_body", standalone_body))

    original = disable_page_numbers(patch_common_color_names(tex_code))
    candidates.append(("original", original))

    return candidates


def render_tex_to_png(
    tex_code: str,
    output_path: str | Path,
    dpi: int = 200,
    border: str = "2pt",
    engines: tuple[str, ...] = ("pdflatex", "lualatex", "xelatex"),
    crop_pdf: bool = True,
    crop_png: bool = True,
    pdf_margin: int = 0,
    png_padding: int = 4,
    png_tolerance: int = 10,
    normalize_canvas: bool = True,
    upscale_canvas: bool = True,
) -> Path:
    errors = []

    ref_image_size = int(os.getenv("REF_IMAGE_SIZE", "384"))
    target_size = (ref_image_size, ref_image_size)

    available_engines = [engine for engine in engines if shutil.which(engine)]

    if not available_engines:
        raise TikzRenderError(
            f"No LaTeX engine found. Tried: {', '.join(engines)}"
        )

    candidates = make_render_candidates(
        tex_code=tex_code,
        border=border,
    )

    for candidate_name, candidate_tex in candidates:
        for engine in available_engines:
            try:
                return compile_candidate_to_png(
                    tex_code=candidate_tex,
                    output_path=output_path,
                    target_size=target_size,
                    dpi=dpi,
                    engine=engine,
                    crop_pdf=crop_pdf,
                    crop_png=crop_png,
                    pdf_margin=pdf_margin,
                    png_padding=png_padding,
                    png_tolerance=png_tolerance,
                    normalize_canvas=normalize_canvas,
                    upscale_canvas=upscale_canvas,
                )

            except Exception as e:
                errors.append(
                    f"\n--- candidate={candidate_name}, engine={engine} ---\n{e}"
                )

    raise TikzRenderError(
        "All render strategies failed.\n"
        + "\n".join(errors[-3:])
    )