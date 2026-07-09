import os
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageChops
from pf_utils.tikz_rendering import render_tex_to_png


os.environ["LATEX_DPI"] = "2000"
os.environ["REF_IMAGE_SIZE"] = "512"
os.environ["LATEX_NORMALIZE_CANVAS"] = "true"
os.environ["LATEX_UPSCALE_CANVAS"] = "true"
os.environ["LATEX_CROP_PDF"] = "true"
os.environ["LATEX_CROP_PNG"] = "true"


DATASET_PATH = Path("/data")
MANIFEST_PATH = DATASET_PATH / "hardest_manifest.csv"
OUTPUT_DIR = Path("/data/hardest_preview")

N = 50


def resolve(root: Path, p: str) -> Path:
    p = Path(str(p))
    return p if p.is_absolute() else root / p


def clean_code(text: str) -> str:
    text = str(text).strip()

    fence_match = re.search(
        r"```(?:latex|tex|tikz)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        text = fence_match.group(1).strip()

    text = (
        text.replace("```latex", "")
        .replace("```tex", "")
        .replace("```tikz", "")
        .replace("```", "")
        .strip()
    )

    start = r"\documentclass"
    if start in text:
        text = text[text.index(start):]

    end = r"\end{document}"
    if end in text:
        text = text[: text.index(end) + len(end)]

    return text.strip()


def is_mostly_white(path: Path, threshold=250, max_nonwhite_ratio=0.002):
    img = Image.open(path).convert("L")
    data = img.getdata()
    total = len(data)
    nonwhite = sum(1 for p in data if p < threshold)
    ratio = nonwhite / total
    return ratio < max_nonwhite_ratio, ratio


def add_label(img: Image.Image, text: str) -> Image.Image:
    label_h = 48
    out = Image.new("RGB", (img.width, img.height + label_h), "white")
    out.paste(img.convert("RGB"), (0, 0))

    draw = ImageDraw.Draw(out)
    draw.text((8, img.height + 10), text, fill="black")

    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)
    df = df.sort_values("sft_loss", ascending=False).reset_index(drop=True)

    rendered = 0
    blank = 0
    skipped = 0

    for idx, row in df.head(N).iterrows():
        code_path = resolve(DATASET_PATH, row["code_path"])
        tex_code = clean_code(code_path.read_text(encoding="utf-8"))

        out_png = OUTPUT_DIR / f"rank_{idx:04d}_loss_{row['sft_loss']:.4f}.png"

        metrics = {}
        try:
            render_tex_to_png(
                tex_code=tex_code,
                output_path=out_png,
                metrics=metrics,
                create_ds=True,
            )

            is_blank, nonwhite_ratio = is_mostly_white(out_png)

            if is_blank:
                blank += 1
                print(
                    f"[blank] rank={idx} "
                    f"loss={row['sft_loss']:.4f} "
                    f"nonwhite_ratio={nonwhite_ratio:.6f} "
                    f"path={code_path}"
                )
                continue

            img = Image.open(out_png).convert("RGB")
            img = add_label(
                img,
                f"rank={idx} loss={row['sft_loss']:.4f} tokens={int(row['sft_answer_tokens'])}",
            )
            img.save(out_png)

            rendered += 1
            print(f"[ok] {out_png} nonwhite_ratio={nonwhite_ratio:.6f}")

        except Exception as e:
            skipped += 1
            print(f"[skip] rank={idx} error={e} path={code_path}")

    print()
    print(f"Rendered: {rendered}")
    print(f"Blank:    {blank}")
    print(f"Skipped:  {skipped}")
    print(f"Output:   {OUTPUT_DIR}")


if __name__ == "__main__":
    main()