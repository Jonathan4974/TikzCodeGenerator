"""
Builds one contact-sheet image per example, laying out:
  rendered (original) | real human sketch | displacement-only
  ... followed by each UltraSketch prompt variant's output

Run this on the server, in the same env you used for the eval script (just needs PIL/numpy):
    python make_contact_sheets.py

Output: BASE_OUTPUT_DIR/contact_sheets/{i}_contact_sheet.png
Open these in a file browser / image viewer for fast side-by-side comparison.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# --- Config: match these to your eval script -------------------------------
NUM_EXAMPLES = 5
BASE_OUTPUT_DIR = Path("/usr/prakt/$USER/tikzcodegenerator/data/ultrasketch_outputs_diff_prompts")
APPLY_DISPLACEMENT = True

PROMPT_VARIANTS = [
    "baseline",
    "pencil",
    "scientific",
    "minimal",
    "structure_preserve",
    "scientific_labels",
    "light_pencil",
    "anti_artifact",
    "student_notes",
]

THUMB_SIZE = 220       # each cell is THUMB_SIZE x THUMB_SIZE
LABEL_HEIGHT = 24      # space for the text label under each thumbnail
PADDING = 8            # gap between cells
COLS = 4               # how many cells per row in the contact sheet
# ---------------------------------------------------------------------------


def load_thumb(path: Path, size: int) -> Image.Image:
    """Load an image and fit it into a size x size white canvas, preserving aspect ratio."""
    canvas = Image.new("RGB", (size, size), "white")
    if not path.exists():
        # Draw a placeholder so missing files are obvious rather than silently skipped
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, size - 1, size - 1], outline="red", width=2)
        draw.text((10, size // 2 - 10), "MISSING", fill="red")
        return canvas

    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_cell(path: Path, label: str) -> Image.Image:
    """A thumbnail with a text label underneath."""
    cell = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE + LABEL_HEIGHT), "white")
    thumb = load_thumb(path, THUMB_SIZE)
    cell.paste(thumb, (0, 0))

    draw = ImageDraw.Draw(cell)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    # Center the label text
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    x = max(0, (THUMB_SIZE - text_w) // 2)
    draw.text((x, THUMB_SIZE + 4), label, fill="black", font=font)
    return cell


def build_contact_sheet(cells: list[tuple[Path, str]], cols: int, title: str) -> Image.Image:
    rows = (len(cells) + cols - 1) // cols
    cell_w = THUMB_SIZE + PADDING
    cell_h = THUMB_SIZE + LABEL_HEIGHT + PADDING
    title_h = 30

    sheet = Image.new(
        "RGB",
        (cols * cell_w + PADDING, rows * cell_h + PADDING + title_h),
        "white",
    )

    draw = ImageDraw.Draw(sheet)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        title_font = ImageFont.load_default()
    draw.text((PADDING, 6), title, fill="black", font=title_font)

    for idx, (path, label) in enumerate(cells):
        cell = make_cell(path, label)
        row, col = divmod(idx, cols)
        x = PADDING + col * cell_w
        y = title_h + PADDING + row * cell_h
        sheet.paste(cell, (x, y))

    return sheet


def main():
    out_dir = BASE_OUTPUT_DIR / "contact_sheets_combined" # added _combined
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = BASE_OUTPUT_DIR / "ground_truth"
    

    for i in range(NUM_EXAMPLES):
        cells: list[tuple[Path, str]] = [
            (gt_dir / f"{i}_rendered.png", "rendered (input)"),
            (gt_dir / f"{i}_real_sketch.png", "real human sketch"),
        ]
        if APPLY_DISPLACEMENT:
            cells.append((gt_dir / f"{i}_displacement.png", "displacement only"))

        for variant in PROMPT_VARIANTS:
            variant_dir = BASE_OUTPUT_DIR / f"ultrasketch_{variant}"
            cells.append((variant_dir / f"{i}_visual_blend.png", variant))

        sheet = build_contact_sheet(cells, COLS, title=f"Example {i}")
        out_path = out_dir / f"{i}_contact_sheet.png"
        sheet.save(out_path)
        print(f"Saved {out_path}")

    print(f"\nDone. {NUM_EXAMPLES} contact sheets in: {out_dir}")


if __name__ == "__main__":
    main()