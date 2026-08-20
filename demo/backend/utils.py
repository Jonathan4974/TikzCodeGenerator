from pathlib import Path
from PIL import Image, ImageChops

IMAGE_SIZE = 512

def render_instruction(prompt: str, description: str | None) -> str:
    final_prompt = prompt.strip()
    if description and description.strip():
        final_prompt += (
            "\n\nAdditionally, here is a description of the image "
            "with some creation hints:\n"
            f"{description.strip()}"
        )
    return final_prompt


def normalize_canvas(path: Path) -> None:
    image = Image.open(path).convert("RGBA")
    scale = min(IMAGE_SIZE / image.width, IMAGE_SIZE / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), "white")
    canvas.alpha_composite(image, ((IMAGE_SIZE - width) // 2, (IMAGE_SIZE - height) // 2))
    canvas.convert("RGB").save(path) 


def crop_png(path, padding=4):
    path = Path(path)

    img = Image.open(path).convert("RGB")
    bg = Image.new("RGB", img.size, "white")

    diff = ImageChops.difference(img, bg)
    diff = diff.point(lambda p: 255 if p > 10 else 0)

    bbox = diff.getbbox()
    if bbox:
        l, t, r, b = bbox
        l = max(l - padding, 0)
        t = max(t - padding, 0)
        r = min(r + padding, img.width)
        b = min(b + padding, img.height)

        img.crop((l, t, r, b)).save(path)