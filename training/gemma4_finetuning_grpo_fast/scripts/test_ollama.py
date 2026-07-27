from pathlib import Path

from ollama import Client


MODEL = "gemma4-tikz-sft:latest"
IMAGE_PATH = Path("/root/projects/00000004.png")

PROMPT = """As a LaTeX graphics expert, translate the image into TikZ code suitable for academic publications.
Focus on recreating geometric precision, typographic elements, and color schemes.
The code must be compilable, efficient, and maintain the original image's visual fidelity for professional document integration.
Return only the full LaTeX document.
Do not use markdown fences.
Do not add explanations."""

DESCRIPTION = """The image depicts a schematic diagram illustrating a layered structure composed of a gray trapezoidal cone-like base labeled $X_t$, with a white upper trapezoid overlay and black circular nodes representing elements such as $u_1$ through $u_9$ and $h_1$ through $h_3$."""

full_prompt = (
    PROMPT
    + "\n\nAdditionally, here is a description of the image "
      "with some creation hints:\n"
    + DESCRIPTION
)

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

client = Client(host="http://localhost:11434")

response = client.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": full_prompt,
            "images": [str(IMAGE_PATH)],
        }
    ],
    options={
        "temperature": 0,
        "num_ctx": 9216,
        "num_predict": 8192,
    },
)

latex_code = response["message"]["content"]

Path("generated.tex").write_text(latex_code, encoding="utf-8")

print(latex_code)