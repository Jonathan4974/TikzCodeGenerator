import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


INSTRUCTION = """Take this image and generate a complete LaTeX TikZ document.

Visual description:
__VLM_DESCRIPTION__

Return only LaTeX.
Start with \\documentclass.
Stop after \\end{document}.
"""


def clean_code(text: str) -> str:
    text = str(text).strip()

    start = r"\documentclass"
    if start in text:
        text = text[text.index(start):]

    end = r"\end{document}"
    if end in text:
        text = text[: text.index(end) + len(end)]

    return text.strip()


class DaTikZDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg.dataset_path)
        self.manifest = self.root / "manifest.csv"

        with open(self.manifest, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        if cfg.num_examples is not None:
            self.rows = self.rows[:cfg.num_examples]

    def __len__(self):
        return len(self.rows)

    def _path(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    def __getitem__(self, idx):
        row = self.rows[idx]

        image_path = self._path(row[self.cfg.image_column])
        code_path = self._path(row[self.cfg.code_column])
        vlm_description_path = self._path(row[self.cfg.vlm_description_column])

        image = Image.open(image_path).convert("RGB")

        tikz_code = clean_code(code_path.read_text(encoding="utf-8"))
        vlm_description = vlm_description_path.read_text(encoding="utf-8").strip()

        text_content = INSTRUCTION.replace(
            "__VLM_DESCRIPTION__",
            vlm_description,
        )

        prompt = (
            "<bos><|turn>user\n"
            "<|image|>"
            f"{text_content}"
            "<turn|>\n"
            "<|turn>model\n"
        )

        return {
            "prompt": prompt,
            "image": image,
            "answer": tikz_code,
            "image_path": str(image_path),
        }