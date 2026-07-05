import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


INSTRUCTION = """Take this image and write the LaTeX/TikZ code for it.

VLM description:
__VLM_DESCRIPTION__

Return only complete compilable LaTeX code.
Do not explain anything.
Do not use Markdown.
Stop immediately after \\end{document}.
"""


class DaTikZDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg.dataset_path)
        self.manifest = self.root / "manifest.csv"

        with open(self.manifest, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        if cfg.num_examples is not None:
            self.rows = self.rows[: cfg.num_examples]

    def __len__(self):
        return len(self.rows)

    def _path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, idx):
        row = self.rows[idx]

        image_path = self._path(row[self.cfg.image_column])
        code_path = self._path(row[self.cfg.code_column])
        vlm_description_path = self._path(row[self.cfg.vlm_description_column])

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.cfg.image_size, self.cfg.image_size))

        tikz_code = code_path.read_text(encoding="utf-8")
        vlm_description = vlm_description_path.read_text(encoding="utf-8").strip()

        text_content = INSTRUCTION.replace(
            "__VLM_DESCRIPTION__",
            vlm_description,
        )

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_content},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": tikz_code},
                    ],
                },
            ]
        }