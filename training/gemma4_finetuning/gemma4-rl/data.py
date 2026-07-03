from datasets import load_dataset


INSTRUCTION = """Take this image and write the LaTeX/TikZ code for it.
Return only complete compilable LaTeX code.
Do not explain anything.
Do not use Markdown.
"""


class DaTikZDatasetBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def prepare_image(self, image):
        image = image.resize((self.cfg.image_size, self.cfg.image_size))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def make_conversation(self, example):
        image = self.prepare_image(example["png_image"])

        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": INSTRUCTION},
                ],
            }
        ]

        return {
            "prompt": prompt,
            "image": image,
            "answer": example["tikz_code"],
        }

    def load(self):
        dataset = load_dataset(self.cfg.dataset_path, split="train")

        if self.cfg.num_examples is not None:
            dataset = dataset.select(range(self.cfg.num_examples))

        dataset = dataset.map(
            self.make_conversation,
            remove_columns=dataset.column_names,
        )

        return dataset