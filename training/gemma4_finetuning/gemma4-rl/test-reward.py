from pathlib import Path
from PIL import Image

from config import TrainingConfig
from rewards import TikZReward


cfg = TrainingConfig()
TEST_DIR = Path("/data/test-reward/case_001")


def main():
    reward_fn = TikZReward(cfg)

    image = Image.open(TEST_DIR / "input.png").convert("RGB")
    generated_code = (TEST_DIR / "generated.tex").read_text(encoding="utf-8")
    reference_code = (TEST_DIR / "reference.tex").read_text(encoding="utf-8")

    scores = reward_fn(
        completions=[generated_code],
        answer=[reference_code],
        images=[image],
    )

    print(f"score={scores[0]:.4f}")


if __name__ == "__main__":
    main()