import math
import re
from pathlib import Path

import pandas as pd


benchmark_base = Path("/usr/prakt/s0030/projects/data/benchmark_data/") 
parquet_base = Path("/usr/prakt/s0030/projects/data/paqrquet_files/datikz-v2")

parquet_files = sorted(parquet_base.glob("*.parquet"))

train_files = [
    path for path in parquet_files
    if "train" in path.name.lower()
]

test_files = [
    path for path in parquet_files
    if "test" in path.name.lower()
]


corpus_dir = benchmark_base / "benchmark_corpus"
image_dir = benchmark_base / "images"
code_dir = benchmark_base / "references"
#caption_dir = benchmark_base / "captions"
split_dir = benchmark_base / "manifest_splits"

for directory in [corpus_dir, image_dir, code_dir, split_dir]:
    directory.mkdir(parents=True, exist_ok=True)


def safe_stem(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name)


def extract_image_bytes(image):
    return image["bytes"] if isinstance(image, dict) else image


########build the corpus for crystalbleu #############
corpus_count = 0

for parquet_path in train_files:
    df = pd.read_parquet(parquet_path)
    file_stem = safe_stem(parquet_path.stem)

    for row_number, (_, row) in enumerate(df.iterrows()):
        stem = f"{file_stem}_row_{row_number:08d}"
        code_name = f"{stem}.txt"

        code_path = corpus_dir / code_name
        code_path.write_text(str(row["code"]), encoding="utf-8")

        corpus_count += 1


########build the test dataset#############
manifest_rows = []
test_count = 0

for parquet_path in test_files:
    df = pd.read_parquet(parquet_path)
    file_stem = safe_stem(parquet_path.stem)

    for row_number, (_, row) in enumerate(df.iterrows()):
        stem = f"{file_stem}_row_{row_number:08d}"

        image_name = f"{stem}.png"
        code_name = f"{stem}.txt"
        #caption_name = f"{stem}.txt"

        image = row["image"]
        image_bytes = extract_image_bytes(image)

        image_path = image_dir / image_name
        code_path = code_dir / code_name
        #caption_path = caption_dir / caption_name

        image_path.write_bytes(image_bytes)
        code_path.write_text(str(row["code"]), encoding="utf-8")
        #caption_path.write_text(str(row["caption"]), encoding="utf-8")

        manifest_rows.append({
            "reference_image": f"{image_dir}/{image_name}",
            "reference_code": f"{code_dir}/{code_name}",
            "input_image_format_for_vlm": f"file:///{image_dir}/{image_name}",
        })

        test_count += 1


manifest = pd.DataFrame(manifest_rows)

manifest_path = benchmark_base / "image_manifest.csv"
manifest.to_csv(manifest_path, index=False)


##############manifest splits########################

num_splits = 10
rows_per_split = math.ceil(len(manifest) / num_splits)

for split_index in range(num_splits):
    start = split_index * rows_per_split
    end = start + rows_per_split

    manifest_split = manifest.iloc[start:end]
    split_path = split_dir / f"image_manifest_{split_index + 1}.csv"

    manifest_split.to_csv(split_path, index=False)


print("Finished!!!")
