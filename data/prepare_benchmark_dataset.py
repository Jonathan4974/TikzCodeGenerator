import math
from pathlib import Path

import pandas as pd


base = Path("/usr/prakt/s0030/projects/data/benchmark_data")
df = pd.read_parquet(base / "datikz-benchmark.parquet")

image_dir = base / "images"
code_dir = base / "references"
caption_dir = base / "captions"
split_dir = base / "manifest_splits"

for directory in [image_dir, code_dir, caption_dir, split_dir]:
    directory.mkdir(exist_ok=True)

manifest_rows = []

for row_index, row in df.iterrows():
    stem = f"row_{row_index:08d}"

    image_name = f"{stem}.png"
    code_name = f"{stem}.txt"
    caption_name = f"{stem}.txt"

    image = row["image"]
    image_bytes = image["bytes"] if isinstance(image, dict) else image

    image_path = image_dir / image_name
    code_path = code_dir / code_name
    caption_path = caption_dir / caption_name

    image_path.write_bytes(image_bytes)
    code_path.write_text(str(row["code"]), encoding="utf-8")
    caption_path.write_text(str(row["caption"]), encoding="utf-8")

    manifest_rows.append({
        "reference_image": f"{base}/images/{image_name}",
        "reference_code": f"{base}/references/{code_name}",
        "caption": f"file:///{base}/captions/{caption_name}",
        "input_image_format_for_vlm": f"file:///{base}/images/{image_name}",
    })

manifest = pd.DataFrame(manifest_rows)

manifest_path = base / "image_manifest.csv"
manifest.to_csv(manifest_path, index=False)

# Create a small random manifest for quick tests.
n = 100

if n > len(manifest):
    raise ValueError(
        f"n={n} ist größer als die Anzahl der verfügbaren Zeilen: {len(manifest)}"
    )

small_manifest = manifest.sample(
    n=n,
    replace=False,
    random_state=None,
)

small_manifest_path = base / "small_manifest.csv"
small_manifest.to_csv(small_manifest_path, index=False)

# Split the full manifest into multiple parts for separate Slurm jobs.
num_splits = 10
rows_per_split = math.ceil(len(manifest) / num_splits)

for split_index in range(num_splits):
    start = split_index * rows_per_split
    end = start + rows_per_split

    manifest_split = manifest.iloc[start:end]
    split_path = split_dir / f"image_manifest_{split_index + 1}.csv"

    manifest_split.to_csv(split_path, index=False)

print("Finished!!!")
print(f"Full manifest: {len(manifest)} rows")
print(f"Full manifest path: {manifest_path}")
print(f"Small manifest: {len(small_manifest)} rows")
print(f"Small manifest path: {small_manifest_path}")
print(f"Manifest splits: {num_splits}")
print(f"Manifest split directory: {split_dir}")

for split_index in range(num_splits):
    split_path = split_dir / f"image_manifest_{split_index + 1}.csv"
    split_rows = pd.read_csv(split_path)
    print(f"  {split_path.name}: {len(split_rows)} rows")