import pandas as pd
from pathlib import Path

base = Path("/home/jonas/Datasets/TikZ/benchmark_data")
df = pd.read_parquet(base / "datikz-benchmark.parquet")

image_dir = base / "images"
code_dir = base / "references"
caption_dir = base / "captions"

for directory in [image_dir, code_dir, caption_dir]:
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
        "reference_image": f"/images/{image_name}",
        "reference_code": f"/references/{code_name}",
        "caption": f"file:///captions/{caption_name}",
        "input_image_format_for_vlm": f"file:///images/{image_name}"

    })

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(base / "image_manifest.csv", index=False)

print("Finished!!!")