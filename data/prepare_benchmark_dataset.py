import pandas as pd
from pathlib import Path

base = Path("/usr/prakt/s0030/projects/data/benchmark_data")
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
        "reference_image": f"{base}/images/{image_name}",
        "reference_code": f"{base}/references/{code_name}",
        "caption": f"file:///{base}/captions/{caption_name}",
        "input_image_format_for_vlm": f"file:///{base}/images/{image_name}"
    })

manifest = pd.DataFrame(manifest_rows)

manifest.to_csv(base / "image_manifest.csv", index=False)

n = 100

if n > len(manifest):
    raise ValueError(f"n={n} ist größer als die Anzahl der verfügbaren Zeilen: {len(manifest)}")

small_manifest = manifest.sample(
    n=n,
    replace=False,      # keine Dopplungen
    random_state=None   # jedes Mal andere zufällige Auswahl
)

small_manifest.to_csv(base / "small_manifest.csv", index=False)

print("Finished!!!")
print(f"Full manifest: {len(manifest)} rows")
print(f"Small manifest: {len(small_manifest)} rows")