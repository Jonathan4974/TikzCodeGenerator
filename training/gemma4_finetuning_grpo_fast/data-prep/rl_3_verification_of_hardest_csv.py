from pathlib import Path

import pandas as pd


# Konfiguration
BASE_PATH = Path("/root/path/to/normal/train")
INPUT_MANIFEST = Path("manifest-worst-loss.csv")
OUTPUT_MANIFEST = Path("manifest-worst-loss-verified.csv")

PATH_COLUMNS = {
    "image_path": "images",
    "input_image_path": "images",
    "code_path": "references",
    "vlm_description_path": "descriptions",
}


df = pd.read_csv(INPUT_MANIFEST)

missing_columns = [column for column in PATH_COLUMNS if column not in df.columns]
if missing_columns:
    raise ValueError(f"Fehlende Spalten im Manifest: {missing_columns}")

valid_rows = pd.Series(True, index=df.index)
missing_files = []

for column, subfolder in PATH_COLUMNS.items():
    new_paths = df[column].map(
        lambda old_path: BASE_PATH / subfolder / Path(str(old_path)).name
    )

    exists = new_paths.map(Path.is_file)
    valid_rows &= exists

    for row_index, path in new_paths[~exists].items():
        missing_files.append((row_index, column, path))

    df[column] = new_paths.map(str)

verified = df.loc[valid_rows].reset_index(drop=True)
verified.to_csv(OUTPUT_MANIFEST, index=False)

print(f"Zeilen im Original: {len(df)}")
print(f"Vollständig vorhanden: {len(verified)}")
print(f"Entfernte Zeilen: {len(df) - len(verified)}")
print(f"Fehlende Dateien: {len(missing_files)}")
print(f"Gespeichert: {OUTPUT_MANIFEST.resolve()}")

if missing_files:
    print("\nErste fehlende Dateien:")
    for row_index, column, path in missing_files[:20]:
        print(f"Zeile {row_index}: {column} -> {path}")