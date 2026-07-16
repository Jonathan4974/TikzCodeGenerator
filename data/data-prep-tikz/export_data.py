import glob
import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


INPUT_DIR = Path("./data/clean_parquets/our_dataset_benchmark")
OUTPUT_DIR = Path("./data/exported_dataset/our_dataset_benchmark_merged")

MODES = {
    "simple_vlm_description": (
        "image_with_text",
        "code_with_text",
        "llm_description_with_text",
    ),
    "full_cleaning": (
        "image_without_text_full",
        "code_without_text_full",
        "llm_description_without_text_full",
    ),
    "deterministic_cleaning": (
        "image_without_text_deterministic",
        "code_without_text_deterministic",
        "llm_description_without_text_deterministic",
    ),
}


def key(code):
    return hashlib.sha256(str(code).encode()).hexdigest()


def load_image(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return Image.open(BytesIO(value)).convert("RGB")
    if value.get("bytes") is not None:
        return Image.open(BytesIO(value["bytes"])).convert("RGB")
    return Image.open(value["path"]).convert("RGB")


def load_mode(mode, cols):
    image_col, code_col, description_col = cols

    files = sorted(
        glob.glob(str(INPUT_DIR / f"*-{mode}_part-*.parquet"))
    )
    if not files:
        raise FileNotFoundError(f"Keine Dateien für {mode}")

    columns = list(dict.fromkeys([
        "code_with_text",
        image_col,
        code_col,
        description_col,
    ]))

    df = pd.concat(
        [pd.read_parquet(file, columns=columns) for file in files],
        ignore_index=True,
    )

    # Gemeinsamer CrystalBLEU-Korpus
    corpus_dir = OUTPUT_DIR / "crystalbleu_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    missing = (
        df[description_col].isna()
        | df[description_col].astype("string").str.strip().eq("")
    )

    for code in df.loc[missing, "code_with_text"].dropna():
        if str(code).strip():
            (corpus_dir / f"{key(code)}.txt").write_text(
                str(code),
                encoding="utf-8",
            )

    # Nur vollständige Samples
    complete = (
        df[image_col].notna()
        & df[code_col].notna()
        & df[description_col].notna()
        & df[description_col].astype("string").str.strip().ne("")
    )

    df = df.loc[complete].copy()
    df["sample_key"] = df["code_with_text"].map(key)

    return df.drop_duplicates("sample_key").set_index("sample_key")


data = {
    mode: load_mode(mode, cols)
    for mode, cols in MODES.items()
}

# Schnittmenge aller Modi
common = set.intersection(*(set(df.index) for df in data.values()))
first_mode = next(iter(MODES))
common = [key for key in data[first_mode].index if key in common]

manifests = {mode: [] for mode in MODES}

for mode in MODES:
    for folder in ("images", "references", "descriptions"):
        (OUTPUT_DIR / mode / folder).mkdir(parents=True, exist_ok=True)

for number, sample_key in enumerate(tqdm(common)):
    sample_id = f"{number:08d}"

    for mode, (image_col, code_col, description_col) in MODES.items():
        row = data[mode].loc[sample_key]
        mode_dir = OUTPUT_DIR / mode

        image_path = Path("images") / f"{sample_id}.png"
        code_path = Path("references") / f"{sample_id}.tex"
        description_path = Path("descriptions") / f"{sample_id}.txt"

        load_image(row[image_col]).save(mode_dir / image_path)

        (mode_dir / code_path).write_text(
            str(row[code_col]),
            encoding="utf-8",
        )

        (mode_dir / description_path).write_text(
            str(row[description_col]).strip(),
            encoding="utf-8",
        )

        manifests[mode].append({
            "id": sample_id,
            "sample_key": sample_key,
            "image_path": (mode_dir / image_path).resolve().as_posix(),
            "code_path": (mode_dir / code_path).resolve().as_posix(),
            "description_path": (
                mode_dir / description_path
            ).resolve().as_posix(),
        })

for mode, manifest in manifests.items():
    pd.DataFrame(manifest).to_csv(
        OUTPUT_DIR / mode / "manifest.csv",
        index=False,
    )

print(f"Exportiert: {len(common):,} gemeinsame Samples")
print(f"Ausgabe: {OUTPUT_DIR}")