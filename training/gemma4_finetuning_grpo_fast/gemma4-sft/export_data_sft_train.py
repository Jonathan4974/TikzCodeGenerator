from io import BytesIO
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm


# =========================
# CONFIG
# =========================
REPO_ID = "loss-boss/tikz-train"
OUTPUT_DIR = Path("../data")

TRAIN_PERCENT = 90
VAL_PERCENT = 10
CRYSTALBLEU_SIZE = 50_000
RANDOM_SEED = 42

SCHEMAS = (
    {
        "image_with_text": "image",
        "code_with_text": "code",
        "llm_description_with_text": "description",
    },
    {
        "image_without_text_full": "image",
        "code_without_text_full": "code",
        "llm_description_without_text_full": "description",
    },
)


def load_data() -> pd.DataFrame:
    dataset_dir = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns="*.parquet",
        )
    )

    parquet_files = sorted(dataset_dir.rglob("*.parquet"))
    parts = []

    for file in tqdm(parquet_files, desc="Loading parquet files"):
        columns = set(pq.read_schema(file).names)
        loaded_rows = 0
        details = []

        for mapping in SCHEMAS:
            if not mapping.keys() <= columns:
                continue

            part = pd.read_parquet(file, columns=list(mapping))
            total_rows = len(part)

            part = part.rename(columns=mapping)
            part = part.dropna(subset=["image", "code"])
            part["description"] = part["description"].fillna("")

            details.append(
                f"{next(iter(mapping))}: {len(part):,}/{total_rows:,}"
            )

            if not part.empty:
                parts.append(part)
                loaded_rows += len(part)

        if loaded_rows:
            tqdm.write(
                f"Loaded: {file.name} "
                f"({loaded_rows:,} usable rows; {', '.join(details)})"
            )
        else:
            tqdm.write(
                f"Skipped: {file.name} "
                "(no rows containing both image and code)"
            )

    if not parts:
        raise RuntimeError("No dataset rows containing both image and code found.")

    return pd.concat(parts, ignore_index=True)


def create_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if TRAIN_PERCENT + VAL_PERCENT != 100:
        raise ValueError("TRAIN_PERCENT + VAL_PERCENT must equal 100.")

    has_no_description = df["description"].astype(str).str.strip().eq("")
    corpus_candidates = df[has_no_description]

    if CRYSTALBLEU_SIZE > len(corpus_candidates):
        raise ValueError(
            f"CRYSTALBLEU_SIZE is {CRYSTALBLEU_SIZE:,}, but only "
            f"{len(corpus_candidates):,} rows without descriptions are available."
        )

    corpus = corpus_candidates.sample(
        n=CRYSTALBLEU_SIZE,
        random_state=RANDOM_SEED,
    )

    remaining = df.drop(index=corpus.index).sample(
        frac=1,
        random_state=RANDOM_SEED,
    )

    train_end = int(len(remaining) * TRAIN_PERCENT / 100)

    train = remaining.iloc[:train_end].reset_index(drop=True)
    val = remaining.iloc[train_end:].reset_index(drop=True)
    corpus = corpus.reset_index(drop=True)

    return train, val, corpus


def save_image(value, path: Path) -> None:
    if isinstance(value, dict):
        value = (
            BytesIO(value["bytes"])
            if value.get("bytes")
            else value["path"]
        )
    elif isinstance(value, bytes):
        value = BytesIO(value)

    image = value if isinstance(value, Image.Image) else Image.open(value)

    try:
        image.convert("RGB").save(path, "PNG")
    finally:
        if image is not value:
            image.close()


def export_split(df: pd.DataFrame, split: str) -> None:
    for folder in ("images", "references", "descriptions"):
        (OUTPUT_DIR / split / folder).mkdir(parents=True, exist_ok=True)

    manifest = []

    rows = df.itertuples(index=False)

    for index, row in enumerate(
        tqdm(
            rows,
            total=len(df),
            desc=f"Exporting {split}",
            unit="sample",
        )
    ):
        name = f"{index:08d}"

        image_path = Path(split, "images", f"{name}.png")
        code_path = Path(split, "references", f"{name}.tex")
        description_path = Path(split, "descriptions", f"{name}.txt")

        save_image(row.image, OUTPUT_DIR / image_path)

        (OUTPUT_DIR / code_path).write_text(
            str(row.code),
            encoding="utf-8",
        )

        (OUTPUT_DIR / description_path).write_text(
            str(row.description),
            encoding="utf-8",
        )

        manifest.append(
            {
                "image_path": image_path.as_posix(),
                "code_path": code_path.as_posix(),
                "vlm_description_path": description_path.as_posix(),
            }
        )

    pd.DataFrame(manifest).to_csv(
        OUTPUT_DIR / f"manifest_{split}.csv",
        index=False,
    )


def export_corpus(df: pd.DataFrame) -> None:
    corpus_dir = OUTPUT_DIR / "crystalbleu-corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    for index, code in enumerate(
        tqdm(
            df["code"],
            total=len(df),
            desc="Exporting CrystalBLEU corpus",
            unit="file",
        )
    ):
        (corpus_dir / f"{index:08d}.txt").write_text(
            str(code),
            encoding="utf-8",
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_data()
    train, val, corpus = create_splits(dataframe)

    print(
        f"Train: {len(train):,} | "
        f"Validation: {len(val):,} | "
        f"Corpus: {len(corpus):,}"
    )

    export_split(train, "train")
    export_split(val, "val")
    export_corpus(corpus)

    print(f"Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()