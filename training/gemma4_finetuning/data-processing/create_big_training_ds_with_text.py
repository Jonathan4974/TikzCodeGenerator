import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
from pathlib import Path
import csv
import gc
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tqdm.auto import tqdm
import hashlib
from PIL import Image

from pf_utils.tikz_rendering import render_tex_to_png


# ============================================================
# RENDER CONFIG
# ============================================================

os.environ["LATEX_DPI"] = "600"
os.environ["REF_IMAGE_SIZE"] = "512"

# ============================================================
# CONFIG
# ============================================================

p_val = 1
p_code_corpus = 5
seed = 42

max_reference_tokens = 8192
batch_size = 512
num_render_workers = 12
render_chunksize = 2

tokenizer_name = "/models/huggingface/hub/gemma-4-31B-it-unsloth-bnb-4bit"
tokenizer = Tokenizer.from_file(str(Path(tokenizer_name) / "tokenizer.json"))

datasets = [
    {
        "source": "datikz_v4",
        "path": Path("/parquet_files/DaTikZ-v4"),
        "description_col": "vlm_description",
        "code_col": "tikz_code",
        "image_col": "png_image",  # wird nicht mehr verwendet
        "sample_percent": 0,
    },
    {
        "source": "geotikz_bridge_base",
        "path": Path("/parquet_files/geotikz_bridge_base"),
        "description_col": None,
        "code_col": "response",
        "image_col": "image",  # wird nicht mehr verwendet
        "sample_percent": 0,
    },
    {
        "source": "our_dataset",
        "path": Path("/parquet_files/our_dataset_train"),
        "description_col": None,
        "code_col": "code",
        "image_col": "image",  # wird nicht mehr verwendet
        "sample_percent": 1,
    },
]

output_dir = Path("/data")

# Das ist der Pfad, der später in den Manifest-Dateien stehen soll.
# Falls dein Training im Docker unter /data läuft, passt das so.
manifest_root = Path("/data")

image_folder = output_dir / "images"
code_folder = output_dir / "references"
description_folder = output_dir / "vlm_descriptions"
code_corpus_folder = output_dir / "code_corpus"
render_tmp_folder = output_dir / "_render_tmp"

train_manifest_raw_path = output_dir / "manifest_train_raw.csv"
val_manifest_raw_path = output_dir / "manifest_val_raw.csv"

train_manifest_path = output_dir / "manifest_train.csv"
val_manifest_path = output_dir / "manifest_val.csv"

delete_failed_files = True


# ============================================================
# SETUP
# ============================================================

for folder in [
    image_folder,
    code_folder,
    description_folder,
    code_corpus_folder,
    render_tmp_folder,
]:
    folder.mkdir(parents=True, exist_ok=True)

if p_val + p_code_corpus >= 100:
    raise ValueError("p_val + p_code_corpus muss kleiner als 100 sein.")

print(f"Output folder: {output_dir}")
print(f"Batch size: {batch_size}")
print(f"Max reference tokens: {max_reference_tokens}")
print(f"Validation split: {p_val}%")
print(f"Code corpus split: {p_code_corpus}%")
print("Images are NOT copied from parquet.")
print("Images will be rendered in third pass from TikZ code.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value)


def clean_code(text: str) -> str:
    text = normalize_text(text).strip()

    # Markdown-Fence extrahieren, z. B. ```tikz ... ```
    fence_match = re.search(
        r"```(?:latex|tex|tikz)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        text = fence_match.group(1).strip()

    text = (
        text.replace("```latex", "")
        .replace("```tex", "")
        .replace("```tikz", "")
        .replace("```", "")
        .strip()
    )

    # Text vor \documentclass entfernen
    start = r"\documentclass"
    if start in text:
        text = text[text.index(start):]

    # Nach erstem \end{document} abschneiden
    end = r"\end{document}"
    if end in text:
        text = text[: text.index(end) + len(end)]

    return text.strip()


def is_empty_code(text: str) -> bool:
    text = text.strip()
    return not text or text.lower() == "nan"


def token_count(text: str) -> int:
    return len(tokenizer.encode(str(text), add_special_tokens=False).ids)


def keep_by_sample_percent(source: str, source_idx: int, sample_percent: float, seed: int) -> bool:
    if sample_percent >= 100:
        return True

    if sample_percent <= 0:
        return False

    key = f"{seed}:{source}:{source_idx}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    random_value = value / 2**64

    return random_value < (sample_percent / 100)


def manifest_path(folder_name: str, filename: str) -> str:
    return str(manifest_root / folder_name / filename)


def resolve_manifest_path(path_value: str) -> Path:
    """
    Wandelt Manifest-Pfade wie /data/references/x.txt zurück
    auf echte lokale Pfade unter output_dir.
    """
    p = Path(str(path_value))

    if p.is_absolute() and str(p).startswith(str(manifest_root) + "/"):
        return output_dir / p.relative_to(manifest_root)

    if p.is_absolute():
        return p

    return output_dir / p


def is_mostly_white(path: Path, threshold=250, max_nonwhite_ratio=0.002):
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    nonwhite_ratio = float(np.mean(arr < threshold))
    return nonwhite_ratio < max_nonwhite_ratio, nonwhite_ratio


def safe_unlink(path: Path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def delete_files_for_row(row: dict):
    safe_unlink(resolve_manifest_path(row["image_path"]))
    safe_unlink(resolve_manifest_path(row["code_path"]))
    safe_unlink(resolve_manifest_path(row["vlm_description_path"]))


# ============================================================
# FIRST PASS: COUNT VALID ITEMS ONLY
# ============================================================

valid_count = 0
total_seen = 0
too_long = 0
empty_code = 0
sampled_out = 0

print("\nFirst pass: sampling and filtering by token length...")

for dataset in datasets:
    source = dataset["source"]
    sample_percent = dataset.get("sample_percent", 100)

    if not (0 <= sample_percent <= 100):
        raise ValueError(f"sample_percent für {source} muss zwischen 0 und 100 liegen.")

    source_idx = 0
    parquet_files = sorted(dataset["path"].glob("*.parquet"))

    print(f"\nSource: {source}")
    print(f"Parquet files: {len(parquet_files)}")

    for parquet_file in tqdm(parquet_files, desc=f"Filtering {source}"):
        pf = pq.ParquetFile(parquet_file)
        file_rows = pf.metadata.num_rows

        with tqdm(total=file_rows, desc=parquet_file.name, leave=False) as pbar:
            for batch in pf.iter_batches(
                batch_size=batch_size,
                columns=[dataset["code_col"]],
            ):
                data = batch.to_pydict()
                codes = data[dataset["code_col"]]

                for code_value in codes:
                    row_idx = source_idx
                    source_idx += 1

                    code_text = clean_code(code_value)
                    total_seen += 1

                    if not keep_by_sample_percent(source, row_idx, sample_percent, seed):
                        sampled_out += 1
                        continue

                    if is_empty_code(code_text):
                        empty_code += 1
                    elif token_count(code_text) <= max_reference_tokens:
                        valid_count += 1
                    else:
                        too_long += 1

                pbar.update(len(codes))

        gc.collect()

print("\nFiltering summary:")
print(f"Total rows seen: {total_seen}")
print(f"Sampled out: {sampled_out}")
print(f"Rows after sampling: {total_seen - sampled_out}")
print(f"Valid rows: {valid_count}")
print(f"Filtered too long: {too_long}")
print(f"Filtered empty code: {empty_code}")


# ============================================================
# SPLIT VALID ITEMS INTO TRAIN / VAL / CODE CORPUS
# ============================================================

print("\nSplitting valid items...")

rng = np.random.default_rng(seed)

n_code_corpus = round(valid_count * p_code_corpus / 100)
n_val = round(valid_count * p_val / 100)
n_train = valid_count - n_code_corpus - n_val

# 0 = train, 1 = val, 2 = code_corpus
split_labels = np.zeros(valid_count, dtype=np.uint8)

indices = rng.permutation(valid_count)

split_labels[indices[:n_code_corpus]] = 2
split_labels[indices[n_code_corpus:n_code_corpus + n_val]] = 1

del indices
gc.collect()

print(f"Target train rows: {n_train}")
print(f"Target val rows: {n_val}")
print(f"Target code corpus rows: {n_code_corpus}")


# ============================================================
# SECOND PASS: WRITE CODE, DESCRIPTIONS AND RAW MANIFESTS
# ============================================================

manifest_columns = [
    "code_path",
    "image_path",
    "vlm_description_path",
]

written_train_raw = 0
written_val_raw = 0
written_code_corpus = 0
valid_idx = 0

print("\nSecond pass: writing code, descriptions and raw manifests...")
print("No images are written in second pass.")

with train_manifest_raw_path.open("w", newline="", encoding="utf-8") as train_file, \
     val_manifest_raw_path.open("w", newline="", encoding="utf-8") as val_file:

    train_writer = csv.DictWriter(train_file, fieldnames=manifest_columns)
    val_writer = csv.DictWriter(val_file, fieldnames=manifest_columns)

    train_writer.writeheader()
    val_writer.writeheader()

    for dataset in datasets:
        source = dataset["source"]
        sample_percent = dataset.get("sample_percent", 100)
        source_idx = 0
        parquet_files = sorted(dataset["path"].glob("*.parquet"))

        print(f"\nSource: {source}")

        for parquet_file in tqdm(parquet_files, desc=f"Writing {source}"):
            pf = pq.ParquetFile(parquet_file)
            file_rows = pf.metadata.num_rows

            columns = [dataset["code_col"]]

            if dataset["description_col"] is not None:
                columns.append(dataset["description_col"])

            with tqdm(total=file_rows, desc=parquet_file.name, leave=False) as pbar:
                for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
                    data = batch.to_pydict()

                    codes = data[dataset["code_col"]]

                    if dataset["description_col"] is None:
                        descriptions = [""] * len(codes)
                    else:
                        descriptions = data[dataset["description_col"]]

                    for description_value, code_value in zip(descriptions, codes):
                        row_idx = source_idx
                        stem = f"{source}_{source_idx:08d}"
                        source_idx += 1

                        if not keep_by_sample_percent(source, row_idx, sample_percent, seed):
                            continue

                        code_text = clean_code(code_value)

                        if is_empty_code(code_text):
                            continue

                        if token_count(code_text) > max_reference_tokens:
                            continue

                        split = split_labels[valid_idx]
                        valid_idx += 1

                        if split == 2:
                            code_corpus_path = code_corpus_folder / f"{stem}.txt"
                            code_corpus_path.write_text(code_text, encoding="utf-8")
                            written_code_corpus += 1
                            continue

                        generated_image = image_folder / f"{stem}.png"
                        code_path = code_folder / f"{stem}.txt"
                        description_path = description_folder / f"{stem}.txt"

                        code_path.write_text(code_text, encoding="utf-8")
                        description_path.write_text(
                            normalize_text(description_value),
                            encoding="utf-8",
                        )

                        manifest_row = {
                            "code_path": manifest_path("references", code_path.name),
                            "image_path": manifest_path("images", generated_image.name),
                            "vlm_description_path": manifest_path(
                                "vlm_descriptions",
                                description_path.name,
                            ),
                        }

                        if split == 1:
                            val_writer.writerow(manifest_row)
                            written_val_raw += 1
                        else:
                            train_writer.writerow(manifest_row)
                            written_train_raw += 1

                    pbar.update(batch.num_rows)

            gc.collect()


# ============================================================
# CHECKS AFTER SECOND PASS
# ============================================================

if valid_idx != valid_count:
    raise RuntimeError(
        f"Valid count mismatch: first pass found {valid_count}, "
        f"second pass wrote/processed {valid_idx}"
    )


# ============================================================
# THIRD PASS: PARALLEL RENDER IMAGES FROM RAW MANIFESTS
# ============================================================

def render_one_row(args):
    row, split_name = args

    code_path = resolve_manifest_path(row["code_path"])
    image_path = resolve_manifest_path(row["image_path"])

    image_path.parent.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    tmp_dir = render_tmp_folder / split_name / str(pid)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_image_path = tmp_dir / image_path.name

    safe_unlink(tmp_image_path)

    try:
        tex_code = clean_code(code_path.read_text(encoding="utf-8"))

        metrics = {}
        render_tex_to_png(
            tex_code=tex_code,
            output_path=tmp_image_path,
            metrics=metrics,
            create_ds=True,
        )

        is_blank, nonwhite_ratio = is_mostly_white(tmp_image_path)

        if is_blank:
            safe_unlink(tmp_image_path)

            if delete_failed_files:
                safe_unlink(code_path)
                safe_unlink(resolve_manifest_path(row["vlm_description_path"]))

            return {
                "status": "blank",
                "row": None,
                "tmp_image_path": "",
                "image_path": "",
                "error": "",
            }

        return {
            "status": "ok",
            "row": row,
            "tmp_image_path": str(tmp_image_path),
            "image_path": str(image_path),
            "error": "",
        }

    except Exception as e:
        safe_unlink(tmp_image_path)

        if delete_failed_files:
            safe_unlink(code_path)
            safe_unlink(resolve_manifest_path(row["vlm_description_path"]))

        return {
            "status": "failed",
            "row": None,
            "tmp_image_path": "",
            "image_path": "",
            "error": str(e),
        }


def render_manifest_parallel(
    raw_manifest_path: Path,
    final_manifest_path: Path,
    split_name: str,
    num_workers: int,
):
    rendered = 0
    render_failed = 0
    blank = 0

    print(f"\nThird pass: rendering {split_name} images in parallel...")
    print(f"Raw manifest:   {raw_manifest_path}")
    print(f"Final manifest: {final_manifest_path}")
    print(f"Workers:        {num_workers}")

    with raw_manifest_path.open("r", newline="", encoding="utf-8") as raw_file:
        reader = csv.DictReader(raw_file)
        rows = list(reader)

    total = len(rows)
    tasks = [(row, split_name) for row in rows]

    with final_manifest_path.open("w", newline="", encoding="utf-8") as final_file:
        writer = csv.DictWriter(final_file, fieldnames=manifest_columns)
        writer.writeheader()
        final_file.flush()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(render_one_row, task)
                for task in tasks
            ]

            for future in tqdm(as_completed(futures), total=total, desc=f"Rendering {split_name}"):
                result = future.result()
                status = result["status"]

                if status == "ok":
                    tmp_image_path = Path(result["tmp_image_path"])
                    image_path = Path(result["image_path"])

                    safe_unlink(image_path)
                    tmp_image_path.replace(image_path)

                    writer.writerow(result["row"])
                    rendered += 1

                    # wichtig: Manifest regelmäßig sichtbar machen
                    if rendered % 100 == 0:
                        final_file.flush()
                        os.fsync(final_file.fileno())

                elif status == "blank":
                    blank += 1

                else:
                    render_failed += 1

        final_file.flush()
        os.fsync(final_file.fileno())

    print(f"\n{split_name} render summary:")
    print(f"Total raw rows:  {total}")
    print(f"Rendered kept:   {rendered}")
    print(f"Blank removed:   {blank}")
    print(f"Render failed:   {render_failed}")
    print(f"Final manifest:  {final_manifest_path}")

    return rendered, blank, render_failed, total


rendered_train, blank_train, failed_train, raw_train_total = render_manifest_parallel(
    raw_manifest_path=train_manifest_raw_path,
    final_manifest_path=train_manifest_path,
    split_name="train",
    num_workers=num_render_workers,
)

rendered_val, blank_val, failed_val, raw_val_total = render_manifest_parallel(
    raw_manifest_path=val_manifest_raw_path,
    final_manifest_path=val_manifest_path,
    split_name="val",
    num_workers=num_render_workers,
)

# ============================================================
# SUMMARY
# ============================================================

print("\nDone.")
print(f"Total rows seen: {total_seen}")
print(f"Valid rows before rendering: {valid_count}")
print(f"Filtered too long: {too_long}")
print(f"Filtered empty code: {empty_code}")
print(f"Sampled out: {sampled_out}")

print()
print(f"Raw train rows: {written_train_raw}")
print(f"Raw val rows: {written_val_raw}")
print(f"Code corpus rows: {written_code_corpus}")

print()
print(f"Final train rows: {rendered_train}")
print(f"Final val rows: {rendered_val}")

print()
print(f"Train blank removed: {blank_train}")
print(f"Train render failed: {failed_train}")
print(f"Val blank removed: {blank_val}")
print(f"Val render failed: {failed_val}")

print()
print(f"Raw train manifest: {train_manifest_raw_path}")
print(f"Raw val manifest: {val_manifest_raw_path}")
print(f"Final train manifest: {train_manifest_path}")
print(f"Final val manifest: {val_manifest_path}")
print(f"Code corpus folder: {code_corpus_folder}")
print(f"Output folder: {output_dir}")