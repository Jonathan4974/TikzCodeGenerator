from pathlib import Path
import re
import shutil

import pandas as pd


# Configuration
DATA_DIR = Path("/root/projects/training/gemma4_finetuning_grpo_fast/data")
MANIFESTS = ["manifest_train.csv", "manifest_val.csv"]
REVIEW_DIR = DATA_DIR / "images_with_plots"
MAX_NUMERIC_COORDINATES = 5


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
COORDINATE_RE = re.compile(rf"\(\s*{NUMBER}\s*,\s*{NUMBER}\s*\)")
PLOT_RE = re.compile(
    r"coordinates\s*\{|\\addplot3?\*?.*?\b(?:table|file)\b|"
    r"\\pgfplotstableread|\\begin\{filecontents\*?\}",
    re.IGNORECASE | re.DOTALL,
)
AXIS_RE = re.compile(r"\\begin\{axis\*?\}", re.IGNORECASE)


def resolve(path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else DATA_DIR / path


def has_plot_data(code: str) -> tuple[bool, str, int]:
    code = re.sub(r"(?<!\\)%.*", "", code)
    coordinate_count = len(COORDINATE_RE.findall(code))

    reasons = []
    if PLOT_RE.search(code):
        reasons.append("plot_data")
    if AXIS_RE.search(code) and coordinate_count:
        reasons.append("axis_with_coordinates")
    if coordinate_count > MAX_NUMERIC_COORDINATES:
        reasons.append("many_coordinates")

    return bool(reasons), ";".join(reasons), coordinate_count


def copy_removed(row: pd.Series) -> None:
    for column in ("image_path", "code_path", "vlm_description_path"):
        if column not in row or pd.isna(row[column]):
            continue

        source = resolve(str(row[column]))
        if not source.is_file():
            raise FileNotFoundError(source)

        relative = Path(str(row[column]))
        if relative.is_absolute():
            relative = Path(column) / source.name

        target = REVIEW_DIR / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def filter_manifest(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    df = pd.read_csv(path)
    keep = []
    removed = []

    for index, row in df.iterrows():
        code_path = resolve(str(row["code_path"]))
        code = code_path.read_text(encoding="utf-8", errors="replace")
        remove, reason, count = has_plot_data(code)

        if remove:
            copy_removed(row)
            removed.append({
                **row.to_dict(),
                "source_manifest": name,
                "source_row": index,
                "filter_reason": reason,
                "numeric_coordinate_count": count,
            })
        else:
            keep.append(row)

    output = path.with_name(f"{path.stem}_without_plot.csv")
    pd.DataFrame(keep, columns=df.columns).to_csv(output, index=False)

    print(f"{name}: kept={len(keep):,}, removed={len(removed):,}")
    print(f"Saved: {output}")
    return pd.DataFrame(removed)


def main() -> None:
    if REVIEW_DIR.exists():
        shutil.rmtree(REVIEW_DIR)
    REVIEW_DIR.mkdir(parents=True)

    removed = pd.concat(
        [filter_manifest(name) for name in MANIFESTS],
        ignore_index=True,
    )
    removed.to_csv(REVIEW_DIR / "removed_examples.csv", index=False)


if __name__ == "__main__":
    main()

