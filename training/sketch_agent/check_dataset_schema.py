"""Manual script: streams one row per split of loss-boss/tikz-dataset and
prints its real columns.

Usage:
    python -m training.sketch_agent.check_dataset_schema
    python -m training.sketch_agent.check_dataset_schema --splits datikz_v4 our_dataset_benchmark
"""
from __future__ import annotations

import argparse

from .dataset_loader import HF_REPO_ID, SPLITS


def _try_plain_load(split_name: str, streaming: bool):
    from datasets import load_dataset

    return load_dataset(HF_REPO_ID, split_name, streaming=streaming)


def _try_parquet_glob_load(split_name: str, streaming: bool):
    from datasets import load_dataset

    pattern = f"hf://datasets/{HF_REPO_ID}/{split_name}_part-*.parquet"
    return load_dataset(
        "parquet", data_files={split_name: pattern}, split=split_name, streaming=streaming
    )


def check_split(split_name: str, streaming: bool = True) -> None:
    print(f"\n=== split: {split_name} ===")

    ds = None
    load_path = None
    for label, loader in (("plain load_dataset(repo, split)", _try_plain_load),
                           ("hf://...parquet glob (SplitLoader-style)", _try_parquet_glob_load)):
        try:
            ds = loader(split_name, streaming)
            load_path = label
            break
        except Exception as exc:
            print(f"  [{label}] failed: {type(exc).__name__}: {exc}")

    if ds is None:
        print(f"  COULD NOT LOAD split '{split_name}' via either shape.")
        return

    print(f"  loaded via: {load_path}")

    try:
        row = next(iter(ds)) if streaming else ds[0]
    except Exception as exc:
        print(f"  COULD NOT PEEK a row: {type(exc).__name__}: {exc}")
        return

    print(f"  columns: {sorted(row.keys())}")
    for key, value in row.items():
        value_type = type(value).__name__
        shape_or_len = getattr(value, "size", None) or (
            len(value) if isinstance(value, str) else None
        )
        print(f"    - {key}: {value_type}" + (f" ({shape_or_len})" if shape_or_len else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=SPLITS)
    parser.add_argument("--no-streaming", action="store_true")
    args = parser.parse_args()

    print(f"Checking {HF_REPO_ID} - {len(args.splits)} split(s), streaming={not args.no_streaming}")
    for split_name in args.splits:
        check_split(split_name, streaming=not args.no_streaming)


if __name__ == "__main__":
    main()
