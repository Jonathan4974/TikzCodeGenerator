"""Manual script: streams a few rows of loss-boss/tikz-train's `train` split and prints
their real columns.

Usage:
    python -m training.sketch_agent.check_dataset_schema
    python -m training.sketch_agent.check_dataset_schema --num-rows 5
"""
from __future__ import annotations

import argparse

from .dataset_loader import HF_REPO_ID, SPLIT_NAME, load_train_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rows", type=int, default=1)
    parser.add_argument("--no-streaming", action="store_true")
    args = parser.parse_args()

    streaming = not args.no_streaming
    print(f"Checking {HF_REPO_ID} - split {SPLIT_NAME!r}, streaming={streaming}")

    ds = load_train_split(streaming=streaming)
    rows = iter(ds) if streaming else (ds[i] for i in range(args.num_rows))

    for i, row in zip(range(args.num_rows), rows):
        print(f"\n=== row {i} ===")
        print(f"  columns: {sorted(row.keys())}")
        for key, value in row.items():
            value_type = type(value).__name__
            shape_or_len = getattr(value, "size", None) or (
                len(value) if isinstance(value, str) else None
            )
            print(f"    - {key}: {value_type}" + (f" ({shape_or_len})" if shape_or_len else ""))


if __name__ == "__main__":
    main()
