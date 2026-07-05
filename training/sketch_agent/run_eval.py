from __future__ import annotations

import argparse
import json
from pathlib import Path

from .eval import evaluate_generated_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sketch-agent predictions")
    parser.add_argument("--output-dir", default="training/sketch_agent/output_smoke/predictions")
    parser.add_argument("--reference-dir", default="training/sketch_agent/output_smoke/synthetic_pairs/targets")
    parser.add_argument("--metrics", default="pixel_cc", help="Comma-separated metrics: pixel_cc,siglip,dreamsim")
    args = parser.parse_args()

    metrics = tuple(args.metrics.split(","))
    results = evaluate_generated_outputs(args.output_dir, args.reference_dir, metrics=metrics)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
