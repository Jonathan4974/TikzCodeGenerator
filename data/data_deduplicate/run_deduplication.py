#!/usr/bin/env python3
"""
Run deduplication between two HF datasets:
- nllg/datikz-v3 (145k)
- SJY-1995/GeoTikz-Base (2.45m)
"""

from dataset_deduplicator import DatasetDeduplicator

if __name__ == "__main__":
    dedup = DatasetDeduplicator(
        dataset1_name="nllg/datikz-v3",
        dataset2_name="SJY-1995/GeoTikz-Base",
        image_column="image",   # adjust if needed
        split="train",
        threshold=2,
        cache_dir="./hash_cache",
        num_proc=4          # adjust based on your CPU cores
    )

    # For quick test, set max_samples; remove to process full datasets
    # dedup.run(max_samples1=1000, max_samples2=1000, output_prefix="test")
    dedup.run(output_prefix="dedup_results")