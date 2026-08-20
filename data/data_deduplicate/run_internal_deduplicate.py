#!/usr/bin/env python3
"""
Internal deduplication for a single Hugging Face image dataset.
Uses cached perceptual hashes if available, otherwise computes them.
Outputs two CSV files: kept_indices.csv (unique images) and duplicated_indices.csv (to remove).
"""

import os
import csv
import pickle
from typing import Dict, List, Optional
from tqdm import tqdm
from datasets import load_dataset
from perceptual_hash import PerceptualHash
from dataset_deduplicator import DatasetDeduplicator  # reuse hash computation


class InternalDeduplicator:
    def __init__(
        self,
        dataset_name: str,
        image_column: str = 'image',
        split: str = 'train',
        threshold: int = 2,
        cache_dir: str = './hash_cache',
        num_proc: int = 4
    ):
        self.dataset_name = dataset_name
        self.image_column = image_column
        self.split = split
        self.threshold = threshold
        self.cache_dir = cache_dir
        self.num_proc = num_proc
        os.makedirs(cache_dir, exist_ok=True)

    def _load_dataset(self):
        print(f"Loading dataset: {self.dataset_name}")
        return load_dataset(self.dataset_name, split=self.split)

    def _load_or_compute_hashes(self, dataset, max_samples: Optional[int] = None) -> Dict[int, int]:
        """Reuse the caching logic from DatasetDeduplicator (parallel map)."""
        # Create a temporary deduplicator just to use its hash computation method
        temp = DatasetDeduplicator(
            dataset1_name=self.dataset_name,
            dataset2_name="dummy",
            image_column=self.image_column,
            split=self.split,
            threshold=self.threshold,
            cache_dir=self.cache_dir,
            num_proc=self.num_proc
        )
        # We'll call _compute_hashes, but it expects a dataset and a name
        return temp._compute_hashes(dataset, self.dataset_name, max_samples)

    def find_internal_duplicates(self, hashes: Dict[int, int]) -> (List[int], List[int]):
        """
        Find duplicate indices within the same dataset.
        Returns: (kept_indices, duplicated_indices)
        Keeps the first occurrence of each duplicate group.
        """
        # Generate all masks for Hamming distance <= threshold
        def gen_masks(threshold: int, bits: int = 64):
            masks = [0]
            if threshold >= 1:
                for i in range(bits):
                    masks.append(1 << i)
            if threshold >= 2:
                for i in range(bits):
                    for j in range(i+1, bits):
                        masks.append((1 << i) | (1 << j))
            return masks

        masks = gen_masks(self.threshold)
        print(f"Generated {len(masks)} variant masks for threshold={self.threshold}")

        kept_hashes = set()          # set of original hashes that are kept
        kept_indices = []
        duplicated_indices = []

        # Iterate in index order to preserve first occurrence
        for idx, h in tqdm(hashes.items(), desc="Deduplicating internally"):
            # Check if any variant of h is already in kept_hashes
            is_dup = False
            for mask in masks:
                variant = h ^ mask
                if variant in kept_hashes:
                    is_dup = True
                    break
            if is_dup:
                duplicated_indices.append(idx)
            else:
                kept_indices.append(idx)
                kept_hashes.add(h)

        print(f"Kept {len(kept_indices)} unique images, removed {len(duplicated_indices)} duplicates.")
        return kept_indices, duplicated_indices

    def run(self, max_samples: Optional[int] = None, output_prefix: str = "internal_dedup"):
        dataset = self._load_dataset()
        hashes = self._load_or_compute_hashes(dataset, max_samples)

        kept, dup = self.find_internal_duplicates(hashes)

        total = len(hashes)
        pct_kept = len(kept) / total * 100 if total else 0
        pct_dup = len(dup) / total * 100 if total else 0

        print(f"\n--- Results ---")
        print(f"Dataset: {self.dataset_name}")
        print(f"Total images: {total}")
        print(f"Unique images (kept): {len(kept)} ({pct_kept:.2f}%)")
        print(f"Duplicates (removed): {len(dup)} ({pct_dup:.2f}%)")

        # Export CSVs
        kept_csv = f"{output_prefix}_{self.dataset_name.replace('/', '_')}_kept_indices.csv"
        dup_csv = f"{output_prefix}_{self.dataset_name.replace('/', '_')}_duplicated_indices.csv"

        with open(kept_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["index"])
            for idx in kept:
                writer.writerow([idx])

        with open(dup_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["index"])
            for idx in dup:
                writer.writerow([idx])

        print(f"Exported kept indices to {kept_csv}")
        print(f"Exported duplicated indices to {dup_csv}")


if __name__ == "__main__":
    # Example: run on GeoTikz-Base
    dedup = InternalDeduplicator(
        dataset_name="SJY-1995/GeoTikz-Base",
        image_column="image",
        split="train",
        threshold=2,
        cache_dir="./hash_cache",
        num_proc=4
    )
    # For a quick test, set max_samples; remove to process full dataset
    # dedup.run(max_samples=1000, output_prefix="test")
    dedup.run(output_prefix="geotikz_internal")