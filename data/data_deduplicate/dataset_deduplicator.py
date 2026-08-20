"""
Load Hugging Face datasets, compute perceptual hashes for images,
find duplicates between two datasets, and export results.
"""

import os
import csv
import pickle
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import numpy as np
from perceptual_hash import PerceptualHash
from multiprocessing import Pool, cpu_count


class DatasetDeduplicator:
    """Main class for deduplication between two image datasets."""

    def __init__(
        self,
        dataset1_name: str,
        dataset2_name: str,
        image_column: str = 'image',
        split: str = 'train',
        threshold: int = 2,
        cache_dir: str = './hash_cache',
        num_proc: Optional[int] = None,
    ):
        self.ds1_name = dataset1_name
        self.ds2_name = dataset2_name
        self.image_column = image_column
        self.split = split
        self.threshold = threshold
        self.cache_dir = cache_dir
        self.num_proc = num_proc or os.cpu_count() 
        os.makedirs(cache_dir, exist_ok=True)

    def _load_dataset(self, name: str):
        """Load dataset from Hugging Face."""
        print(f"Loading dataset: {name}")
        return load_dataset(name, split=self.split)

    def _compute_hashes(
        self,
        dataset,
        dataset_name: str,
        max_samples: Optional[int] = None
    ) -> Dict[int, int]:
        """
        Compute perceptual hash for each sample using parallel map.
        Returns dict {index: hash_int}.
        Uses cache to avoid recomputation.
        """
        safe_name = dataset_name.replace('/', '_')
        total_samples = len(dataset)
        actual_count = max_samples if max_samples is not None else total_samples
        cache_path = os.path.join(self.cache_dir, f"{safe_name}_{actual_count}_hashes.pkl")
        if os.path.exists(cache_path):
            print(f"Loading cached hashes from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # limit sample number
        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))

        print(f"Computing hashes for {len(dataset)} images using {self.num_proc} processes...")

        # define the hash function for each sample
        def hash_sample(sample):
            try:
                image = sample[self.image_column]
                ph = PerceptualHash(image)
                h = ph.compute()
                return {'hash': str(h)}
            except Exception:
                return {'hash': None}     # error mark

        # parallel map：remove all original columns, keep only 'hash' column
        mapped = dataset.map(
            hash_sample,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,   # save memory
            keep_in_memory=False,                  # write to disk cache to avoid memory explosion
            desc=f"Hashing {dataset_name}"
        )

        # extract hash values, filter errors
        hashes = {}
        for idx, h_str in enumerate(mapped['hash']):
            if h_str is not None:
                try:
                    hashes[idx] = int(h_str)
                except ValueError:
                    continue

        # cache results
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(hashes, f)

        print(f"Cached hashes to {cache_path}")
        return hashes

    # def _compute_hashes(
    #     self,
    #     dataset,
    #     dataset_name: str,
    #     max_samples: Optional[int] = None
    # ) -> Dict[int, int]:
    #     """
    #     Compute perceptual hash for each sample in the dataset.
    #     Returns dict {index: hash_int}.
    #     Uses cache to avoid recomputation.
    #     """
    #     cache_path = os.path.join(self.cache_dir, f"{dataset_name}_hashes.pkl")
    #     if os.path.exists(cache_path):
    #         print(f"Loading cached hashes from {cache_path}")
    #         with open(cache_path, 'rb') as f:
    #             return pickle.load(f)

    #     hashes = {}
    #     total = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    #     # Use a simple loop; for large datasets, consider multiprocessing
    #     for idx in tqdm(range(total), desc=f"Hashing {dataset_name}"):
    #         sample = dataset[idx]
    #         image = sample.get(self.image_column)
    #         if image is None:
    #             continue
    #         # image may be PIL Image or bytes
    #         try:
    #             ph = PerceptualHash(image)
    #             h = ph.compute()
    #             hashes[idx] = h
    #         except Exception as e:
    #             print(f"Error at index {idx}: {e}")
    #             continue

    #     # Save cache
    #     os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    #     with open(cache_path, 'wb') as f:
    #         pickle.dump(hashes, f)
    #     return hashes

    def find_duplicates(
        self,
        hashes1: Dict[int, int],
        hashes2: Dict[int, int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Find all pairs (idx1, idx2) where hashes are within threshold Hamming distance.
        Returns:
            - duplicate_pairs: list of (idx1, idx2)
            - duplicated_indices1: indices in dataset1 that have any duplicate in dataset2
            - duplicated_indices2: indices in dataset2 that have any duplicate in dataset1
        """
        # We'll use the smaller dataset to generate variants (faster)
        if len(hashes1) <= len(hashes2):
            small_hashes, large_hashes = hashes1, hashes2
            small_is_first = True
        else:
            small_hashes, large_hashes = hashes2, hashes1
            small_is_first = False

        # Build a dict mapping hash -> list of indices in the large dataset
        large_dict = defaultdict(list)
        for idx, h in large_hashes.items():
            large_dict[h].append(idx)

        # For each hash in small set, generate all variants within distance <= threshold
        # and check if any variant exists in large_dict
        duplicate_pairs = []
        duplicated_small = set()
        duplicated_large = set()

        # Precompute all bit masks for flipping up to `threshold` bits
        # For threshold=2, we generate 1 + 64 + C(64,2) = 2081 masks
        def gen_masks(threshold: int, bits: int = 64):
            masks = [0]  # distance 0
            if threshold >= 1:
                # distance 1: flip one bit
                for i in range(bits):
                    masks.append(1 << i)
            if threshold >= 2:
                # distance 2: flip two bits
                for i in range(bits):
                    for j in range(i+1, bits):
                        masks.append((1 << i) | (1 << j))
            return masks

        masks = gen_masks(self.threshold)
        print(f"Generated {len(masks)} variant masks for threshold={self.threshold}")

        # Process small hashes in batches to save memory
        for idx, h in tqdm(small_hashes.items(), desc="Finding duplicates"):
            found = False
            for mask in masks:
                variant = h ^ mask   # XOR flips bits
                if variant in large_dict:
                    # There is at least one match
                    for large_idx in large_dict[variant]:
                        duplicate_pairs.append((idx, large_idx) if small_is_first else (large_idx, idx))
                    found = True
                    duplicated_small.add(idx)
                    duplicated_large.update(large_dict[variant])
            # If we found multiple matches for same small idx, we have multiple pairs; that's okay

        # Prepare result lists
        if small_is_first:
            dup_indices1 = list(duplicated_small)
            dup_indices2 = list(duplicated_large)
        else:
            dup_indices1 = list(duplicated_large)
            dup_indices2 = list(duplicated_small)

        # Ensure each index appears only once
        dup_indices1 = sorted(set(dup_indices1))
        dup_indices2 = sorted(set(dup_indices2))

        return duplicate_pairs, dup_indices1, dup_indices2

    def run(
        self,
        max_samples1: Optional[int] = None,
        max_samples2: Optional[int] = None,
        output_prefix: str = "duplicates"
    ) -> None:
        """Main execution: load datasets, compute hashes, find duplicates, export."""
        # Load datasets
        ds1 = self._load_dataset(self.ds1_name)
        ds2 = self._load_dataset(self.ds2_name)

        # Compute hashes
        hashes1 = self._compute_hashes(ds1, self.ds1_name, max_samples1)
        hashes2 = self._compute_hashes(ds2, self.ds2_name, max_samples2)

        # Find duplicates
        pairs, dup1, dup2 = self.find_duplicates(hashes1, hashes2)

        # Calculate percentages
        total1 = len(hashes1)
        total2 = len(hashes2)
        pct1 = len(dup1) / total1 * 100 if total1 else 0
        pct2 = len(dup2) / total2 * 100 if total2 else 0

        print(f"\n--- Results ---")
        print(f"Dataset1 ({self.ds1_name}): total={total1}, duplicated={len(dup1)} ({pct1:.2f}%)")
        print(f"Dataset2 ({self.ds2_name}): total={total2}, duplicated={len(dup2)} ({pct2:.2f}%)")
        print(f"Duplicate pairs found: {len(pairs)}")

        # Export CSV: indices of duplicated items in each dataset
        csv1 = f"{output_prefix}_{self.ds1_name.replace('/', '_')}_duplicated_indices.csv"
        csv2 = f"{output_prefix}_{self.ds2_name.replace('/', '_')}_duplicated_indices.csv"

        with open(csv1, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["index"])
            for idx in dup1:
                writer.writerow([idx])

        with open(csv2, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["index"])
            for idx in dup2:
                writer.writerow([idx])

        print(f"Exported duplicated indices to {csv1} and {csv2}")

        # Optionally export all pairs
        pairs_csv = f"{output_prefix}_pairs.csv"
        with open(pairs_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["index_in_dataset1", "index_in_dataset2"])
            for i1, i2 in pairs:
                writer.writerow([i1, i2])
        print(f"Exported duplicate pairs to {pairs_csv}")