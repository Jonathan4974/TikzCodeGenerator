"""Independent mode orchestration for the TikZ dataset pipeline."""

from dataclasses import dataclass
from math import gcd
from pathlib import Path
import gc
import json
import random

from tqdm import tqdm

import config
from cleaning import DeterministicCleaner
from ollama_client import OllamaClient
from processing import (
    ProcessingMode,
    SampleProcessingError,
    SampleProcessor,
)
from rendering import ValidatedRenderer
from storage import ParquetShardWriter, SplitLoader
from token_counting import LatexTokenCounter


@dataclass(frozen=True)
class ModePlan:
    mode: ProcessingMode
    num: int
    absolute_num: int

    @classmethod
    def from_config(cls, raw: dict) -> "ModePlan":
        plan = cls(
            mode=ProcessingMode(raw["type"]),
            num=int(raw["num"]),
            absolute_num=int(raw["absolute_num"]),
        )
        if plan.num < 0 or plan.absolute_num < 0:
            raise ValueError("num and absolute_num must be non-negative.")
        if plan.num > plan.absolute_num:
            raise ValueError("num cannot exceed absolute_num.")
        return plan


@dataclass(frozen=True)
class PreparedSample:
    index: int
    original_code: str
    token_count: int


@dataclass
class SelectionStats:
    checked: int = 0
    empty: int = 0
    too_long: int = 0


@dataclass
class ModeStats:
    requested: int = 0
    processed: int = 0
    mode_attempted: int = 0
    mode_saved: int = 0
    base_only_saved: int = 0
    rejected: int = 0

    @property
    def saved(self) -> int:
        return self.mode_saved + self.base_only_saved


class TikzDatasetPipeline:
    def __init__(self) -> None:
        self.cleaner = DeterministicCleaner()
        self.ollama = OllamaClient(self.cleaner)
        self.renderer = ValidatedRenderer()
        self.token_counter = LatexTokenCounter()
        self.processor = SampleProcessor(self.cleaner, self.ollama, self.renderer)
        self.loader = SplitLoader()
        self.failure_dir = config.OUTPUT_DIR / "failures"
        self.failure_dir.mkdir(parents=True, exist_ok=True)

    def check_dependencies(self) -> None:
        plans = self._all_plans()
        if any(plan.num > 0 for plan in plans):
            self.ollama.check_connection()
        if any(
            plan.mode
            in {ProcessingMode.DETERMINISTIC_CLEANING, ProcessingMode.FULL_CLEANING}
            and plan.num > 0
            for plan in plans
        ):
            self.renderer.check_dependencies()

    def run_split(self, split_name: str, raw_plans: list[dict]) -> None:
        plans = [ModePlan.from_config(raw) for raw in raw_plans]
        self._validate_unique_modes(split_name, plans)

        print(f"\n{'=' * 72}\nSplit: {split_name}\n{'=' * 72}")
        self.loader.clear_cache()
        dataset = self.loader.load(split_name)

        try:
            max_absolute = max((plan.absolute_num for plan in plans), default=0)
            prepared, selection_stats = self._prepare_shared_samples(
                dataset=dataset,
                split_name=split_name,
                target=max_absolute,
            )

            print(
                f"Shared selection: {len(prepared):,}/{max_absolute:,} valid; "
                f"{selection_stats.checked:,} checked, "
                f"{selection_stats.too_long:,} over token limit, "
                f"{selection_stats.empty:,} empty"
            )

            for plan in plans:
                self._run_mode(split_name, dataset, prepared, plan)
        finally:
            del dataset
            gc.collect()
            self.loader.clear_cache()

    def _prepare_shared_samples(
        self,
        dataset,
        split_name: str,
        target: int,
    ) -> tuple[list[PreparedSample], SelectionStats]:
        """Create one seeded sample order shared by every mode."""

        if target > len(dataset):
            raise ValueError(
                f"absolute_num={target} exceeds split size={len(dataset)}."
            )

        prepared: list[PreparedSample] = []
        stats = SelectionStats()

        for index in self._random_index_order(len(dataset), split_name):
            if len(prepared) >= target:
                break

            stats.checked += 1
            sample = dataset[index]
            original_code = str(sample.get(config.CODE_WITH_TEXT_COL) or "").strip()

            if not original_code:
                stats.empty += 1
                continue

            fits, token_count = self.token_counter.fits(original_code)
            if not fits:
                stats.too_long += 1
                continue

            prepared.append(
                PreparedSample(
                    index=index,
                    original_code=original_code,
                    token_count=token_count,
                )
            )

        return prepared, stats

    def _run_mode(
        self,
        split_name: str,
        dataset,
        prepared: list[PreparedSample],
        plan: ModePlan,
    ) -> ModeStats:
        selected = prepared[: plan.absolute_num]
        active_count = min(plan.num, len(selected))
        stats = ModeStats(requested=plan.absolute_num)
        writer = ParquetShardWriter(split_name, plan.mode.value)

        print(
            f"\nMode: {plan.mode.value} | active={plan.num:,} | "
            f"absolute={plan.absolute_num:,}"
        )

        try:
            for position, item in enumerate(
                tqdm(selected, desc=f"{split_name}:{plan.mode.value}")
            ):
                stats.processed += 1

                if position >= active_count:
                    writer.add(
                        self.processor.base_only(
                            output_mode=plan.mode,
                            original_code=item.original_code,
                        )
                    )
                    stats.base_only_saved += 1
                    continue

                stats.mode_attempted += 1
                try:
                    row = self.processor.process(
                        mode=plan.mode,
                        sample=dataset[item.index],
                        original_code=item.original_code,
                    )
                except Exception as error:
                    stats.rejected += 1
                    self._log_failure(split_name, item, plan.mode, error)
                    continue

                writer.add(row)
                stats.mode_saved += 1
        finally:
            writer.close()

        if len(selected) < plan.absolute_num:
            print(
                f"Warning: only {len(selected):,}/{plan.absolute_num:,} shared "
                "valid samples were available."
            )

        print(
            f"{split_name}/{plan.mode.value}: {stats.saved:,} saved; "
            f"{stats.mode_saved:,}/{stats.mode_attempted:,} active-mode rows, "
            f"{stats.base_only_saved:,} base-only rows, "
            f"{stats.rejected:,} rejected"
        )
        return stats

    def _all_plans(self) -> list[ModePlan]:
        return [
            ModePlan.from_config(raw)
            for raw_plans in config.SPLITS.values()
            for raw in raw_plans
        ]

    @staticmethod
    def _validate_unique_modes(split_name: str, plans: list[ModePlan]) -> None:
        modes = [plan.mode for plan in plans]
        if len(modes) != len(set(modes)):
            raise ValueError(
                f"Each mode may appear only once per split: {split_name}."
            )

    @staticmethod
    def _random_index_order(size: int, split_name: str):
        """Visit rows once in a deterministic random order shared by all modes."""

        if size <= 0:
            return

        rng = random.Random(f"{config.RANDOM_SEED}:rows:{split_name}")
        start = rng.randrange(size)

        if size == 1:
            yield 0
            return

        step = rng.randrange(1, size)
        while gcd(step, size) != 1:
            step = rng.randrange(1, size)

        for offset in range(size):
            yield (start + offset * step) % size

    def _log_failure(
        self,
        split_name: str,
        item: PreparedSample,
        mode: ProcessingMode,
        error: Exception,
    ) -> None:
        if isinstance(error, SampleProcessingError):
            stage = error.stage
            latest_code = error.latest_code
            cause = error.cause
        else:
            stage = "unknown"
            latest_code = item.original_code
            cause = error

        text_dir = self.failure_dir / split_name / mode.value
        text_dir.mkdir(parents=True, exist_ok=True)
        text_path = text_dir / f"sample-{item.index:09d}.txt"

        text_path.write_text(
            self._failure_text(
                split_name=split_name,
                index=item.index,
                mode=mode,
                token_count=item.token_count,
                stage=stage,
                error=cause,
                original_code=item.original_code,
                latest_code=latest_code,
            ),
            encoding="utf-8",
        )

        jsonl_path = self.failure_dir / f"{split_name}-{mode.value}.jsonl"
        record = {
            "split": split_name,
            "index": item.index,
            "mode": mode.value,
            "stage": stage,
            "token_count": item.token_count,
            "error_type": type(cause).__name__,
            "error": str(cause),
            "details_file": str(text_path),
        }
        with jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _failure_text(
        split_name: str,
        index: int,
        mode: ProcessingMode,
        token_count: int,
        stage: str,
        error: Exception,
        original_code: str,
        latest_code: str,
    ) -> str:
        divider = "=" * 80
        return (
            f"Split: {split_name}\n"
            f"Index: {index}\n"
            f"Mode: {mode.value}\n"
            f"Stage: {stage}\n"
            f"Token count: {token_count}\n"
            f"Error type: {type(error).__name__}\n"
            f"Error: {error}\n\n"
            f"{divider}\nORIGINAL CODE\n{divider}\n"
            f"{original_code}\n\n"
            f"{divider}\nLATEST NEW CODE\n{divider}\n"
            f"{latest_code}\n"
        )
