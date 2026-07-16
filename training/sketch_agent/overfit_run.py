"""Overfit-to-1-sample: trains on one real SketchFig (sketch, clean_render) pair
and evaluates on that same pair.

Usage:
  conda activate sketch-agent
  python -m training.sketch_agent.overfit_run --max-steps 600
  python -m training.sketch_agent.overfit_run --max-steps 1200           # resumes from checkpoint
  python -m training.sketch_agent.overfit_run --sketchfig-index 3        # pick a different held-out example
  python -m training.sketch_agent.overfit_run --conditioning-mode canny  # canny/scribble/lineart/anime_lineart

Then:
  tensorboard --logdir training/sketch_agent/output_overfit/<conditioning_mode>/tensorboard
  training/sketch_agent/output_overfit/<conditioning_mode>/eval_previews/step_*/0.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import SketchAgentConfig, build_training_config, seed_everything
from .data import SketchAgentDataset, SyntheticPair
from .model_loader import SketchAgentModelLoader
from .trainer import SketchAgentTrainer

OVERFIT_DIR = "training/sketch_agent/output_overfit"


def _pick_sketchfig_pair(cache_dir: str, index: int) -> SyntheticPair:
    """Picks one already-cached SketchFig pair by index, without calling
    real_data.py::load_sketchfig_dataset (which re-writes every cached image to disk on every
    call - wasteful just to pick one)."""
    eval_dir = Path(cache_dir) / "eval"
    input_paths = sorted(eval_dir.glob("*_input.png"))
    if not input_paths:
        raise FileNotFoundError(
            f"no cached SketchFig pairs under {eval_dir} - populate it first via train.py/"
            f"smoke_run.py, or:\n"
            f'  python -c "from training.sketch_agent.config import build_training_config; '
            f"from training.sketch_agent.real_data import load_sketchfig_dataset; "
            f"cfg = build_training_config(); load_sketchfig_dataset(cfg.sketchfig_cache_dir, "
            f"train_fraction=cfg.sketchfig_train_fraction, seed=cfg.sketchfig_split_seed, "
            f'dataset_name=cfg.sketchfig_dataset_name)"'
        )
    if not 0 <= index < len(input_paths):
        raise IndexError(f"--sketchfig-index {index} out of range: {len(input_paths)} cached pairs under {eval_dir}")
    input_path = input_paths[index]
    target_path = eval_dir / input_path.name.replace("_input", "_target")
    return SyntheticPair(
        input_path=input_path,
        target_path=target_path,
        source_name=input_path.stem.replace("_input", ""),
        method="real_sketchfig",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=50)
    parser.add_argument(
        "--sketchfig-index",
        type=int,
        default=0,
        help="which cached held-out SketchFig example to overfit to (0-indexed, sorted "
        "alphabetically - same order as check_sketch_agent.py's TEST_PAIRS)",
    )
    parser.add_argument(
        "--conditioning-mode",
        type=str,
        default=None,
        choices=["canny", "scribble", "lineart", "anime_lineart"],
        help="overrides cfg.conditioning_mode; also namespaces output paths under "
        "output_overfit/<mode>/ so different modes never collide",
    )
    parser.add_argument("--run-name", type=str, default=None, help="overrides cfg.run_name")
    args = parser.parse_args()

    conditioning_mode = args.conditioning_mode or SketchAgentConfig().conditioning_mode
    mode_dir = f"{OVERFIT_DIR}/{conditioning_mode}"

    overrides = {
        "output_dir": mode_dir,
        "checkpoint_dir": f"{mode_dir}/checkpoints",
        "lora_output_dir": f"{mode_dir}/lora",
        "conditioning_mode": conditioning_mode,
        "max_steps": args.max_steps,
        "checkpoint_interval_steps": args.checkpoint_interval_steps,
        "use_synthetic_data": False,
        "use_sketchfig": False,
        "eval_sample_size": 1,
        "self_resubmit": False,
    }
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    cfg = build_training_config(overrides)
    seed_everything(cfg.seed)

    pair = _pick_sketchfig_pair(cfg.sketchfig_cache_dir, args.sketchfig_index)
    print(f"Overfitting to: {pair.source_name} ({pair.input_path})")

    train_dataset = SketchAgentDataset(cfg, extra_pairs=[pair])
    assert len(train_dataset.pairs) == 1, f"expected exactly 1 pair, got {len(train_dataset.pairs)}"

    models = SketchAgentModelLoader(cfg).load()
    trainer = SketchAgentTrainer(
        cfg=cfg,
        models=models,
        train_dataset=train_dataset,
        eval_pairs=train_dataset.pairs,  # eval on the exact same sample we're training on
    )
    trainer.train()
    print(f"Overfit run finished at max_steps={args.max_steps}, output under {OVERFIT_DIR}")


if __name__ == "__main__":
    main()
