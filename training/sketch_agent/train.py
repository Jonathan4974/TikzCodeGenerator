from __future__ import annotations

import argparse

from .config import SketchAgentConfig, build_training_config, seed_everything
from .data import SketchAgentDataset
from .model_loader import SketchAgentModelLoader
from .real_data import load_sketchfig_dataset
from .trainer import SketchAgentTrainer

# output_dir/checkpoint_dir/lora_output_dir are namespaced under here per <conditioning_mode>_<sketchfig_only|mixed>
EXPERIMENTS_DIR = "training/sketch_agent/output/experiments"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit-hours", type=float, default=None)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="explicit run name; auto-generated (timestamp + job id) if unset",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="overrides cfg.max_steps")
    parser.add_argument(
        "--conditioning-mode",
        type=str,
        default=None,
        choices=["canny", "scribble", "lineart", "anime_lineart"],
        help="overrides cfg.conditioning_mode; also namespaces output paths under "
        f"{EXPERIMENTS_DIR}/<mode>_<sketchfig_only|mixed>/ so different experiments never collide",
    )
    parser.add_argument(
        "--use-synthetic-data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="overrides cfg.use_synthetic_data (--no-use-synthetic-data for SketchFig-only training)",
    )
    args = parser.parse_args()

    defaults = SketchAgentConfig()
    conditioning_mode = args.conditioning_mode or defaults.conditioning_mode
    use_synthetic_data = defaults.use_synthetic_data if args.use_synthetic_data is None else args.use_synthetic_data
    data_tag = "mixed" if use_synthetic_data else "sketchfig_only"
    experiment_dir = f"{EXPERIMENTS_DIR}/{conditioning_mode}_{data_tag}"

    overrides = {
        "output_dir": experiment_dir,
        "checkpoint_dir": f"{experiment_dir}/checkpoints",
        "lora_output_dir": f"{experiment_dir}/lora",
        "conditioning_mode": conditioning_mode,
        "use_synthetic_data": use_synthetic_data,
    }
    if args.time_limit_hours is not None:
        overrides["time_limit_hours"] = args.time_limit_hours
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    cfg = build_training_config(overrides)
    seed_everything(cfg.seed)

    sketchfig = (
        load_sketchfig_dataset(
            cfg.sketchfig_cache_dir,
            train_fraction=cfg.sketchfig_train_fraction,
            seed=cfg.sketchfig_split_seed,
            dataset_name=cfg.sketchfig_dataset_name,
        )
        if cfg.use_sketchfig
        else None
    )

    train_dataset = SketchAgentDataset(cfg, extra_pairs=sketchfig.train if sketchfig else [])
    models = SketchAgentModelLoader(cfg).load()
    trainer = SketchAgentTrainer(
        cfg=cfg,
        models=models,
        train_dataset=train_dataset,
        eval_pairs=sketchfig.eval if sketchfig else [],
    )
    trainer.train()


if __name__ == "__main__":
    main()
