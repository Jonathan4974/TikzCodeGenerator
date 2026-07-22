# Sketch Agent - synthetic-sketch data augmentation

A clean image is turned into a synthetic "sketch" by one of two methods:

- **UltraSketch** (`nllg/ultrasketch`) - a fine-tuned img2img diffusion model.
- **Random displacement field** - a classical Gaussian-filtered pixel warp, no learning.

The pipeline has two stages:

1. **Offline (`build_sketch_dataset.py`)** - for every row, generates a sketch: one draw
   between UltraSketch (50%) and displacement (50%). Writes rows with
   `image`/`sketch`/`code`/`description`/`source` for the dataset we're building
   (`loss-boss/tikz-dataset`).
2. **Train-time (`sketch_choice_dataset.py`)** - `SketchChoiceDataset` wraps the pushed
   dataset and draws, on every access, whether to feed the clean `image` or the
   precomputed `sketch`.

Composed, this gives 50% clean / 25% ultrasketch / 25% displacement overall.

## Building the training dataset (offline)

```bash
python -m training.sketch_agent.build_sketch_dataset \
    --split datikz_v4 --source-label datikzv4 \
    --split geotikz_bridge_base --source-label geotikz \
    --output-dir training/sketch_agent/output_final/sketch_dataset \
    --max-rows 100

# to resume an interrupted run: re-run the exact same command

python -c "
from datasets import load_dataset
ds = load_dataset('parquet', data_files='training/sketch_agent/output_final/sketch_dataset/shards/*.parquet', split='train')
ds.push_to_hub('your-username/your-repo')
"
```

`--split`/`--source-label` are repeatable and paired positionally, from
`dataset_loader.SPLITS` (`datikz_v4`, `geotikz_bridge_base`, `our_dataset_train`,
`our_dataset_benchmark`). `--source-label` is used unless a row already carries its own
`source` value.

## Consuming the training dataset (train-time)

```python
from training.sketch_agent import SketchChoiceDataset
from datasets import load_dataset

ds = SketchChoiceDataset(load_dataset("loss-boss/tikz-dataset", split="train"))
item = ds[0]  # {"input_image": ..., "code": ..., "description": ..., "source": ..., "used_sketch": bool}
```

Standalone and import-only, meant to be dropped into `training/gemma4_finetuning_grpo_fast/`'s
training loop.

## Files

- `config.py` - `SketchAugmentationConfig`: method params + the two-stage knobs
  (`sketch_always_populated`, `train_time_sketch_probability`) + dataset-sourcing knobs.
- `sketch_generation.py` - `generate_synthetic_sketch(image, seed, sketch_probability, ultrasketch_probability, ...)`,
  the three-way draw (original/ultrasketch/displacement) in one call. Used by
  `build_sketch_dataset.py` with `sketch_probability=1.0` to always substitute.
- `dataset_loader.py` - per-split loading against `loss-boss/tikz-dataset`'s four known
  splits (`SPLITS`). `code_column(split_name)` returns `"response"` for
  `geotikz_bridge_base`, `"code"` for everything else, per the dataset card.
- `check_dataset_schema.py` - manual script, needs network: streams one row per split and
  prints its real columns, to verify the rest of the schema (`our_dataset_train`/
  `our_dataset_benchmark` columns aren't documented on the dataset card).
- `check_sketch_generation.py` - manual script, needs GPU + network: runs both methods
  on real images, saves comparison PNGs under `output_check/sketch_generation/`.
- `build_sketch_dataset.py` - offline generation stage: checkpointed, resumable,
  self-resubmitting via `build_sketch_dataset.sbatch`.
- `build_sketch_dataset.sbatch` - SLURM script `build_sketch_dataset.py` resubmits
  itself through.
- `sketch_choice_dataset.py` - `SketchChoiceDataset`, the train-time loading stage.

## Running

```bash
bash training/sketch_agent/setup_env.sh   # once, to create the sketch-agent conda env
conda activate sketch-agent

pytest training/sketch_agent/tests/

# Verify a split's real columns
python -m training.sketch_agent.check_dataset_schema
python -m training.sketch_agent.check_dataset_schema --splits datikz_v4 our_dataset_benchmark

python -m training.sketch_agent.check_sketch_generation --split datikz_v4 --num-samples 2

python -m training.sketch_agent.build_sketch_dataset \
    --split datikz_v4 --source-label datikzv4 \
    --output-dir training/sketch_agent/output_final/sketch_dataset --max-rows 100
```