# Sketch Agent - synthetic-sketch data augmentation

Builds two new method-dedicated splits directly on `loss-boss/tikz-train`: one where every
row's image is UltraSketch output, one where every row's image is displacement-field output.

## Source data

`loss-boss/tikz-train` is one split, `train` (410k rows)

- `image_with_text` / `code_with_text` / `llm_description_with_text`
- `image_without_text_full` / `code_without_text_full` / `llm_description_without_text_full`
  (only populated for ~10k of the 410k rows - the "no text" augmentation slice; `None` for
  the rest, not just a missing description)

For each row, `dataset_loader.pick_variant` randomly (50/50) picks one variant and returns
its **correctly paired** image/code/description - and falls back to the other variant if the
drawn one isn't populated for that row (confirmed necessary: crashed a real job otherwise,
`PIL.Image.open(None)` on a row without the without-text variant).

## Generating a split

Every row in a given run is sketch-ified by the one method that run is dedicated to.

Two dedicated scripts, already set up for a 100-row first pass:

```bash
sbatch training/sketch_agent/build_sketch_dataset_displacement.sbatch  # CPU-only, no GPU requested
sbatch training/sketch_agent/build_sketch_dataset_ultrasketch.sbatch   # needs a GPU
```

Once the 100-row pass looks correct, scale up by editing the `--max-rows 100` line in
whichever script (or override without editing: `sbatch build_sketch_dataset_ultrasketch.sbatch --max-rows 500000` -
extra args are appended after the hardcoded ones and win, since argparse keeps the last
value of a repeated flag).

Both self-resubmit (`sbatch --dependency=afterany:$SLURM_JOB_ID <the same script>` - each
passes `--sbatch-script "$0"` so it always resubmits *itself*, not the other one) if the 8h
limit is hit before finishing.

**Interactively / not on SLURM** - the two commands the scripts above wrap:

```bash
python -m training.sketch_agent.build_sketch_dataset \
    --method displacement \
    --output-dir training/sketch_agent/output_final/sketch_dataset_displacement \
    --max-rows 100

python -m training.sketch_agent.build_sketch_dataset \
    --method ultrasketch \
    --output-dir training/sketch_agent/output_final/sketch_dataset_ultrasketch \
    --max-rows 100
```

## Pushing both splits

Once both output dirs have `DONE` (or you're pushing a partial/first-pass run), combine and
push in **one** `DatasetDict.push_to_hub` call - pushing splits individually, or with
different tooling per split, is what causes the HF viewer's
`FileFormatMismatchBetweenSplitsError`:

```python
from datasets import DatasetDict
from training.sketch_agent.build_sketch_dataset import assemble_dataset_dict

displacement = assemble_dataset_dict("training/sketch_agent/output_final/sketch_dataset_displacement")
ultrasketch = assemble_dataset_dict("training/sketch_agent/output_final/sketch_dataset_ultrasketch")
DatasetDict({**displacement, **ultrasketch}).push_to_hub("loss-boss/tikz-train")
```

Each new split's rows have: `image` (the sketch - this split's whole point is every row's
image is the sketch, there's no separate clean-image column here), `code` (correctly paired
per the rule above), `description` (may be null), `source_variant` (`"with_text"` or
`"without_text"` - which source pair this row came from), `sketch_method` (`"ultrasketch"`
or `"displacement"`: constant within one split, kept for clarity). This exact set is also
`_build_features()`'s schema in `build_sketch_dataset.py` - keep the two in sync (a prior
mismatch, the schema missing `code`, crashed a real job with `KeyError: 'code'` inside
`datasets`' `encode_column`; regression-tested in `test_write_shard_accepts_a_real_row_with_every_buffered_field`).

`assemble_dataset_dict(..., rename={...})` renames the inferred split (the `--split-name`
value, default `--method`) before pushing, if you want the pushed splits named something
other than `ultrasketch`/`displacement`.


## Files

- `config.py` - `SketchAugmentationConfig`: method params (`displacement_alpha`/`sigma`) +
  `max_rows`. `displacement_alpha=8.0`/`displacement_sigma=3.0` confirmed (visually, on a
  100-row pass) to produce a clearly visible hand-drawn wobble at our real 512x512 render
  size - a wide `sigma` (e.g. the `data/ultrasketch_quality_test.py` reference's `alpha=6,
  sigma=12`) pushes nearby points on the same line in the same direction, i.e. a rigid,
  barely-visible shift rather than a tremor; a short-wavelength (low-`sigma`) field is what
  actually reads as hand-drawn.
- `sketch_generation.py` - `generate_synthetic_sketch(image, seed, sketch_probability, ultrasketch_probability, ...)`.
  `build_sketch_dataset.py` always calls this with `sketch_probability=1.0` (every row in a
  dedicated split is substituted) and `ultrasketch_probability` forced to `0.0`/`1.0` per the
  run's `--method` (never a real per-row mix in this pipeline's usage, though the function
  itself still supports one).
- `dataset_loader.py` - loads `loss-boss/tikz-train`'s single `train` split;
  `pick_variant(row, seed)` does the correctly-paired with-text/without-text-full draw.
- `check_dataset_schema.py` - manual script, needs network: streams a few `train` rows and
  prints their real columns.
- `check_sketch_generation.py` - manual script, needs GPU + network: runs both methods on a
  couple of real images (using whichever variant `pick_variant` draws), saves comparison
  PNGs under `output_check/sketch_generation/`.
- `build_sketch_dataset.py` - offline generation stage: checkpointed, resumable,
  self-resubmitting via whichever `--sbatch-script` launched it.
- `build_sketch_dataset_displacement.sbatch` / `build_sketch_dataset_ultrasketch.sbatch` -
  dedicated SLURM launchers, one per method, so they run as independent jobs (displacement
  requests no GPU). `build_sketch_dataset.sbatch` (generic, `"$@"`-driven) still exists too,
  for ad hoc/other args.

## Running

```bash
bash training/sketch_agent/setup_env.sh   # once, to create the sketch-agent conda env
conda activate sketch-agent

pytest training/sketch_agent/tests/

# Verify the real columns
python -m training.sketch_agent.check_dataset_schema
python -m training.sketch_agent.check_dataset_schema --num-rows 5

python -m training.sketch_agent.check_sketch_generation --num-samples 2

python -m training.sketch_agent.build_sketch_dataset \
    --method displacement \
    --output-dir training/sketch_agent/output_final/sketch_dataset_displacement --max-rows 100
```
