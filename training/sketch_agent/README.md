# Sketch Agent — SDXL + ControlNet + LoRA Training

Sketch Agent: hand-drawn/synthetic sketch -> clean, structure-preserved image, feeding
the Structure/Text Agent

## Architecture
- Base: `stabilityai/stable-diffusion-xl-base-1.0`
- ControlNet: `diffusers/controlnet-canny-sdxl-1.0` (default), `xinsir/controlnet-scribble-sdxl-1.0` (fallback if canny underperforms)
- VAE: `madebyollin/sdxl-vae-fp16-fix`
- LoRA on the UNet's attention projections only (`to_q`/`to_k`/`to_v`/`to_out.0`), rank 16 / alpha 16. Text encoders, ControlNet, VAE stay fully frozen.
- Text conditioning: a single fixed prompt for every sample (`config.training_prompt`, `"a clean technical line drawing"`)
- ControlNet conditioning image: Canny edges (`cv2.Canny`, thresholds in config) computed from the sketch.

## Data
- **Synthetic pairs**: clean renders streamed from `nllg/DaTikZ-V4` (`data.py::iter_datikz_renders`), each paired with a sketch generated via exactly one of two methods chosen per-sample at random - UltraSketch (`nllg/ultrasketch`) or a classical displacement-field warp (`ultrasketch_methods.py`). Cached to disk under `synthetic_dir` with a `manifest.jsonl`. Toggle with `cfg.use_synthetic_data` (default `True`).
- **Real data**: `nllg/sketchfig` (500+ real hand-drawn pairs). Split (`real_data.py::load_sketchfig_dataset`) into a small slice into training (`sketchfig_train_fraction`) and the majority held out as the real eval set. Toggle with `cfg.use_sketchfig` (default `True`).
- At least one of the two must be enabled

## Files
- `config.py` - all training/model/data hyperparameters.
- `data.py` - DaTikZ-V4 sourcing + synthetic pair generation + `sketch_to_canny` (canny-conversion helper created here as its used bu multiple files).
- `real_data.py` - SketchFig train/eval split loading.
- `model_loader.py` - loads SDXL UNet/text encoders/VAE/ControlNet, freezes everything except the LoRA adapter
- `trainer.py` - the training loop: checkpoint/resume via `accelerate`, self-resubmit before the 8h SLURM limit, TensorBoard logging, and periodic eval against the real held-out SketchFig examples.
- `train.py` - `main()` wiring config -> dataset -> model -> trainer together.
- `vram_dry_run.py` - loads the real model bundle and runs a few forward/backward steps on random tensors to check VRAM headroom before committing to a full run. Takes `--image-size` to test headroom at a different resolution than `cfg.image_size`.
- `overfit_run.py` - overfit-to-1-sample
- `smoke_run.py` - tiny end-to-end run to test things before a full run.
- `check_sketch_agent.py` - manual check script: loads the trained LoRA weights, runs a few real held-out SketchFig examples, prints pixel-CC/SigLIP/DreamSim scores.
- `eval.py` - `pixel_congruence_coefficient` (cheap secondary metric). SigLIP/DreamSim are called directly from `evaluation/promptfoo/pf_utils`.
- `ultrasketch_methods.py` - UltraSketch/displacement-field helpers.
- `setup_env.sh` - one-time conda env setup
- `train.sbatch` - SLURM launcher.
- `tests/` - tests for the parts that don't need a GPU (Canny/displacement math, checkpoint-resolution logic, SketchFig split math). The two GPU/network-dependent calls (`iter_datikz_renders`, `load_ultrasketch_pipeline`) are mocked via `unittest.mock.patch`.

## Additional Notes
- 8h SLURM job limit; `time_limit_hours` (default 7.5h) leaves a buffer, checkpointing and self-resubmitting (`sbatch --dependency=afterany:$SLURM_JOB_ID train.sbatch`) automatically before hitting it. For a local run with no clock limit, override it: `python -m training.sketch_agent.train --time-limit-hours 24`.
- `HF_HOME` and related HF/torch cache env vars are set in `setup_env.sh`/`train.sbatch`.
- Each TensorBoard run logs to its own named subdirectory under `output/tensorboard/<run_name>`. A resubmit that resumes from a checkpoint reuses the same run name; otherwise a new run name will created `<timestamp>_job<SLURM_JOB_ID>` by default, or set with `cfg.run_name` to specific name.
- On job completion (reaching `max_steps`), `trainer.py` deletes `output/checkpoints/` and the run-name file (`checkpoint_utils.clear_resumable_state`). Set `cfg.save_checkpoints = False` to skip `accelerate` checkpoint saving entirely.

## Running
```bash
bash training/sketch_agent/setup_env.sh                        # conda env setup
python -m training.sketch_agent.vram_dry_run                   # check VRAM headroom before a full run
python -m training.sketch_agent.vram_dry_run --image-size 1024 # at a different resolutions
python -m training.sketch_agent.overfit_run --max-steps 600     # overfit-to-1-sample
sbatch training/sketch_agent/train.sbatch                      # submit the real training job
tensorboard --logdir training/sketch_agent/output/tensorboard  # watch loss/eval curves live
python -m training.sketch_agent.check_sketch_agent             # watch scores on real held-out examples once trained
```

## Tests
```bash
pytest training/sketch_agent/tests/
```
