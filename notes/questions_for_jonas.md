# Questions for Jonas — before finalizing report/slides

Open items found while auditing the repo for the final report/presentation, where the answer
lives outside the repo (e.g. on his machine/Synology) or isn't verifiable from checked-in code.

1. **How was the synthetic sketch data (`03kiko/tikz-sketch-splits`) actually used in training?**
   - `training/sketch_agent/` builds standalone `ultrasketch`/`displacement` splits from
     `loss-boss/tikz-train`, but nothing in `data/data-prep-tikz/` (the pipeline that builds
     `tikz-dataset-clean(-extended)`, which `gemma4_finetuning` actually trains on) references
     `sketch`, `03kiko`, or `tikz-sketch-splits` anywhere.
   - `gemma4-sft/data.py` / `gemma4-rl/data.py` / `common/data_utils.py` also have no
     sketch-related logic — no per-epoch clean-vs-sketch choice, no A+B merge visible.
   - So: was the merge done manually outside the repo when building
     `/home/jonas/Datasets/TikZ/tikz-dataset-clean-extended/`? Or was the sketch-augmentation
     pipeline built but never actually folded into the SFT/RL training run?
   - This directly determines what the "Synthetic Sketch Generation" slide can honestly claim.
