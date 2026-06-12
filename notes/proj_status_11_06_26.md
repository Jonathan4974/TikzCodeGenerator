# Project Status — Sketch to TikZ with VLM
**Last updated:** 12 June 2026  
**Week:** 4 of ~10  
**Final demo/presentation:** ~mid/end July 2026 (TBC)  

---

## Overall Progress Summary

| Milestone | Status | Notes |
|---|---|---|
| M0 — Organisation | ✅ Done|
| M1 — Data | 🔄 In Progress | Blocked on storage admin |
| M2 — Benchmarking | 🔄 In Progress | Blocked on test dataset + Docker permissions on server |
| M3 — Training | ⏳ Not started | Blocked on M2 completion + architecture decision |
| M4 — Evaluation | ⏳ Not started | Blocked on M3 |
| M5 — Demo & Report | ⏳ Not started | Blocked on M4 |

**Critical path:** test dataset creation → benchmark runs end-to-end → architecture decision → training starts

---

## M0 — Organisation ✅

**Done:**
- GitLab board set up with milestones, labels, issue board
- Repo structure created (`/data`, `/benchmarking`, `/evaluation`, `/training`, `/demo`, `/notes`)
- GPU server access confirmed for all team members
- conda base environment set up with pinned dependencies
- wandb project created (teammates still need to join)
- Storage admin (Quirin) contacted — awaiting reply
- READMEs written for all folders

**Pending:**
- ⏳ wandb — all teammates join shared project when training starts
- ⏳ Storage admin reply

---

## M1 — Data 🔄

**Done:**
- SketchFig audited → `/data/sketchfig_audit.md`
  - 549 examples, no TikZ code, no captions
  - sketch + rendered figure pairs only
- DaTikZv4 audited + arXiv extraction completed
- GeoTikZ audited
- UltraSketch tested on GPU server (node9) → `/data/ultrasketch_audit.md`
  - Pixel-level CC computed on 5 examples
  - Visual quality assessed
- Synthetic sketch pipeline approach: **TikZero hybrid approach** ← current SOTA for this task
  - UltraSketch (CC 0.74) + random displacement field (CC 0.75) = **CC 0.82**
  - Scripts: `data/ultrasketch_quality_test.py`
- Similarity metrics researched → `/data/sketch_similarity_metrics.md`
  - lot of them already implemented (`evaluation` folder)
- Architecture proposal written → `/notes/architecture_proposal.md`

**Pending:**
- ⏳ Dataset download to central storage — blocked on admin reply
- ⏳ URI cross-reference SketchFig ↔ DaTikZv2 — ?
- ⏳ Test dataset creation (#18) — **blocker for M2**
  - Need: hand-drawn figures (photo + scan) + synthetic sketches
- ⏳ SciTikZ-230K audit — not confirmed done
- ⏳ Scale synthetic sketch generation to full dataset — blocked on storage

**Note on synthetic sketch quality:**
CC 0.82 matches TikZero SOTA. Fine-tuning UltraSketch further is deprioritized — the higher-leverage improvement is scaling the existing pipeline. The GeoTikZBridge localized code masking technique will be used as a VLM training signal (see M3), not as a sketch quality improvement.

---

## M2 — Benchmarking 🔄

### Metrics
| Metric | Owner | Status |
|---|---|---|
| CrystalBLEU (cBLEU) | — | ✅ Done |
| Token Edit Distance (TED) | — | ✅ Done |
| Output length vs reference | — | ✅ Done |
| Time to generate | — | ✅ Done |
| DreamSim | ✅ Done |
| CC SigLIP embedding | ✅ Done |
| CLIP similarity | — | ✅ Done |
| SigLIP similarity | — | ✅ Done |
| FID | 🔴 Blocked — group metric issue in promptfoo |
| KID | 🔴 Blocked — group metric issue in promptfoo |

**FID/KID blocker:** These metrics require groups of images, not pairs. promptfoo handles pairs only. Resolution options: run FID/KID outside promptfoo as a separate script, or drop them from the benchmark. .

### Models
**Image → TikZ (specialized):**
| Model | Owner | Status |
|---|---|---|
| DeTikZify-v2-8b | ✅ Done |
| DeTikZify-v2.5-8b | ✅ Done |
| TikZero-plus-10b | ✅ Done |
| GeoTikZBridge-Instruct-8B | ✅ Wrapper created, untested |
| AutomaTikZ | ✅ Wrapper created, untested |

**Text → TikZ:**
| Model | Owner | Status |
|---|---|---|
| AutomaTikZ | ✅ Wrapper created, untested (text input only — no image) |
| TikZilla | — | 🔄 Not done |

**General VLMs:**
| Model | Owner | Status |
|---|---|---|
| ChatGPT-latest | ✅ Done (~$5 API budget used) |
| DeepSeek | 🔄 Not done |
| Qwen3.6-27B | 🔄 Not done |
| Gemma-4 | 🔄 Not done |

### Infrastructure
| Component | Status | Notes |
|---|---|---|
| Docker — local | ✅ Working |
| Docker — GPU server | 🔴 Blocked | Permission denied on `/var/run/docker.sock` |
| promptfoo — local | ✅ Working | |
| promptfoo — GPU server | ⏳ Not verified | Blocked on Docker permissions above |
| Full end-to-end benchmark run | ⏳ Not done | Blocked on test dataset + server Docker |

**promptfoo setup (local):** Follow README. Pattern: model containers expose FastAPI endpoints, promptfoo calls them via HTTP (`multipart image upload → POST /modelname → JSON {tikz: "..."}`). Port forwarding needed to view promptfoo UI from local browser when running on server.

---

## M3 — Training ⏳

**Not started.** Initial Architecture proposal ready needs discussion.

### Proposed Architecture (from `/notes/architecture_proposal.md`)

| Component | Approach | Source | Note |
|---|---|---|---|
| Learning paradigm | DSC-RL (GRPO) | SciTikZer | Paper only — model not publicly available |
| SFT strategy | Localized code masking (40%) | GeoTikZBridge | Data processing step, no extra training cost |
| Inference | MCTS + compiler feedback | DeTikZify | |
| Data engine | Execution-centric repair + VLM descriptions | TikZilla | |
| RL reward | Domain-specific SigLIP critic + EMD | TikZilla | |

**Base model:** DeTikZify-v2 weights as starting point.

**Note on SciTikZer:** Model weights not publicly released. DSC-RL approach (dual self-consistency via GRPO) will be implemented based on the paper description, not reused directly.

**Note on localized code masking (GeoTikZBridge):** Randomly removing 40% of TikZ code lines and rendering partial figures creates training pairs that force the VLM to learn structural reasoning. This is a pure data preprocessing step — high tractability, no additional training runs needed.

### Synthetic Sketch Pipeline for Training Data
- Current: TikZero hybrid (UltraSketch + displacement field, CC 0.82) — **adopted, matches SOTA**
- Scale to full dataset pending storage resolution
- Fine-tuning UltraSketch itself: deprioritized (SketchFig has only 549 human pairs — insufficient to push meaningfully beyond 0.82 without new human data)

---

## M4 — Evaluation ⏳
Not started. Depends on M3.

Planned:
- Compare fine-tuned model vs M2 baseline
- SigLIP embedding-based CC (more accurate than pixel-level)
- Lightweight human evaluation (Likert 1–5, 20–30 examples)

---

## M5 — Demo & Report ⏳
Not started. Depends on M4.

---

## Open Questions for (Supervisor) Meeting

1. **Architecture discussion** — is the proposed combination of DSC-RL (GRPO), localized code masking, execution-centric repair, and domain-specific SigLIP critic the right direction? What should we prioritise first in M3?
2. **FID/KID** — these require image groups, not pairs; promptfoo doesn't handle this natively. Should we run them outside promptfoo as a separate script, or drop them from the benchmark entirely?
3. **Test dataset** — should we create hand-drawn examples ourselves, or is synthetic-only sufficient for the benchmark?
4. **Storage** — still no reply from Quirin. Unblock this?
5. **Timeline** — given we're at week 4 with M2 still in progress, is the current pace sufficient to reach a fine-tuned model before the demo?

---

## Blockers Summary

| Blocker | Impact | Owner |
|---|---|---|
| Test dataset not created (#18) | Benchmark can't run end-to-end | Unassigned |
| Docker permissions on GPU server | Model serving on server untested |
| FID/KID group metric issue | Full metric suite incomplete |
| Storage admin reply pending | Datasets not centralised | Quirin |
| architecture approach discussion | M3 training can't start |

---

## Timeline Estimate

| Week | Target |
|---|---|
| Week 4 (now) | Resolve benchmark blockers, supervisor sign-off on architecture |
| Week 5 | M2 benchmark complete, baseline results ready |
| Week 6–7 | M3 training — SFT with localized masking, first fine-tuning runs |
| Week 8–9 | M4 evaluation, iterate on training |
| Week 10 | M5 demo + report |