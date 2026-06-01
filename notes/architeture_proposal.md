# Architecture Proposal — Sketch to TikZ with VLM

> Initial proposal based on literature review of SciTikZer, GeoTikZBridge,
> TikZilla, and DeTikZify. To be discussed and refined with supervisor/team.

---

## Overview

Rather than adopting any single prior model, we propose combining the
strongest ideas from recent work into a unified architecture that is:
- **Visually faithful** — via TikZilla's scale and RL training
- **Structurally logical** — via GeoTikZBridge's localized masking
- **Self-verifiable** — via SciTikZer's dual consistency

---

## Component 1 — Core Architecture: Dual Self-Consistency (DSC)
*Source: SciTikZer*

### Idea
Adopt the Round-Trip Verification paradigm as the foundational training
objective. The model generates TikZ code, renders it to an image, and
must then reconstruct the original code back from that rendered image.

### Why
Standard models are "structurally blind" — they can produce code that
looks visually similar but is logically unsound. DSC forces the model
to be self-consistent: generated code must be not just visually lucky
but programmatically and logically correct.

---

## Component 2 — Training Strategy: Localized Geometric Transformation
*Source: GeoTikZBridge*

### Idea
During the SFT phase, randomly remove or mask up to 40% of TikZ code
lines. The model must fill in the blanks based on visual input alone.

### Why
Forces the model to learn fine-grained structural semantics and
geometric dependencies rather than copying text sequences. Shown to
reduce code repetition by 15% in GeoTikZBridge experiments.

---

## Component 3 — Data Engine: Execution-Centric Repair
*Source: TikZilla + SciTikZer*

### Idea
Use a "Repair Agent" (a larger MLLM) to fix uncompilable TikZ code
snippets using compiler error messages as feedback, rather than
discarding them.

### Why
TikZilla used this approach to recover 600k previously unusable arXiv
instances. Ensures the training set is strictly compilable and
high-fidelity — the model only ever learns from correct examples.

---

## Component 4 — Reward Mechanism: Domain-Specific Vision Critic
*Source: TikZilla*

### Idea
For the RL stage, avoid general metrics like CLIPScore. Instead,
retrain a vision encoder (SigLIP) specifically on an inverse-graphics
task (Image → TikZ) first, then use it as a domain-specific critic.
Use Earth Mover's Distance (EMD) on image patch embeddings rather
than cosine similarity.

### Why
General vision models miss subtle scientific details — segment
relations, axis labels, structural topology. A retrained encoder
produces richer, more specialized embeddings for scientific diagrams,
giving the RL reward signal more signal fidelity.

---

## Summary Table

| Component | Approach | Source |
|---|---|---|
| Learning Paradigm | DSC-RL (GRPO): visual alignment + symbolic back-translation | SciTikZer |
| SFT Strategy | Localized code masking: structural logic over text copying | GeoTikZBridge |
| Inference | MCTS with compiler feedback: iterative refinement, no extra training | DeTikZify |
| Data Engine | VLM-generated descriptions + execution-centric repair | TikZilla + SciTikZer |
| RL Reward | Domain-specific SigLIP critic + EMD on patch embeddings | TikZilla |

---

## Open Questions for Supervisor/team
- Is DSC-RL (GRPO) feasible within our compute budget?
- Should we start from DeTikZifyv2 weights or a general pretrained VLM?
- Is 40% masking rate the right starting point for our dataset size?
- Priority order: which component to implement first if time is limited?

---

## Status
- Draft — not yet discussed with supervisor/team
- To be presented at next supervisor/team meeting