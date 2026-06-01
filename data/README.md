# Data

This folder contains everything related to datasets — audits, 
preprocessing scripts, and synthetic data generation pipelines.

## What goes here
- Dataset audit markdown files (e.g. `sketchfig_audit.md`)
- Preprocessing and cleaning scripts
- Synthetic sketch generation scripts (e.g. `ultrasketch_test.py`)

## What does NOT go here
- Raw dataset files — these are loaded via HuggingFace Hub or 
  stored on the GPU server at `/usr/prakt/<user>/data/`
- Model weights

## Key datasets
| Dataset | Source | Size | Notes |
|---|---|---|---|
| SketchFig | nllg/sketchfig | 549 examples | Hand-drawn sketches, no TikZ code |
| DaTikZv4 | nllg/datikz-v4 | 450k+ | arXiv portion requires retrieval scripts |
| GeoTikZ | sjy-1995/GeoTikZBridge | — | Geometry-focused |
| SciTikZ-230K | — | 230k | — |