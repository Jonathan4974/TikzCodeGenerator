# SketchFig Dataset Audit

## Overview
- Source: nllg/sketchfig (HuggingFace)
- Date audited: 31-05-2026
- Paper: DeTikZify (NeurIPS 2024)

## Size
| Split | Examples |
|---|---|
| train | 549 |
| test | — (none) |
| **Total** | **549** |

## Structure
| Column | Type | Description |
|---|---|---|
| sketch | Image (RGB) | Hand-drawn sketch, ~1012x686px |
| image | Image (RGB) | Corresponding rendered TikZ figure, ~478x510px |
| uri | String | Source URL (TeX StackExchange) |

## Caption / Text Analysis
- No captions present
- No TikZ source code included
- No other text descriptions
- Only pairing information is the source URI

## Key Findings
- **549 hand-drawn examples** — very small dataset
- No TikZ code included → cannot use directly for sketch→code training
- Used for fine-tuning InstructPix2Pix and UltraSketch
- To get TikZ source: would need to scrape via URI or cross-reference with DaTikZv2
- Single train split only — no predefined test set
- Source is exclusively TeX StackExchange figures

## Implications for the Project
- 549 real sketches is insufficient for training alone
- Synthetic sketch generation is confirmed necessary
- DeTikZifyv2 addressed this with UltraSketch — worth reusing
- Need to investigate if DaTikZv2 URIs overlap with SketchFig URIs to recover TikZ code

## Open Questions
- How many SketchFig URIs overlap with DaTikZv2 entries?
- Is UltraSketch (from DeTikZifyv2 repo) available to reuse?
- What's the sketch quality distribution across the 549 examples?