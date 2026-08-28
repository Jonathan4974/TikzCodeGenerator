# COMMAND_TIKZ = """As a LaTeX graphics expert, translate the image into TikZ code suitable for academic publications.
# Focus on recreating geometric precision, typographic elements, and color schemes.
# The code must be compilable, efficient, and maintain the original image's visual fidelity for professional document integration.
# Return only the full LaTeX document.
# Do not use markdown fences.
# Do not add explanations."""

COMMAND_TIKZ = """As a LaTeX/TikZ expert, translate the provided image into a complete, compilable LaTeX document.

**Primary Reference:**
- The UPLOADED IMAGE is your absolute primary source of truth for layout, geometry, colors, and elements.
- The "User Description" provided below is entirely OPTIONAL and serves ONLY as an auxiliary hint. Use it only to resolve ambiguous details (e.g., blurry regions, occluded text, or uncertain color shades) or to incorporate specific user requests. If the description contradicts the image, ALWAYS defer to the image.

**Implementation Requirements:**
1. Accurately recreate all visible shapes, text labels, lines, arrows, and color schemes directly from the image.
2. Use `\\documentclass[tikz,border=2pt]{standalone}` or `article` with `\\usepackage{tikz}` and necessary libraries (e.g., `\\usetikzlibrary{positioning, shapes, arrows.meta, calc, fit}`).
3. Ensure strict compilability—do not omit required packages or introduce unsupported syntax.
4. Maintain professional academic publication quality: crisp strokes, appropriate typography, and balanced whitespace.

**Output Format Constraints:**
- Return ONLY the full LaTeX document text.
- Do NOT wrap the code in markdown fences (```).
- Do NOT add any explanations, annotations, or apologies before/after the code."""