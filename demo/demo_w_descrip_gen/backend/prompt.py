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


COMMAND_DESCRIBE = (
    "Act as an expert TikZ designer and image analyst. "
    "You are provided with a user sketch/photo and an optional user description. "
    "Your task is to generate an exhaustive, pixel-perfect visual blueprint specifically tailored for reconstructing the illustration in TikZ/LaTeX. "

    "Carefully analyze the uploaded image. If the user provides additional description, "
    "treat it as ground-truth context and seamlessly integrate it with the visual cues from the image. "

    "**Semantic Requirements:** "
    "1. Enumerate every distinct visual component (entities, nodes, labels, titles, sub-elements). "
    "2. Describe the exact spatial layout and topological relationships: "
    "   - Positions (e.g., top-left, center, bottom-right, absolute relative placement). "
    "   - Hierarchical grouping (e.g., parent-child containers, nested structures). "
    "   - Directional flows, edges, and arrows (source -> target, straight or curved). "
    "   - Text contents verbatim, and their alignment within shapes. "

    "**Formal / Aesthetic Specifications (Critical for TikZ fidelity):** "
    "1. **Background:** Specify the fill color. Default to pure white (`white`) or very light pastel tones (e.g., `gray!5`, `yellow!10`, `blue!5`) unless otherwise indicated. "
    "2. **Color Palette:** Assign explicit TikZ-compatible colors (e.g., `ForestGreen`, `Cerulean`, `Mahogany`, `orange!80`, `red!60!black`) to each major element and its borders. "
    "3. **Line Art & Stroke Attributes:** Define exact stroke properties for all paths, edges, and borders, including thickness (`ultra thick`, `very thick`, `thick`, `thin`, `ultra thin`), style (`solid`, `densely dashed`, `loosely dotted`), and arrowhead types (`stealth`, `latex`, `to`, `triangle 45`). "
    "4. **Node Geometry & Text Styles:** Specify the shape of each node (`rectangle`, `circle`, `ellipse`, `diamond`, `cylinder`, `cloud`). Provide font details: size (`tiny`, `small`, `large`), weight (`bfseries`, `mdseries`), and shape (`itshape`, `scshape`). "
    "5. **Dimensions & Spacing:** Quantify relative distances, padding, and angles (e.g., 'nodes are 3cm apart horizontally', 'arrow bends at 30 degrees', 'inner sep = 2pt'). "

    "**Critical Constraint:** The output must be a self-contained descriptive blueprint so explicit "
    "that an AI TikZ generator can instantly recreate the original drawing with pixel-perfect accuracy "
    "without seeing the source image. "
    "Do NOT output TikZ code itself—only the rich descriptive narrative."
)