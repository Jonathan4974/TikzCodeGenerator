from datasets import load_dataset
from diffusers import DiffusionPipeline
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
import torch
from pathlib import Path
import sys

NUM_EXAMPLES = 5
BASE_OUTPUT_DIR = Path("/usr/prakt/s0031/ultrasketch_outputs")

MULTIPLE_OF = 16
DISPLACEMENT_ALPHA = 6
DISPLACEMENT_SIGMA = 12
APPLY_DISPLACEMENT = True
COMBINED_BLEND_ALPHA = 0.5

INFERENCE_PARAMS = dict(
    num_inference_steps=50,
    image_guidance_scale=1.7,
    guidance_scale=1.5,
    strength=0.9,
)

PROMPT_VARIANTS = {
    "baseline": "Turn it into a hand-drawn sketch",

    "pencil": "Convert this diagram into a rough hand-drawn pencil sketch as if drawn by a student",

    "scientific": "Redraw this as a hand-drawn scientific figure with pencil, showing natural line variation and imperfection",

    "minimal": "Sketch this diagram by hand, keeping only essential lines",

    "structure_preserve": "Turn it into a hand-drawn sketch, preserving all geometric shapes, lines, arrows, nodes, and text labels exactly as they appear",
    
    "scientific_labels": "Redraw this scientific TikZ diagram as a hand-drawn pencil sketch while keeping every label, symbol, arrow, and annotation legible",
    
    "light_pencil": "Make a light pencil sketch of this diagram on white paper, preserving the exact layout and readable text",
    
    "anti_artifact": "Convert this figure to a clean hand-drawn pencil sketch. Preserve text, geometry, arrows, and fine details. Avoid blur, smearing, shadows, and distorted labels",
    
    "student_notes": "Transform this TikZ figure into a hand-drawn sketch as if drawn by a student in lecture notes, keeping shapes, arrows, nodes, and labels intact",
}

def pixel_congruence_coefficient(img1, img2):
    a = np.array(img1.convert("L")).astype(float).flatten()
    b = np.array(img2.convert("L")).astype(float).flatten()
    numerator = np.sum(a * b)
    denominator = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return numerator / denominator if denominator != 0 else 0.0

def try_import_siglip_cc():
    promptfoo_dir = Path(__file__).resolve().parents[1] / "evaluation" / "promptfoo"
    if promptfoo_dir.exists():
        sys.path.insert(0, str(promptfoo_dir))

    try:
        from siglip_cc_similarity import compute_siglip_cc, embed_image

        return compute_siglip_cc, embed_image
    except ImportError as exc:
        print(f"Could not import SigLIP CC utility: {exc}")
        return None, None

def assert_multiple_of(image, label, multiple=MULTIPLE_OF):
    width, height = image.size
    if width % multiple != 0 or height % multiple != 0:
        raise ValueError(
            f"{label} has size {image.size}, expected both dimensions to be "
            f"multiples of {multiple}"
        )

def resize_to_multiple(image, multiple=MULTIPLE_OF):
    width, height = image.size
    new_width = max(multiple, (width // multiple) * multiple)
    new_height = max(multiple, (height // multiple) * multiple)
    if (new_width, new_height) == image.size:
        return image
    return image.resize((new_width, new_height), Image.LANCZOS)

def resize_like(image, reference):
    if image.size == reference.size:
        return image
    return image.resize(reference.size, Image.LANCZOS)

def make_dirs(base, variants):
    (base / "ground_truth").mkdir(parents=True, exist_ok=True)
    for name in variants:
        (base / f"ultrasketch_{name}").mkdir(parents=True, exist_ok=True)

def random_displacement_field(image, alpha=6, sigma=12, seed=42):
    assert_multiple_of(image, "displacement input", MULTIPLE_OF)
    rng = np.random.default_rng(seed)
    img_array = np.array(image.convert("RGB")).astype(np.float32)
    shape = img_array.shape[:2]

    dx = gaussian_filter((rng.random(shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((rng.random(shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (
        np.clip(y + dy, 0, shape[0] - 1).ravel(),
        np.clip(x + dx, 0, shape[1] - 1).ravel(),
    )

    distorted = np.stack(
        [
            map_coordinates(img_array[:, :, channel], indices, order=1).reshape(shape)
            for channel in range(img_array.shape[2])
        ],
        axis=2,
    )

    return Image.fromarray(np.clip(distorted, 0, 255).astype(np.uint8))

def run_ultrasketch(pipe, image, prompt):
    image = resize_to_multiple(image).convert("RGB")
    assert_multiple_of(image, "UltraSketch input", MULTIPLE_OF)
    output = pipe(
        prompt=prompt,
        image=image,
        mask_img=Image.new("RGB", image.size, "white"),
        **INFERENCE_PARAMS,
    ).images[0]
    output = resize_like(output.convert("RGB"), image)
    assert_multiple_of(output, "UltraSketch output", MULTIPLE_OF)
    return output

def image_blend(image_a, image_b, alpha=COMBINED_BLEND_ALPHA):
    image_b = resize_like(image_b.convert("RGB"), image_a)
    return Image.blend(image_a.convert("RGB"), image_b, alpha)

def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))

def print_summary(title, results_by_variant, file=None):
    print(f"\n### {title}\n", file=file)
    print(f"| {'Variant':<22} | {'Mean CC':>7} | {'Std CC':>6} |", file=file)
    print(f"|:{'-' * 22}-|{'-' * 8}:|{'-' * 7}:|", file=file)
    for variant_name, values in results_by_variant.items():
        mean, std = mean_std(values)
        print(f"| {variant_name:<22} | {mean:7.3f} | {std:6.3f} |", file=file)
    print("", file=file)

def print_single_summary(title, name, values, file=None):
    mean, std = mean_std(values)
    print(
        f"\n### {title}\n\n"
        f"| {'Variant':<22} | {'Mean CC':>7} | {'Std CC':>6} |\n"
        f"|:{'-' * 22}-|{'-' * 8}:|{'-' * 7}:|\n"
        f"| {name:<22} | {mean:7.3f} | {std:6.3f} |\n",
        file=file
    )

def evaluate_pair(output_path, target_path, siglip_cc_fn=None):
    output = Image.open(output_path).convert("RGB")
    target = resize_like(Image.open(target_path).convert("RGB"), output)

    pixel_cc = pixel_congruence_coefficient(output, target)
    siglip_cc = None
    if siglip_cc_fn is not None:
        siglip_cc = siglip_cc_fn(output_path, target_path)

    return pixel_cc, siglip_cc

def cosine_similarity(vector_a, vector_b):
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    return float(np.dot(vector_a, vector_b) / denominator) if denominator != 0 else 0.0

def siglip_embedding_hybrid_cc(ultrasketch_path, displacement_path, target_path, embed_image_fn):
    from sklearn.decomposition import PCA

    ultrasketch_embedding = embed_image_fn(ultrasketch_path)
    displacement_embedding = embed_image_fn(displacement_path)
    target_embedding = embed_image_fn(target_path)

    hybrid_embedding = (ultrasketch_embedding + displacement_embedding) / 2.0

    if hybrid_embedding.shape != target_embedding.shape:
        raise ValueError(
            "Patch embedding shapes do not match: "
            f"{hybrid_embedding.shape} vs {target_embedding.shape}"
        )

    if hybrid_embedding.shape[0] < 2:
        return cosine_similarity(
            hybrid_embedding.mean(axis=0),
            target_embedding.mean(axis=0),
        )

    diff = hybrid_embedding - target_embedding
    pca = PCA(n_components=1)
    pca.fit(diff)

    pc = pca.components_[0]
    cc = cosine_similarity(hybrid_embedding @ pc, target_embedding @ pc)
    return max(-1.0, min(1.0, cc))

def main():
    dataset = load_dataset("nllg/sketchfig", split="train")

    make_dirs(BASE_OUTPUT_DIR, PROMPT_VARIANTS.keys())

    pipe = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path="nllg/ultrasketch",
        custom_pipeline="nllg/ultrasketch",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda:0")

    pixel_cc_ultrasketch = {name: [] for name in PROMPT_VARIANTS}
    pixel_cc_displacement = []
    pixel_cc_combined = {name: [] for name in PROMPT_VARIANTS}
    siglip_cc_ultrasketch = {name: [] for name in PROMPT_VARIANTS}
    siglip_cc_displacement = []
    siglip_cc_hybrid = {name: [] for name in PROMPT_VARIANTS}

    siglip_cc_fn, siglip_embed_fn = try_import_siglip_cc()
    if siglip_cc_fn is None:
        print("SigLIP CC utility is not importable; skipping SigLIP summary.")

    for i in range(NUM_EXAMPLES):
        figure = resize_to_multiple(dataset[i]["image"]).convert("RGB")
        real_sketch = resize_like(dataset[i]["sketch"].convert("RGB"), figure)

        gt_dir = BASE_OUTPUT_DIR / "ground_truth"
        rendered_path = gt_dir / f"{i}_rendered.png"
        real_sketch_path = gt_dir / f"{i}_real_sketch.png"
        figure.save(rendered_path)
        real_sketch.save(real_sketch_path)

        displacement_out = None
        if APPLY_DISPLACEMENT:
            displacement_out = random_displacement_field(
                figure,
                alpha=DISPLACEMENT_ALPHA,
                sigma=DISPLACEMENT_SIGMA,
                seed=i,
            )
            displacement_reference_path = gt_dir / f"{i}_displacement.png"
            displacement_out.save(displacement_reference_path)
            displacement_pixel_cc, displacement_siglip_cc = evaluate_pair(
                displacement_reference_path,
                real_sketch_path,
                siglip_cc_fn=siglip_cc_fn,
            )
            pixel_cc_displacement.append(displacement_pixel_cc)
            if displacement_siglip_cc is not None:
                siglip_cc_displacement.append(displacement_siglip_cc)

        print(f"\nExample {i} (size {figure.size})")

        for variant_name, prompt in PROMPT_VARIANTS.items():
            print(f"  Running {variant_name}...")
            variant_dir = BASE_OUTPUT_DIR / f"ultrasketch_{variant_name}"

            ultrasketch_out = run_ultrasketch(pipe, figure, prompt)
            ultrasketch_path = variant_dir / f"{i}_ultrasketch.png"
            ultrasketch_out.save(ultrasketch_path)

            pixel_cc, siglip_cc = evaluate_pair(
                ultrasketch_path,
                real_sketch_path,
                siglip_cc_fn=siglip_cc_fn,
            )
            pixel_cc_ultrasketch[variant_name].append(pixel_cc)
            if siglip_cc is not None:
                siglip_cc_ultrasketch[variant_name].append(siglip_cc)

            if APPLY_DISPLACEMENT:
                displacement_path = variant_dir / f"{i}_displacement.png"
                combined_path = variant_dir / f"{i}_visual_blend.png"

                displacement_out.save(displacement_path)
                combined_out = image_blend(ultrasketch_out, displacement_out)
                combined_out.save(combined_path)

                pixel_cc, siglip_cc = evaluate_pair(
                    combined_path,
                    real_sketch_path,
                )
                pixel_cc_combined[variant_name].append(pixel_cc)
                if siglip_embed_fn is not None:
                    siglip_hybrid_cc = siglip_embedding_hybrid_cc(
                        ultrasketch_path,
                        displacement_path,
                        real_sketch_path,
                        siglip_embed_fn,
                    )
                    siglip_cc_hybrid[variant_name].append(siglip_hybrid_cc)

            print(f"  Saved -> {variant_dir}")

    md_file_path = BASE_OUTPUT_DIR / "summary.md"
    print(f"\nWriting summary evaluation metrics to: {md_file_path}")
    
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write("# UltraSketch Evaluation Summary\n")

        print_summary("Pixel-level CC (UltraSketch output)", pixel_cc_ultrasketch, file=f)
        if APPLY_DISPLACEMENT:
            print_single_summary(
                "Pixel-level CC (displacement-only baseline)",
                "displacement",
                pixel_cc_displacement,
                file=f
            )
            print_summary(
                "Pixel-level CC (saved UltraSketch + displacement visual blend)",
                pixel_cc_combined,
                file=f
            )

        if siglip_cc_fn is not None:
            print_summary("SigLIP embedding CC (UltraSketch output)", siglip_cc_ultrasketch, file=f)
            if APPLY_DISPLACEMENT:
                print_single_summary(
                    "SigLIP embedding CC (displacement-only baseline)",
                    "displacement",
                    siglip_cc_displacement,
                    file=f
                )
                print_summary(
                    "SigLIP embedding CC (UltraSketch + displacement embedding average)",
                    siglip_cc_hybrid,
                    file=f
                )
        else:
            f.write("\n## SigLIP Metrics\n")
            f.write("> *SigLIP CC utility was not importable during runtime; embedding summaries were skipped.*\n")

    print("\nAll done.")

if __name__ == "__main__":
    main()
