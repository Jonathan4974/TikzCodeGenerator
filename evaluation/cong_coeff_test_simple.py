# run this locally on the downloaded outputs
import numpy as np
from PIL import Image

def congruence_coefficient(img1, img2):
    a = np.array(img1.convert("L")).astype(float).flatten()
    b = np.array(img2.convert("L")).astype(float).flatten()
    # resize b to match a if needed
    if len(a) != len(b):
        img2 = img2.resize(img1.size)
        b = np.array(img2.convert("L")).astype(float).flatten()
    numerator = np.sum(a * b)
    denominator = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return numerator / denominator if denominator != 0 else 0.0

output_dir = "../data/ultrasketch_outputs"
for i in range(5):
    real = Image.open(f"{output_dir}/{i}_2_real_sketch.png")
    ultrasketch = Image.open(f"{output_dir}/{i}_3_ultrasketch.png")
    displacement = Image.open(f"{output_dir}/{i}_4_displacement.png")
    combined = Image.open(f"{output_dir}/{i}_5_combined.png")

    print(f"Example {i}:")
    print(f"  UltraSketch CC:   {congruence_coefficient(real, ultrasketch):.3f}")
    print(f"  Displacement CC:  {congruence_coefficient(real, displacement):.3f}")
    print(f"  Combined CC:      {congruence_coefficient(real, combined):.3f}")