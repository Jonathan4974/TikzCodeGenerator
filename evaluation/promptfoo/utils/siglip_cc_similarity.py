from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

from .semantic_similarity import load_model

@lru_cache(maxsize=2)
def _load_siglip():
    proc, model = load_model("siglip")
    return proc, model


def embed_image(image_path: str | Path):
    """Return per-patch embeddings for an image using the SigLIP model.

    Output: numpy array shape (n_patches, dim)
    """
    processor, model = _load_siglip()

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model(pixel_values=inputs["pixel_values"])

    # Prefer last_hidden_state (per-patch). Fallbacks are discouraged for CC.
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        features = output.last_hidden_state.squeeze(0)
    elif hasattr(output, "image_embeds") and output.image_embeds is not None:
        # fallback: single-vector embed -> return as single patch
        features = output.image_embeds
    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        features = output.pooler_output
    else:
        raise RuntimeError("Could not extract patch embeddings from SigLIP model output")

    features = features.detach().cpu().numpy()

    # ensure shape (n_patches, dim)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    elif features.ndim == 2:
        pass
    elif features.ndim == 3:
        # (seq_len, dim) expected; if shape (1, seq_len, dim) squeeze
        features = features.reshape(features.shape[0] * features.shape[1], -1)

    return features


def compute_siglip_cc(image_a: str | Path, image_b: str | Path) -> float:
    """Compute SigLIP embedding-based Congruence Coefficient (CC) for a pair.

    Steps:
    - extract per-patch embeddings for both images
    - compute local difference vectors (A - B)
    - fit PCA(n_components=1) on the difference vectors
    - project both images' patch embeddings onto the first PC
    - compute cosine similarity (congruence coefficient) between the projected vectors
    Returns a float in [-1, 1].
    """
    emb_a = embed_image(image_a)
    emb_b = embed_image(image_b)

    if emb_a.shape != emb_b.shape:
        # try to handle simple mismatches by resizing via processor's default size
        raise ValueError(f"Patch embedding shapes do not match: {emb_a.shape} vs {emb_b.shape}")

    diff = emb_a - emb_b  # shape (n_patches, dim)

    if diff.shape[0] < 2:
        # Not enough patches for PCA; fall back to cosine of mean embeddings
        vec_a = emb_a.mean(axis=0)
        vec_b = emb_b.mean(axis=0)
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        return float(np.dot(vec_a, vec_b) / denom) if denom != 0 else 0.0

    pca = PCA(n_components=1)
    pca.fit(diff)
    pc = pca.components_[0]  # shape (dim,)

    proj_a = emb_a.dot(pc)  # shape (n_patches,)
    proj_b = emb_b.dot(pc)

    denom = np.linalg.norm(proj_a) * np.linalg.norm(proj_b)
    cc = float(np.dot(proj_a, proj_b) / denom) if denom != 0 else 0.0

    return cc


def siglip_cc_to_score(cc: float) -> float:
    """Map congruence coefficient in [-1,1] to a 0-1 similarity score."""
    score = (float(cc) + 1.0) / 2.0
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0
    return score
