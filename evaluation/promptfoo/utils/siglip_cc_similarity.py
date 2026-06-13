from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

from .semantic_similarity import load_model


@lru_cache(maxsize=2)
def _load_siglip():
    processor, model = load_model("siglip")
    model.eval()
    return processor, model


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_vision_model(model):
    if hasattr(model, "vision_model"):
        return model.vision_model

    if hasattr(model, "model") and hasattr(model.model, "vision_model"):
        return model.model.vision_model

    return model


def embed_image(image_path: str | Path) -> np.ndarray:
    processor, model = _load_siglip()
    device = _get_model_device(model)

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    pixel_values = inputs["pixel_values"].to(device)

    vision_model = _get_vision_model(model)

    with torch.no_grad():
        output = vision_model(pixel_values=pixel_values)

    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        # shape: [1, seq_len, dim]
        features = output.last_hidden_state.squeeze(0)

    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        # fallback: shape [1, dim] -> one vector
        features = output.pooler_output.squeeze(0).unsqueeze(0)

    else:
        raise RuntimeError(
            "Could not extract patch embeddings from SigLIP vision model output"
        )

    features = features.detach().float().cpu().numpy()

    if features.ndim == 1:
        features = features.reshape(1, -1)

    if features.ndim != 2:
        raise RuntimeError(f"Unexpected SigLIP feature shape: {features.shape}")

    return features


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


def compute_siglip_cc(image_a: str | Path, image_b: str | Path) -> float:
    """
    Compute SigLIP patch-embedding Congruence Coefficient.

    Steps:
    - extract per-patch embeddings for both images
    - compute local difference vectors
    - fit PCA(n_components=1) on differences
    - project both patch-embedding sets onto first PC
    - compute cosine similarity between projected patch vectors

    Returns:
        float in approximately [-1, 1]
    """
    emb_a = embed_image(image_a)
    emb_b = embed_image(image_b)

    if emb_a.shape != emb_b.shape:
        raise ValueError(
            f"Patch embedding shapes do not match: {emb_a.shape} vs {emb_b.shape}"
        )

    if emb_a.shape[0] < 2:
        vec_a = emb_a.mean(axis=0)
        vec_b = emb_b.mean(axis=0)
        return _cosine(vec_a, vec_b)

    diff = emb_a - emb_b

    pca = PCA(n_components=1)
    pca.fit(diff)

    pc = pca.components_[0]

    proj_a = emb_a @ pc
    proj_b = emb_b @ pc

    cc = _cosine(proj_a, proj_b)

    # numerical safety
    cc = max(-1.0, min(1.0, cc))

    return cc


def siglip_cc_to_score(cc: float) -> float:
    """
    Map congruence coefficient from [-1, 1] to [0, 1].
    """
    score = (float(cc) + 1.0) / 2.0
    return max(0.0, min(1.0, score))