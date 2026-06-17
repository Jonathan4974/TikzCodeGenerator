from functools import lru_cache
from pathlib import Path
import os

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_NAMES = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
}

HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "/root/.cache/huggingface")

HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


@lru_cache(maxsize=2)
def load_model(model_key: str):
    if model_key not in MODEL_NAMES:
        raise ValueError(f"Unknown model_key: {model_key}")

    model_name = MODEL_NAMES[model_key]

    processor = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir=HF_CACHE_DIR,
        local_files_only=HF_LOCAL_FILES_ONLY,
    )

    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=HF_CACHE_DIR,
        local_files_only=HF_LOCAL_FILES_ONLY,
    )

    model.eval()

    return processor, model


def extract_tensor_features(model_output) -> torch.Tensor:
    if torch.is_tensor(model_output):
        return model_output

    if hasattr(model_output, "image_embeds") and model_output.image_embeds is not None:
        return model_output.image_embeds

    if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
        return model_output.pooler_output

    if hasattr(model_output, "last_hidden_state") and model_output.last_hidden_state is not None:
        return model_output.last_hidden_state.mean(dim=1)

    if isinstance(model_output, (tuple, list)):
        for item in model_output:
            if torch.is_tensor(item):
                return item

    raise RuntimeError(f"Could not extract tensor features from: {type(model_output)}")


def image_embedding(image_path: str | Path, model_key: str) -> torch.Tensor:
    processor, model = load_model(model_key)

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            output = model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
        else:
            output = model(
                pixel_values=inputs["pixel_values"]
            )

        features = extract_tensor_features(output)

    features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze(0)


def image_cosine_similarity(
    image_a: str | Path,
    image_b: str | Path,
    model_key: str,
) -> float:
    emb_a = image_embedding(image_a, model_key)
    emb_b = image_embedding(image_b, model_key)

    return float(torch.dot(emb_a, emb_b))