import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
VIBES_PATH = ROOT / "configs" / "vibes.json"
TRAIN_DIR = ROOT / "train"
OUT_PATH = ROOT / "data" / "labels" / "pseudo_labels.json"

BATCH_SIZE = 32
MODEL_NAME = "openai/clip-vit-large-patch14"


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Prompt encoding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_texts(prompts: list[str], model: CLIPModel, processor: CLIPProcessor, device: torch.device) -> torch.Tensor:
    """Return L2-normalised text embeddings, shape (N, D)."""
    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    feats = model.get_text_features(**inputs)
    if hasattr(feats, 'pooler_output'):
        feats = feats.pooler_output
    return F.normalize(feats, dim=-1)


@torch.no_grad()
def encode_images(images: list[Image.Image], model: CLIPModel, processor: CLIPProcessor, device: torch.device) -> torch.Tensor:
    """Return L2-normalised image embeddings, shape (N, D)."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    if hasattr(feats, 'pooler_output'):
        feats = feats.pooler_output
    return F.normalize(feats, dim=-1)


# ---------------------------------------------------------------------------
# Pre-encode all prompts from vibes.json
# ---------------------------------------------------------------------------

def build_prompt_embeddings(vibes: dict, model: CLIPModel, processor: CLIPProcessor, device: torch.device) -> dict:
    """
    Returns a dict with pre-computed text embeddings for every dimension.

    All dims (formal / season / gender / time / frequency) are classification:
        {
            class_id: Tensor (num_prompts, D),
            ...
        }
    """
    embeddings = {}

    for dim in ("formal", "season", "gender", "time", "frequency"):
        embeddings[dim] = {}
        for class_id, info in vibes[dim]["classes"].items():
            embeddings[dim][class_id] = encode_texts(info["prompts"], model, processor, device)

    return embeddings


# ---------------------------------------------------------------------------
# Per-image scoring
# ---------------------------------------------------------------------------

def score_classification(img_embs: torch.Tensor, class_embs: dict) -> list[int]:
    """
    class_embs: { class_id_str: Tensor(num_prompts, D) }
    Returns argmax class id (as int) for each image.
    """
    class_ids = sorted(class_embs.keys(), key=lambda x: int(x))
    # mean similarity per class → (B, num_classes)
    sims = torch.stack(
        [(img_embs @ class_embs[cid].T).mean(dim=-1) for cid in class_ids],
        dim=-1,
    )
    argmax_indices = sims.argmax(dim=-1).tolist()
    return [int(class_ids[i]) for i in argmax_indices]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    print(f"Using device: {device}")

    # Load config
    with open(VIBES_PATH) as f:
        vibes = json.load(f)

    # Load model
    print(f"Loading {MODEL_NAME} ...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()

    # Pre-encode all prompts
    print("Encoding prompts ...")
    prompt_embs = build_prompt_embeddings(vibes, model, processor, device)

    # Collect image paths
    image_paths = sorted(
        p for p in TRAIN_DIR.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    print(f"Found {len(image_paths)} images in {TRAIN_DIR}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = {}

    for batch_start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Labeling", unit="batch"):
        batch_paths = image_paths[batch_start : batch_start + BATCH_SIZE]

        # Load images, skip corrupt files
        images, valid_paths = [], []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                tqdm.write(f"Skipping {p.name}: {e}")

        if not images:
            continue

        img_embs = encode_images(images, model, processor, device)   # (B, D)

        formal_labels    = score_classification(img_embs, prompt_embs["formal"])
        season_labels    = score_classification(img_embs, prompt_embs["season"])
        gender_labels    = score_classification(img_embs, prompt_embs["gender"])
        time_labels      = score_classification(img_embs, prompt_embs["time"])
        frequency_labels = score_classification(img_embs, prompt_embs["frequency"])

        for i, path in enumerate(valid_paths):
            results[path.name] = {
                "formal":    formal_labels[i],
                "season":    season_labels[i],
                "gender":    gender_labels[i],
                "time":      time_labels[i],
                "frequency": frequency_labels[i],
            }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} labels → {OUT_PATH}")


if __name__ == "__main__":
    main()