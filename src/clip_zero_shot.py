import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
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

LABEL_MAPS = {
    "formal":    {0: "casual", 1: "smart casual", 2: "formal"},
    "season":    {1: "spring", 2: "summer", 3: "fall", 4: "winter"},
    "gender":    {0: "male", 1: "female", 2: "neutral"},
    "time":      {0: "day", 1: "night"},
    "frequency": {0: "occasional", 1: "everyday"},
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="CLIP zero-shot vibe labeling")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a single image or folder. Omit to batch-label the train/ directory.")
    args = parser.parse_args()

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
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        if input_path.is_dir():
            image_paths = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            image_paths = [input_path]
        save_to_file = False
    else:
        image_paths = sorted(p for p in TRAIN_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        save_to_file = True

    print(f"Found {len(image_paths)} image(s)")

    if not image_paths:
        print("No images found.")
        return

    if save_to_file:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = {}

    for batch_start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Labeling", unit="batch"):
        batch_paths = image_paths[batch_start : batch_start + BATCH_SIZE]

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
            raw = {
                "formal":    formal_labels[i],
                "season":    season_labels[i],
                "gender":    gender_labels[i],
                "time":      time_labels[i],
                "frequency": frequency_labels[i],
            }
            results[path.name] = raw

            if not save_to_file:
                vibes_readable = {dim: LABEL_MAPS[dim][raw[dim]] for dim in raw}
                print(f"\n{path.name}")
                for dim, label in vibes_readable.items():
                    print(f"  {dim:10s}: {label}")

    if save_to_file:
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} labels → {OUT_PATH}")


if __name__ == "__main__":
    main()