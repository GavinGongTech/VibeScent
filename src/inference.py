import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.cnn_baseline import CNNBaseline
from models.clip_standalone import CLIPStandalone
from models.cnn_clip_hybrid import CNNCLIPHybrid

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

LABEL_MAPS = {
    "formal":    {0: "casual", 1: "smart casual", 2: "formal"},
    "season":    {1: "spring", 2: "summer", 3: "fall", 4: "winter"},
    "gender":    {0: "male", 1: "female", 2: "neutral"},
    "time":      {0: "day", 1: "night"},
    "frequency": {0: "occasional", 1: "everyday"},
}

VIBES_DIR      = ROOT / "outputs" / "vibes"
EMBEDDINGS_DIR = ROOT / "outputs" / "embeddings"


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
# Model
# ---------------------------------------------------------------------------

def load_model(model_name: str, ckpt_path: Path, device: torch.device):
    if model_name == "cnn":
        model = CNNBaseline()
    elif model_name == "clip":
        model = CLIPStandalone()
    else:
        model = CNNCLIPHybrid()

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def preprocess(image_path: Path) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    return transform(Image.open(image_path).convert("RGB")).unsqueeze(0)   # (1, 3, 224, 224)


@torch.no_grad()
def run_single(model, x: torch.Tensor, device: torch.device) -> tuple[dict, np.ndarray]:
    x = x.to(device)
    output    = model(x)
    embedding = model.get_embedding(x).squeeze(0).cpu().numpy()   # (512,)

    vibes = {}
    for dim in LABEL_MAPS:
        idx = output[dim].argmax(dim=1).item()
        if dim == "season":
            idx += 1   # model outputs 0-3; season labels are 1-indexed (data_loader subtracts 1 at train time)
        vibes[dim] = LABEL_MAPS[dim][idx]
    return vibes, embedding


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vibe inference — single image or folder")
    parser.add_argument("--model",      choices=["cnn", "clip", "hybrid"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best.pt checkpoint")
    parser.add_argument("--input",      type=str, required=True,
                        help="Path to a single image or a folder of images")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ckpt_path  = Path(args.checkpoint)
    input_path = Path(args.input)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Collect images
    if input_path.is_dir():
        image_paths = sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        image_paths = [input_path]

    if not image_paths:
        print("No images found.")
        return

    # Output dirs
    VIBES_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} from {ckpt_path} ...")
    model = load_model(args.model, ckpt_path, device)
    print(f"Processing {len(image_paths)} image(s) ...\n")

    processed, failed = 0, 0

    for image_path in image_paths:
        try:
            x = preprocess(image_path)
            vibes, embedding = run_single(model, x, device)

            stem = image_path.stem
            json_path = VIBES_DIR      / f"{stem}.json"
            npy_path  = EMBEDDINGS_DIR / f"{stem}.npy"

            with open(json_path, "w") as f:
                json.dump(vibes, f, indent=2)
            np.save(npy_path, embedding)

            processed += 1

        except Exception as e:
            print(f"  [skip] {image_path.name}: {e}")
            failed += 1

    print(f"\nDone. {processed} processed, {failed} failed.")
    print(f"  Vibe JSONs   → {VIBES_DIR}")
    print(f"  Embeddings   → {EMBEDDINGS_DIR}")


if __name__ == "__main__":
    main()
