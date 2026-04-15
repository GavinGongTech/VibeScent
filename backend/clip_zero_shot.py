import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
VIBES_PATH = ROOT / "backend" / "configs" / "vibes.json"
MODEL_NAME = "openai/clip-vit-large-patch14"

LABEL_MAPS = {
    "formal":    {0: "casual", 1: "smart casual", 2: "formal"},
    "season":    {1: "spring", 2: "summer", 3: "fall", 4: "winter"},
    "gender":    {0: "male", 1: "female", 2: "neutral"},
    "time":      {0: "day", 1: "night"},
    "frequency": {0: "occasional", 1: "everyday"},
}

FRAGRANCE_DB = [
    {
        "name": "Baccarat Rouge 540",
        "house": "Maison Francis Kurkdjian",
        "notes": ["jasmine", "saffron", "amberwood", "fir resin"],
        "target_vibes": ["formal", "night", "winter", "smart casual"],
        "occasion": "Formal evening event"
    },
    {
        "name": "Black Orchid",
        "house": "Tom Ford",
        "notes": ["black truffle", "ylang ylang", "dark chocolate", "patchouli"],
        "target_vibes": ["formal", "night", "fall", "everyday"],
        "occasion": "Cocktail or gala"
    },
    {
        "name": "Gypsy Water",
        "house": "Byredo",
        "notes": ["bergamot", "lemon", "pepper", "pine needles"],
        "target_vibes": ["casual", "day", "spring", "summer", "everyday"],
        "occasion": "Casual daytime outing"
    },
    {
        "name": "Santal 33",
        "house": "Le Labo",
        "notes": ["sandalwood", "cedarwood", "cardamom", "iris"],
        "target_vibes": ["smart casual", "day", "fall", "neutral"],
        "occasion": "Creative office or coffee date"
    }
]

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
device = None
model = None
processor = None
prompt_embs = None

# ---------------------------------------------------------------------------
# Helper Functions 
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

@torch.no_grad()
def encode_texts(prompts: list[str], model, processor, device) -> torch.Tensor:
    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    feats = model.get_text_features(**inputs)
    if hasattr(feats, 'pooler_output'): feats = feats.pooler_output
    return F.normalize(feats, dim=-1)

@torch.no_grad()
def encode_images(images: list[Image.Image], model, processor, device) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    if hasattr(feats, 'pooler_output'): feats = feats.pooler_output
    return F.normalize(feats, dim=-1)

def build_prompt_embeddings(vibes: dict, model, processor, device) -> dict:
    embeddings = {}
    for dim in ("formal", "season", "gender", "time", "frequency"):
        embeddings[dim] = {}
        for class_id, info in vibes[dim]["classes"].items():
            embeddings[dim][class_id] = encode_texts(info["prompts"], model, processor, device)
    return embeddings

def score_classification(img_embs: torch.Tensor, class_embs: dict) -> list[int]:
    class_ids = sorted(class_embs.keys(), key=lambda x: int(x))
    sims = torch.stack([(img_embs @ class_embs[cid].T).mean(dim=-1) for cid in class_ids], dim=-1)
    argmax_indices = sims.argmax(dim=-1).tolist()
    return [int(class_ids[i]) for i in argmax_indices]

# ---------------------------------------------------------------------------
# Core Exported Functions
# ---------------------------------------------------------------------------
def initialize_model():
    """Loads the model, processor, and prompt embeddings into memory."""
    global device, model, processor, prompt_embs
    
    device = get_device()
    print(f"[ML] Hardware target: {device}", flush=True)
    
    with open(VIBES_PATH) as f:
        vibes = json.load(f)

    print(f"[ML] Loading {MODEL_NAME}...", flush=True)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()

    print("[ML] Pre-encoding text prompts...", flush=True)
    prompt_embs = build_prompt_embeddings(vibes, model, processor, device)
    print("[ML] Initialization complete!", flush=True)

def get_recommendations(img: Image.Image) -> list:
    """Runs a single image through CLIP and returns top 3 fragrances."""
    print("[ML] Extracting visual features...", flush=True)
    img_embs = encode_images([img], model, processor, device)
    
    raw_labels = {
        "formal":    score_classification(img_embs, prompt_embs["formal"])[0],
        "season":    score_classification(img_embs, prompt_embs["season"])[0],
        "gender":    score_classification(img_embs, prompt_embs["gender"])[0],
        "time":      score_classification(img_embs, prompt_embs["time"])[0],
        "frequency": score_classification(img_embs, prompt_embs["frequency"])[0],
    }
    
    detected_vibes = [LABEL_MAPS[dim][raw_labels[dim]] for dim in raw_labels]
    print(f"[ML] Detected vibes: {detected_vibes}", flush=True)

    scored_fragrances = []
    for frag in FRAGRANCE_DB:
        match_score = len(set(detected_vibes).intersection(set(frag["target_vibes"])))
        confidence = min(0.95, 0.50 + (match_score * 0.10)) 
        
        scored_fragrances.append({
            "name": frag["name"],
            "house": frag["house"],
            "score": round(confidence, 2),
            "notes": frag["notes"],
            "reasoning": f"Our vision model detected a {detected_vibes[0]} and {detected_vibes[1]} aesthetic. The notes of {frag['notes'][0]} perfectly complement this energy.",
            "occasion": frag["occasion"],
            "match_score": match_score
        })

    scored_fragrances.sort(key=lambda x: x["match_score"], reverse=True)
    top_3 = scored_fragrances[:3]

    for i, f in enumerate(top_3):
        f["rank"] = i + 1
        del f["match_score"]

    return top_3