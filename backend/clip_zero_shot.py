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
        "occasion": "Formal evening event",
        "description": "crystalline amber and saffron create a luminous signature that commands every room",
    },
    {
        "name": "Black Orchid",
        "house": "Tom Ford",
        "notes": ["black truffle", "ylang ylang", "dark chocolate", "patchouli"],
        "target_vibes": ["formal", "night", "fall", "everyday"],
        "occasion": "Cocktail or gala",
        "description": "dark florals and truffle weave an intoxicating spell of opulence and mystery",
    },
    {
        "name": "Gypsy Water",
        "house": "Byredo",
        "notes": ["bergamot", "lemon", "pepper", "pine needles"],
        "target_vibes": ["casual", "day", "spring", "summer", "everyday"],
        "occasion": "Casual daytime outing",
        "description": "bergamot and pine needles open into a wanderer's trail — free, airy, and effortlessly cool",
    },
    {
        "name": "Santal 33",
        "house": "Le Labo",
        "notes": ["sandalwood", "cedarwood", "cardamom", "iris"],
        "target_vibes": ["smart casual", "day", "fall", "neutral"],
        "occasion": "Creative office or coffee date",
        "description": "sandalwood and cedarwood ground the senses in warm, unhurried sophistication",
    },
    {
        "name": "Sauvage",
        "house": "Dior",
        "notes": ["bergamot", "pepper", "ambroxan", "vetiver"],
        "target_vibes": ["male", "smart casual", "casual", "everyday", "spring"],
        "occasion": "Everyday signature",
        "description": "ambroxan and fresh bergamot project clean charisma — modern masculinity at its most magnetic",
    },
    {
        "name": "Light Blue",
        "house": "Dolce & Gabbana",
        "notes": ["Sicilian cedar", "apple", "bellflower", "white rose"],
        "target_vibes": ["female", "casual", "day", "summer", "everyday"],
        "occasion": "Summer day out",
        "description": "Sicilian cedar and apple fizz into a breezy Mediterranean escape",
    },
    {
        "name": "Chanel No. 5",
        "house": "Chanel",
        "notes": ["ylang-ylang", "jasmine", "rose", "musk"],
        "target_vibes": ["female", "formal", "night", "winter", "occasional"],
        "occasion": "Black tie or gala",
        "description": "rose and ylang-ylang anchor an immortal feminine ideal — bold, powder-soft, timeless",
    },
    {
        "name": "Acqua di Giò",
        "house": "Giorgio Armani",
        "notes": ["marine accord", "bergamot", "jasmine", "patchouli"],
        "target_vibes": ["male", "casual", "day", "summer", "everyday"],
        "occasion": "Warm-weather casual",
        "description": "marine accord and bergamot breathe like an open sea — effortlessly casual and universally loved",
    },
    {
        "name": "Oud Wood",
        "house": "Tom Ford",
        "notes": ["oud", "sandalwood", "rosewood", "amber"],
        "target_vibes": ["smart casual", "night", "fall", "winter", "occasional"],
        "occasion": "Intimate dinner or evening event",
        "description": "rare oud and sandalwood smolder quietly beneath the surface, leaving a warm resinous trail",
    },
    {
        "name": "Flowerbomb",
        "house": "Viktor & Rolf",
        "notes": ["jasmine", "rose", "orchid", "patchouli"],
        "target_vibes": ["female", "smart casual", "night", "fall", "occasional"],
        "occasion": "Evening drinks or date night",
        "description": "jasmine and rose explode into an unapologetically lush floral bomb — daring and magnetic",
    },
    {
        "name": "Aventus",
        "house": "Creed",
        "notes": ["pineapple", "bergamot", "birch", "musk"],
        "target_vibes": ["male", "formal", "smart casual", "winter", "occasional"],
        "occasion": "Boardroom or formal evening",
        "description": "birch smoke and pineapple burst into a bold chypre that wears success like a second skin",
    },
    {
        "name": "Libre",
        "house": "Yves Saint Laurent",
        "notes": ["mandarin", "lavender", "orange blossom", "vanilla"],
        "target_vibes": ["female", "formal", "smart casual", "night", "fall"],
        "occasion": "Evening event or cocktail hour",
        "description": "mandarin and lavender bridge femininity and edge — rule-breaking sensuality in a bottle",
    },
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

# Maps each context field value to the vibe label used in FRAGRANCE_DB target_vibes
_CONTEXT_VIBE_MAP: dict[str, dict[str, str]] = {
    "eventType": {
        "Gala":        "formal",
        "Date Night":  "formal",
        "Casual":      "casual",
        "Business":    "smart casual",
        "Wedding":     "formal",
        "Festival":    "casual",
    },
    "timeOfDay": {
        "Morning":   "day",
        "Afternoon": "day",
        "Evening":   "night",
        "Night":     "night",
    },
    "mood": {
        "Bold":       "formal",
        "Subtle":     "casual",
        "Fresh":      "spring",
        "Warm":       "fall",
        "Mysterious": "night",
    },
}


def _context_to_vibes(context: dict) -> list[str]:
    """Convert a context dict from the API into a list of vibe label strings."""
    vibes: list[str] = []
    for field, mapping in _CONTEXT_VIBE_MAP.items():
        value = context.get(field)
        if value and value in mapping:
            vibes.append(mapping[value])
    return vibes


def get_recommendations(img: Image.Image, context: dict | None = None) -> list:
    """Runs a single image through CLIP and returns top 3 fragrances.

    Args:
        img: PIL Image of the outfit.
        context: Optional dict with keys eventType, timeOfDay, mood.
                 Each matching vibe adds +1 to a fragrance's match_score.
    """
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

    context_vibes = _context_to_vibes(context) if context else []
    if context_vibes:
        print(f"[ML] Context vibes: {context_vibes}", flush=True)

    scored_fragrances = []
    for frag in FRAGRANCE_DB:
        target = set(frag["target_vibes"])
        match_score = len(set(detected_vibes).intersection(target))
        # Boost score for each context vibe that appears in this fragrance's targets
        match_score += len(set(context_vibes).intersection(target))
        confidence = min(0.95, 0.50 + (match_score * 0.10))

        scored_fragrances.append({
            "name": frag["name"],
            "house": frag["house"],
            "score": round(confidence, 2),
            "notes": frag["notes"],
            "reasoning": (
                f"For a {detected_vibes[1]} {detected_vibes[0]} look, "
                f"{frag['name']} by {frag['house']} earns its place — "
                f"{frag['description']}."
            ),
            "occasion": frag["occasion"],
            "match_score": match_score,
        })

    scored_fragrances.sort(key=lambda x: x["match_score"], reverse=True)
    top_3 = scored_fragrances[:3]

    for i, f in enumerate(top_3):
        f["rank"] = i + 1
        del f["match_score"]

    return top_3