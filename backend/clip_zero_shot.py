import json
from pathlib import Path
import base64
from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
VIBES_PATH = Path(__file__).resolve().parent / "configs" / "vibes.json"
MODEL_NAME = "openai/clip-vit-large-patch14"

LABEL_MAPS = {
    "formal":    {0: "casual", 1: "smart casual", 2: "formal"},
    "season":    {1: "spring", 2: "summer", 3: "fall", 4: "winter"},
    "gender":    {0: "male", 1: "female", 2: "neutral"},
    "time":      {0: "day", 1: "night"},
    "frequency": {0: "occasional", 1: "everyday"},
}

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

def extract_vibe_dictionary(base64_string: str, user_text: str) -> tuple[dict, str]:
    """
    Decodes the Next.js image, runs it through the pre-loaded CLIP model, 
    and formats the detected labels into the exact dictionary schema 
    expected by the semantic recommender.
    """
    # 1. Decode the base64 image from Next.js
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    image_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(image_data)).convert("RGB")

    print("[ML] Extracting visual features...", flush=True)
    img_embs = encode_images([img], model, processor, device)

    # 2. Get raw classification IDs
    raw_labels = {
        "formal":    score_classification(img_embs, prompt_embs["formal"])[0],
        "season":    score_classification(img_embs, prompt_embs["season"])[0],
        "gender":    score_classification(img_embs, prompt_embs["gender"])[0],
        "time":      score_classification(img_embs, prompt_embs["time"])[0],
        "frequency": score_classification(img_embs, prompt_embs["frequency"])[0],
    }

    # 3. Map IDs to string labels
    detected_vibes = {dim: LABEL_MAPS[dim][raw_labels[dim]] for dim in raw_labels}
    print(f"[ML] Detected vibes: {detected_vibes}", flush=True)

    # 4. Map string labels to the exact format your Pandas dataframe expects
    formality_map = {"casual": 0.3, "smart casual": 0.5, "formal": 0.8}
    freq_map = {"occasional": "Occasionally", "everyday": "Often"}
    gender_map = {"male": "Male", "female": "Female", "neutral": "Unisex"}

    user_event_input = {
        'Name': 'Curated Session',
        'Formality': formality_map.get(detected_vibes['formal'], 0.5),
        'Season': detected_vibes['season'].capitalize(),
        'Gender': gender_map.get(detected_vibes['gender'], "Unisex"),
        'Time_of_Day': detected_vibes['time'].capitalize(),
        'Frequency': freq_map.get(detected_vibes['frequency'], "Often"),
        'Longevity': 'Long', # CLIP doesn't assess this, default to Long
        'Description': user_text # Inject the frontend text box directly!
    }

    clip_reasoning = (
        f"Our vision model analyzed your look and detected a {detected_vibes['time']}time "
        f"{detected_vibes['season']} aesthetic."
    )

    return user_event_input, clip_reasoning