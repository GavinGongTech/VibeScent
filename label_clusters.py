"""
VibeScent — LLM-Based Cluster Vibe Labeling
Replaces the manual NOTES_VIBE_DICT with LLM-generated vibe vectors per cluster.
Uses Qwen3.5-27B via vLLM + Outlines for structured JSON output.
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from pydantic import BaseModel, Field

# ── Vibe schema (Outlines will enforce this) ─────────────────────────
class VibeVector(BaseModel):
    formality: float = Field(ge=0.0, le=1.0, description="0=casual, 1=black-tie formal")
    season: float    = Field(ge=0.0, le=1.0, description="0=summer/hot, 1=winter/cold")
    frequency: float = Field(ge=0.0, le=1.0, description="0=everyday, 1=special occasion")
    gender: float    = Field(ge=0.0, le=1.0, description="0=feminine, 1=masculine")
    time: float      = Field(ge=0.0, le=1.0, description="0=daytime, 1=nighttime")
    longevity: float = Field(ge=0.0, le=1.0, description="0=light/short, 1=heavy/long-lasting")

VIBE_DIMENSIONS = ['formality', 'season', 'frequency', 'gender', 'time', 'longevity']


# ── Build prompt for a cluster ───────────────────────────────────────
def build_cluster_prompt(rep_df: pd.DataFrame) -> str:
    """
    Build a prompt from representative fragrances of a cluster.
    """
    fragrance_descriptions = []
    for _, r in rep_df.iterrows():
        parts = [f"Name: {r['name']}"]
        if pd.notna(r.get('top_notes')):
            parts.append(f"Top notes: {r['top_notes']}")
        if pd.notna(r.get('middle_notes')):
            parts.append(f"Middle notes: {r['middle_notes']}")
        if pd.notna(r.get('base_notes')):
            parts.append(f"Base notes: {r['base_notes']}")
        if pd.notna(r.get('main_accords')):
            parts.append(f"Accords: {r['main_accords']}")
        if pd.notna(r.get('gender')):
            parts.append(f"Gender: {r['gender']}")
        if pd.notna(r.get('concentration')):
            parts.append(f"Concentration: {r['concentration']}")
        fragrance_descriptions.append("\n".join(parts))

    fragrances_block = "\n---\n".join(fragrance_descriptions)

    return f"""You are a fragrance expert. Below are representative fragrances from the same cluster. 
Analyze their shared characteristics and rate the OVERALL cluster profile on these dimensions.

Each dimension is a float from 0.0 to 1.0:
- formality: 0 = casual/everyday, 1 = black-tie/formal
- season: 0 = summer/hot weather, 1 = winter/cold weather  
- frequency: 0 = everyday wear, 1 = special occasion only
- gender: 0 = feminine, 1 = masculine (0.5 = unisex)
- time: 0 = daytime, 1 = nighttime
- longevity: 0 = light/fleeting, 1 = heavy/long-lasting

Fragrances in this cluster:
{fragrances_block}

Respond with ONLY a JSON object with the six dimensions."""


# ── Get representative fragrances per cluster ────────────────────────
def get_cluster_representatives(df, embeddings, kmeans, cluster_id, n_reps=5):
    """Pick the n fragrances closest to the cluster centroid."""
    cluster_mask = df['vibe_cluster_id'] == cluster_id
    cluster_indices = df.index[cluster_mask].tolist()
    
    if len(cluster_indices) == 0:
        return pd.DataFrame()
    
    centroid = kmeans.cluster_centers_[cluster_id]
    cluster_embeddings = embeddings[cluster_indices]
    dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    
    top_n = min(n_reps, len(cluster_indices))
    top_local = np.argsort(dists)[:top_n]
    top_global = [cluster_indices[i] for i in top_local]
    
    return df.loc[top_global]


# ── Label clusters with vLLM + Outlines ──────────────────────────────
def label_clusters_with_llm(df, embeddings, kmeans, n_clusters,
                             model_name="Qwen/Qwen3-Embedding-0.6B",
                             n_reps=5):
    """
    For each cluster, get representatives → prompt LLM → parse vibe vector.
    Returns a dict of {cluster_id: VibeVector}.
    
    Swap model_name to your actual Qwen3.5-27B-GPTQ-Int4 when running
    on your GPU server.
    """
    from outlines import models, generate

    print(f"Loading model: {model_name} ...")
    model = models.vllm(model_name)
    generator = generate.json(model, VibeVector)

    cluster_vibes = {}
    
    for cluster_id in range(n_clusters):
        reps = get_cluster_representatives(df, embeddings, kmeans, cluster_id, n_reps)
        
        if reps.empty:
            print(f"  Cluster {cluster_id}: empty, skipping")
            cluster_vibes[cluster_id] = VibeVector(
                formality=0.5, season=0.5, frequency=0.5,
                gender=0.5, time=0.5, longevity=0.5
            )
            continue
        
        prompt = build_cluster_prompt(reps)
        
        try:
            vibe = generator(prompt)
            cluster_vibes[cluster_id] = vibe
            
            if cluster_id % 10 == 0:
                print(f"  Cluster {cluster_id}: {vibe.model_dump()}")
                
        except Exception as e:
            print(f"  Cluster {cluster_id}: LLM failed ({e}), using defaults")
            cluster_vibes[cluster_id] = VibeVector(
                formality=0.5, season=0.5, frequency=0.5,
                gender=0.5, time=0.5, longevity=0.5
            )
    
    return cluster_vibes


# ── Fallback: label clusters WITHOUT an LLM (heuristic) ─────────────
def label_clusters_heuristic(df, n_clusters):
    """
    Rule-based fallback for testing without GPU access.
    Uses keyword matching on notes/accords to estimate vibe dimensions.
    Good enough for pipeline testing — replace with LLM version for real runs.
    """
    KEYWORD_SCORES = {
        'formality': {
            'high': ['oud', 'leather', 'amber', 'saffron', 'iris', 'tobacco'],
            'low':  ['citrus', 'lemon', 'lime', 'grapefruit', 'coconut', 'watermelon']
        },
        'season': {  # 0=summer, 1=winter
            'high': ['cinnamon', 'vanilla', 'amber', 'oud', 'tobacco', 'incense'],
            'low':  ['citrus', 'coconut', 'marine', 'aquatic', 'lemon', 'bergamot']
        },
        'frequency': {  # 0=everyday, 1=special
            'high': ['oud', 'saffron', 'ambergris', 'leather', 'iris'],
            'low':  ['musk', 'cotton', 'soap', 'lavender', 'lemon']
        },
        'gender': {  # 0=feminine, 1=masculine
            'high': ['leather', 'tobacco', 'vetiver', 'oud', 'whiskey'],
            'low':  ['rose', 'peony', 'jasmine', 'peach', 'cherry blossom']
        },
        'time': {  # 0=day, 1=night
            'high': ['amber', 'oud', 'patchouli', 'vanilla', 'musk', 'incense'],
            'low':  ['citrus', 'green tea', 'lemon', 'neroli', 'bergamot']
        },
        'longevity': {  # 0=light, 1=heavy
            'high': ['oud', 'amber', 'sandalwood', 'patchouli', 'musk'],
            'low':  ['citrus', 'lemon', 'neroli', 'green tea', 'aquatic']
        }
    }

    cluster_vibes = {}
    
    for cluster_id in range(n_clusters):
        cluster_df = df[df['vibe_cluster_id'] == cluster_id]
        
        if cluster_df.empty:
            cluster_vibes[cluster_id] = VibeVector(
                formality=0.5, season=0.5, frequency=0.5,
                gender=0.5, time=0.5, longevity=0.5
            )
            continue
        
        # combine all text for keyword search
        all_text = " ".join(cluster_df['embedding_text'].fillna("").tolist()).lower()
        
        scores = {}
        for dim in VIBE_DIMENSIONS:
            high_count = sum(1 for kw in KEYWORD_SCORES[dim]['high'] if kw in all_text)
            low_count = sum(1 for kw in KEYWORD_SCORES[dim]['low'] if kw in all_text)
            total = high_count + low_count
            if total == 0:
                scores[dim] = 0.5
            else:
                scores[dim] = round(high_count / total, 2)
        
        cluster_vibes[cluster_id] = VibeVector(**scores)
    
    return cluster_vibes


# ── Save / load helpers ──────────────────────────────────────────────
def save_cluster_vibes(cluster_vibes: dict, path: str):
    """Save cluster vibe mapping as JSON."""
    out = {str(k): v.model_dump() for k, v in cluster_vibes.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(out)} cluster vibe vectors → {path}")

def load_cluster_vibes(path: str) -> dict:
    """Load cluster vibe mapping from JSON."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k): VibeVector(**v) for k, v in raw.items()}


# ── Lookup: fragrance → vibe vector ─────────────────────────────────
def get_fragrance_vibe(fragrance_idx: int, df: pd.DataFrame, 
                        cluster_vibes: dict) -> np.ndarray:
    """Get the 6D vibe vector for a fragrance by its cluster assignment."""
    cluster_id = df.loc[fragrance_idx, 'vibe_cluster_id']
    vibe = cluster_vibes[cluster_id]
    return np.array([vibe.formality, vibe.season, vibe.frequency, 
                     vibe.gender, vibe.time, vibe.longevity])


# ── main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load your existing pipeline outputs
    df = pd.read_csv("data/processed/vibescent_clustered.csv")
    df = df.dropna(subset=['embedding_text', 'name']).reset_index(drop=True)
    embeddings = np.load("embeddings/fragrance_embeddings.npy")
    kmeans = joblib.load("models/kmeans_fragrance_model.pkl")
    N_CLUSTERS = kmeans.n_clusters

    # ── Pick your method ─────────────────────────────────────────────
    USE_LLM = False  # set True when you have GPU access

    if USE_LLM:
        cluster_vibes = label_clusters_with_llm(
            df, embeddings, kmeans, N_CLUSTERS,
            model_name="Qwen/Qwen3.5-27B-GPTQ-Int4",
            n_reps=5
        )
    else:
        print("Using heuristic labeling (no GPU). Switch USE_LLM=True for real run.")
        cluster_vibes = label_clusters_heuristic(df, N_CLUSTERS)

    # Save
    save_cluster_vibes(cluster_vibes, "models/cluster_vibe_mapping.json")
    print("\nDone! Run test_vibe_pipeline.py to validate.")