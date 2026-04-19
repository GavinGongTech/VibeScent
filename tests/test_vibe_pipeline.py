"""
VibeScent — Pipeline Sanity Tests
Run this after label_cluster_vibes.py to check that the vibe space makes sense.
"""

import numpy as np
import pandas as pd
import json
from label_cluster_vibes import (
    load_cluster_vibes, get_fragrance_vibe, 
    VibeVector, VIBE_DIMENSIONS
)

# ── Load everything ──────────────────────────────────────────────────
df = pd.read_csv("data/processed/vibescent_clustered.csv")
df = df.dropna(subset=['embedding_text', 'name']).reset_index(drop=True)
cluster_vibes = load_cluster_vibes("models/cluster_vibe_mapping.json")

print(f"Loaded {len(df)} fragrances across {len(cluster_vibes)} clusters\n")


# ── TEST 1: Inspect specific clusters ───────────────────────────────
def inspect_cluster(cluster_id, n=5):
    """Print a cluster's vibe vector and sample fragrances."""
    vibe = cluster_vibes[cluster_id]
    members = df[df['vibe_cluster_id'] == cluster_id]
    
    print(f"{'='*60}")
    print(f"CLUSTER {cluster_id}  ({len(members)} fragrances)")
    print(f"{'='*60}")
    print(f"Vibe: {vibe.model_dump()}")
    print(f"\nSample fragrances:")
    for _, r in members.head(n).iterrows():
        print(f"  • {r['name']} ({r.get('brand', '?')})")
        print(f"    Notes: {r.get('top_notes', 'N/A')}")
        print(f"    Accords: {r.get('main_accords', 'N/A')}")
    print()

print("── TEST 1: Cluster Inspection ──")
# look at a few random clusters
for cid in [0, 10, 50, 100]:
    if cid < len(cluster_vibes):
        inspect_cluster(cid)


# ── TEST 2: Fake outfit vibe → nearest clusters ─────────────────────
def match_outfit_to_clusters(outfit_vibe: dict, top_k=5):
    """
    Simulate the inference pipeline:
    outfit → LLM → vibe vector → find nearest fragrance clusters.
    """
    outfit_vec = np.array([outfit_vibe[d] for d in VIBE_DIMENSIONS])
    
    distances = {}
    for cid, vibe in cluster_vibes.items():
        cluster_vec = np.array([
            vibe.formality, vibe.season, vibe.frequency,
            vibe.gender, vibe.time, vibe.longevity
        ])
        dist = np.linalg.norm(outfit_vec - cluster_vec)
        distances[cid] = dist
    
    ranked = sorted(distances.items(), key=lambda x: x[1])[:top_k]
    return ranked

print("── TEST 2: Outfit → Cluster Matching ──\n")

# Define test outfits with expected behavior
TEST_OUTFITS = {
    "Black tie gala": {
        "vibe": {"formality": 0.95, "season": 0.6, "frequency": 0.9, 
                 "gender": 0.7, "time": 0.9, "longevity": 0.8},
        "expect": "formal, evening, masculine-leaning, long-lasting"
    },
    "Summer sundress, brunch": {
        "vibe": {"formality": 0.1, "season": 0.1, "frequency": 0.2, 
                 "gender": 0.15, "time": 0.2, "longevity": 0.3},
        "expect": "casual, summer, feminine, light, daytime"
    },
    "Streetwear, night out": {
        "vibe": {"formality": 0.3, "season": 0.4, "frequency": 0.5, 
                 "gender": 0.5, "time": 0.8, "longevity": 0.6},
        "expect": "casual-ish, unisex, nighttime"
    },
    "Business casual, office": {
        "vibe": {"formality": 0.6, "season": 0.5, "frequency": 0.3, 
                 "gender": 0.5, "time": 0.3, "longevity": 0.5},
        "expect": "moderate formality, everyday, unisex, balanced"
    },
}

for outfit_name, config in TEST_OUTFITS.items():
    print(f"Outfit: {outfit_name}")
    print(f"  Expected: {config['expect']}")
    print(f"  Vibe vector: {config['vibe']}")
    
    matches = match_outfit_to_clusters(config['vibe'], top_k=3)
    
    for rank, (cid, dist) in enumerate(matches, 1):
        members = df[df['vibe_cluster_id'] == cid]
        sample_names = members['name'].head(3).tolist()
        print(f"  #{rank} Cluster {cid} (dist={dist:.3f}): {sample_names}")
    print()


# ── TEST 3: Do different outfits give different results? ─────────────
print("── TEST 3: Differentiation Check ──\n")

all_top_clusters = []
for outfit_name, config in TEST_OUTFITS.items():
    matches = match_outfit_to_clusters(config['vibe'], top_k=3)
    top_cluster_ids = [cid for cid, _ in matches]
    all_top_clusters.append(set(top_cluster_ids))
    
# check pairwise overlap
outfit_names = list(TEST_OUTFITS.keys())
all_unique = True
for i in range(len(outfit_names)):
    for j in range(i+1, len(outfit_names)):
        overlap = all_top_clusters[i] & all_top_clusters[j]
        if overlap:
            print(f"⚠ Overlap between '{outfit_names[i]}' and '{outfit_names[j]}': clusters {overlap}")
            all_unique = True  # overlap is okay if not complete
        if all_top_clusters[i] == all_top_clusters[j]:
            print(f"✗ IDENTICAL results for '{outfit_names[i]}' and '{outfit_names[j]}' — scoring needs work")
            all_unique = False

if all_unique:
    print("✓ All outfits produce distinct top-3 cluster matches\n")


# ── TEST 4: Vibe space coverage ──────────────────────────────────────
print("── TEST 4: Vibe Space Coverage ──\n")

all_vecs = np.array([
    [v.formality, v.season, v.frequency, v.gender, v.time, v.longevity]
    for v in cluster_vibes.values()
])

for i, dim in enumerate(VIBE_DIMENSIONS):
    vals = all_vecs[:, i]
    print(f"  {dim:12s}: min={vals.min():.2f}  max={vals.max():.2f}  "
          f"mean={vals.mean():.2f}  std={vals.std():.2f}")

print()
# Flag if any dimension is collapsed (all clusters ~same value)
for i, dim in enumerate(VIBE_DIMENSIONS):
    if all_vecs[:, i].std() < 0.05:
        print(f"⚠ '{dim}' has very low variance — clusters aren't differentiating on this dimension")

print("\n✓ Pipeline test complete.")