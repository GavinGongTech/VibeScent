# Karan — Fragrance Representation Lead

Last updated: April 18, 2026

---

## Scope

You own:

- `display_text` generation — expressive, demo-optimized copy for the UI
- Structured fragrance attribute table (formality scores, season tags, occasion tags)
- Fragrance clustering and vibe space mapping
- Validation that similar fragrances cluster together in embedding space

You do **not** own `retrieval_text` generation. That pipeline was built by Harsh in `src/vibescents/enrich.py` because the original `embedding_text` (raw note concatenation) was insufficient for semantic retrieval.

---

## What You Actually Built (April 17 — `b0d0267`)

Karan built a **cluster-based vibe pipeline** — a separate, parallel approach to fragrance representation:

```
embed_fragrances.py     →  embeddings/fragrance_embeddings.npy   (35,889 × 384-dim)
cluster_fragrances.py   →  data/processed/vibescent_clustered.csv  (adds vibe_cluster_id)
                           models/kmeans_fragrance_model.pkl
label_cluster_vibes.py  →  models/cluster_vibe_mapping.json        ← NOT YET COMMITTED
test_vibe_pipeline.py   →  validation results
```

**What this does:** K-means clusters the full 35,889 fragrances by embedding similarity, then uses an LLM to label each cluster with a vibe descriptor. Every fragrance row now has a `vibe_cluster_id` column pointing to its cluster.

**How this differs from Harsh's enrichment pipeline:**

| | Karan's cluster approach | Harsh's per-fragrance enrichment |
|---|---|---|
| Granularity | Cluster-level — one label per group of ~100 fragrances | Per-fragrance — 11 individual attributes each |
| Embedding dim | 384 (small SentenceTransformer model) | 1024 (Qwen3-Embedding-8B, #1 MTEB) |
| LLM calls | One per cluster | One per fragrance |
| Output | `vibe_cluster_id` integer per row | `formality`, `day_night`, `character_tags`, `vibe_sentence`, etc. |
| Speed | Fast — cheap embeddings, few LLM calls | Slow — 2,000 LLM calls, large model |
| Quality | Coarser — similar fragrances get the same label | Precise — each fragrance is individually characterized |

**These are not interchangeable.** Karan's cluster IDs are useful as a coarse first-pass signal; Harsh's enrichment attributes are what the scoring and reranking use at inference time.

---

## Outstanding Gap

`models/cluster_vibe_mapping.json` — the output of `label_cluster_vibes.py` — **has not been committed to the repo.** The K-means model exists, the clustered CSV exists, but the actual cluster vibe labels are missing. This needs to be pushed before the cluster signal can be used in the fusion formula.

---

## Alignment Note (April 12, still current)

Karan's `embedding_text` field in `vibescent_unified.csv` was plain note concatenation:

```
bergamot, lemon | jasmine, rose | sandalwood, vanilla
```

This is ingredient-level text. It cannot bridge the semantic gap to occasion descriptions. Harsh built `enrich.py` to generate structured `retrieval_text`. The canonical retrieval string is now `retrieval_text` from `data/vibescent_enriched.csv`.

Karan's structured attribute table (`formality_score`, `fresh_warm_score`, `day_night_score`) needs deduplication with the numeric fields already produced by `enrich.py` before Week 3 integration. Do not generate duplicate numeric columns with different names.

---

## Still Pending

- **`display_text` generation** — expressive, demo-optimized copy for the UI. Not built yet. Must be grounded in actual metadata — not creative fiction. Target tone:
  > *A warm, sensual wood that lingers well past midnight. Built for candlelight and slow conversation — sandalwood depth softened by amber, with a whisper of iris that stops just short of sweet.*
- **`models/cluster_vibe_mapping.json`** — commit the output of `label_cluster_vibes.py`
- **Clustering validation chart** — show similar fragrances (e.g. aquatic fougères) cluster closer than dissimilar ones

---

## Required Outputs

| Artifact | Status |
|---|---|
| `embeddings/fragrance_embeddings.npy` (35,889 × 384) | ✅ Done |
| `data/processed/vibescent_clustered.csv` (with `vibe_cluster_id`) | ✅ Done |
| `models/kmeans_fragrance_model.pkl` | ✅ Done |
| `models/cluster_vibe_mapping.json` | ❌ Not committed |
| `display_text` column in enriched CSV | ❌ Not built |
| Clustering validation chart | ❌ Not built |

---

## Interfaces

**You depend on:**
- Darren → canonical fragrance table (`vibescent_500.csv`) ✓
- Harsh → `retrieval_text` schema and enriched corpus ✓

**Others depend on you:**
- Harsh → `display_text` for the reranker output display
- Gavin → structured attributes for benchmark scoring
