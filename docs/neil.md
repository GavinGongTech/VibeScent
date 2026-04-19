# Neil — Image Retrieval Lead

Last updated: April 18, 2026

---

## Scope

You own:

- CNN-CLIP hybrid image classifier (3 classification heads)
- Qwen3-VL-Embedding-8B integration (`qwen3_vl_embedding.py`)
- Image preprocessing pipeline
- CLIP-only, CNN-only, and hybrid branch comparison

---

## Current State (April 18)

**Delivered.** The CNN-CLIP hybrid model is trained, checkpointed, and integrated.

| Artifact | Path | Status |
|---|---|---|
| CNN-CLIP hybrid model definition | `models/cnn_clip_hybrid.py` | ✅ |
| Trained checkpoint | `artifacts/colab_upload_bundle/checkpoints/cnn/best.pt` | ✅ |
| Qwen3-VL-Embedding-8B inner embedder | `src/vibescents/qwen3_vl_embedding.py` | ✅ |
| Image preprocessing | `src/vibescents/image_preprocess.py` | ✅ |
| CLIP zero-shot labeler | `src/clip_zero_shot.py` | ✅ |
| Inference script | `src/inference.py` | ✅ |

Harsh's `image_scoring.py` wraps the CNN checkpoint via `NeilCNNWrapper` and uses it in the fusion pipeline. The integration is complete.

---

## Architecture

**CNN-CLIP Hybrid** — combines visual structure (ResNet-50) with semantic understanding (CLIP ViT-L/14):

```
outfit image (224 × 224, normalized with CLIP mean/std)
    │
    ├── ResNet-50  →  2,048-d feature vector
    └── CLIP ViT-L/14  →  768-d feature vector
                │
           concatenate
                │
         Linear projection
                │
         256-d shared trunk
                │
    ┌───────────┼───────────┐
    │           │           │
formal_head  season_head  time_head
 (3-class)   (4-class)   (2-class)
    │           │           │
 softmax     softmax     softmax
    │           │           │
P(casual/    P(spring/   P(day/
 semi/        summer/     night)
 formal)      fall/
              winter)
```

**Why ResNet-50 + CLIP instead of just CLIP?**
CLIP alone produces an opaque embedding — you can query it but you can't inspect what it sees.
The classification heads give you explicit, debuggable predictions: *P(formal) = 0.83*.
These map directly to the numeric enrichment attributes on fragrances, creating a shared attribute space between the outfit and the fragrance database.

---

## Training Details

**Loss:** Per-head weighted CrossEntropyLoss — each head trains on its own classification target independently.

**Pseudo-labels:** CLIP zero-shot pseudo-labeler (`src/clip_zero_shot.py`) generated training labels across all 5 vibe dimensions before fine-tuning. Ground truth annotations were expensive; zero-shot labels provided enough signal to train the classification heads.

**Checkpoint format:** `best.pt` saves either full model state or `model_state_dict` key — `NeilCNNWrapper` in `image_scoring.py` handles both formats during load.

---

## How It Integrates Into the Pipeline

At inference time, `image_scoring.py:NeilCNNWrapper`:

1. Loads checkpoint with key-format fallback
2. Runs forward pass, extracts 3 head outputs
3. Returns `ImageHeadProbabilities(formal_probs, season_probs, time_probs)`

`score_candidate_pool()` then scores each Tier B fragrance:
```python
score = P_formal[fragrance_formal_class]
      × P_season[fragrance_season_class]
      × P_time[fragrance_time_class]
```

Scores are min-max normalized across the pool and fed into the fusion formula as `sig_img`.

---

## Still Pending

- **Comparison table:** CLIP-only vs CNN-only vs hybrid vs `Qwen3-VL-Embedding-8B` — required to justify the hybrid's value above each single-model baseline
- **Image embeddings as `.npy`:** `artifacts/image_clip/embeddings.npy`, `artifacts/image_cnn/embeddings.npy`
- **Failure mode note:** edge cases where the CNN misclassifies (e.g. avant-garde fashion that reads as casual despite formal intent)

---

## Required Outputs

| Artifact | Status |
|---|---|
| `models/cnn_clip_hybrid.py` | ✅ |
| `artifacts/colab_upload_bundle/checkpoints/cnn/best.pt` | ✅ |
| `src/vibescents/qwen3_vl_embedding.py` | ✅ |
| CLIP-only vs CNN-only vs hybrid comparison table | ❌ Pending |
| `artifacts/image_clip/embeddings.npy` | ❌ Pending |
| `artifacts/image_cnn/embeddings.npy` | ❌ Pending |
| Failure mode analysis | ❌ Pending |

---

## Interfaces

**You depend on:**
- Darren → cleaned fragrance table (`vibescent_500.csv`) ✓

**Others depend on you:**
- Harsh → CNN checkpoint for `NeilCNNWrapper` in `image_scoring.py` ✓
- Harsh → `Qwen3VLMultimodalEmbedder` wraps your `qwen3_vl_embedding.py` ✓
- Gavin → image retrieval scores for fusion integration
