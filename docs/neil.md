# Neil — Image Retrieval Lead

Last updated: April 12, 2026

---

## Scope

You own:

- Image-to-fragrance retrieval
- Image preprocessing
- CLIP-only, CNN-only, and hybrid branch comparison

---

## Status: AT RISK

**`Image_Processing` branch is empty as of April 8.** None of the Week 2 deliverables have been completed.

This blocks:
- The 4-signal fusion formula (`final_score = 0.30 * text + 0.25 * multimodal + 0.30 * image + 0.15 * structured`)
- The CLIP vs CNN vs hybrid comparison required by the project
- Week 3 integration

If Neil cannot deliver by Week 3 start, Gavin must decide: (a) assign image retrieval to another branch, or (b) officially adopt the 2-signal fallback formula. Do not silently let this block integration.

---

## Model Choices

| Branch | Model |
|---|---|
| CLIP-only | OpenCLIP |
| CNN-only | ResNet50 |
| Hybrid | `image_score = 0.70 * clip_score + 0.30 * cnn_score` |

CLIP should carry most of the semantic image-text matching load. CNN features add visual structure without destabilizing the pipeline.

---

## Week 2 Deliverables (all pending)

- CLIP branch: image → fragrance similarity using OpenCLIP
- CNN branch: image → fragrance similarity using ResNet50
- Hybrid branch: score fusion of CLIP and CNN
- Save image embeddings as `.npy`
- Save image-to-fragrance score table as `.csv`
- Produce nearest-neighbor sanity check (visually similar outfits should rank similarly)
- One short note on failure modes

---

## Week 3 Plan

- Integrate the winning image branch into the late-fusion baseline
- Compare image-only vs fused (text + image + multimodal) retrieval
- Compare CLIP / CNN / hybrid behavior against the `Qwen3-VL-Embedding-8B` multimodal signal
  - Note: the multimodal model has changed from `gemini-embedding-2` to `Qwen3-VL-Embedding-8B`. MMEB-V2 score 77.8 (#1). The comparison must show how much unique value the image branch adds on top of this stronger baseline.

---

## Required Outputs

- `artifacts/image_clip/embeddings.npy` + scores
- `artifacts/image_cnn/embeddings.npy` + scores
- `artifacts/image_hybrid/scores.csv`
- CLIP vs CNN vs hybrid comparison table
- One short note on failure modes and edge cases

---

## Interfaces

**You depend on:**
- Darren → cleaned fragrance table (`vibescent_500.csv`) ✓
- Karan → `retrieval_text` and structured fragrance attributes
- Gavin → benchmark case file

**Others depend on you:**
- Gavin → image retrieval scores for late-fusion integration
- Harsh → image scores for fusion formula

---

## Success Criteria

- Visually similar outfits score similarly against the fragrance corpus
- Image-only retrieval produces plausible fragrance candidates (not random)
- The hybrid branch is clearly defined, implemented, and benchmarked against CLIP-only and CNN-only
- The image branch contributes unique signal even when `Qwen3-VL-Embedding-8B` is added to the fusion
