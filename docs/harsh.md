# Harsh — Text Retrieval and Reranking Lead

Last updated: April 18, 2026

---

## Scope

You own:

- Occasion-text embeddings (query side)
- Fragrance-text embeddings (document side)
- Shared-space multimodal embedding retrieval (text + image → fragrance)
- Text-to-fragrance retrieval pipeline
- Fragrance enrichment pipeline (`enrich.py`)
- Score fusion (`fusion.py`)
- Reranker (`reranker.py`)
- FastAPI backend (`backend_app.py`)
- All Pydantic schemas (`schemas.py`)
- 21-stage GPU orchestration notebook

You do not own:

- Fragrance dataset sourcing (Darren)
- CNN image classifier training (Neil)
- `display_text` generation (Karan)
- Frontend (Darren)

**Note on extended scope (April 12):** Harsh also owns the enrichment pipeline because Karan's raw `embedding_text` was plain note concatenation, insufficient for semantic retrieval. `enrich.py` generates structured `retrieval_text` using Gemini Flash / Qwen3.5-27B.

---

## Model Choices

| Role | Model | Provider | Key Metric |
|---|---|---|---|
| Text embedding | `Qwen3-Embedding-8B` | Local GPU | #1 MTEB English 68.32 |
| Multimodal embedding | `Qwen3-VL-Embedding-8B` | Local GPU | #1 MMEB-V2 77.8 |
| Enrichment LLM | `Qwen3.5-27B-GPTQ-Int4` (local) / `gemini-3-flash-preview` (API fallback) | Local / Google | Structured output, schema-enforced |
| Reranker | `gemini-3.1-pro-preview` | Google | Multimodal input, structured output |
| Evaluation judge | `gemini-2.5-pro` | Google | Separated from label generator |

Text embedding dimensionality: **1024** (Matryoshka prefix).
Multimodal embedding dimensionality: **1536** (cross-modal compatibility with Gemini embeddings).

---

## Current State (April 18)

### Done

| Artifact | Path |
|---|---|
| Text embedder — VoyageEmbedder, GeminiEmbedder, Qwen3VLMultimodalEmbedder | `src/vibescents/embeddings.py` |
| Enrichment pipeline — dual-LLM fallback, batch checkpointing, JSONL failure log | `src/vibescents/enrich.py` |
| Score fusion — min-max normalize, weighted sum, weight grid search | `src/vibescents/fusion.py` |
| Reranker — Gemini multimodal, schema-validated output | `src/vibescents/reranker.py` |
| Pipeline orchestration utilities — manifest system, GPU tier detection, disk guard, stage gates, embed_corpus with checkpointing | `src/vibescents/week2_pipeline.py` |
| Pydantic schemas — all pipeline contracts | `src/vibescents/schemas.py` |
| Similarity utilities — cosine matrix, top-k, weighted sum | `src/vibescents/similarity.py` |
| Embedding pipelines — embed-csv, embed-occasions, multimodal-retrieve | `src/vibescents/pipelines.py` |
| FastAPI backend — /healthz, /recommend, protocol injection | `src/vibescents/backend_app.py` |
| Qwen3-VL embedder implementation | `src/vibescents/qwen3_vl_embedding.py` |
| 21-stage orchestration notebook (Kaggle T4 compatible) | `notebooks/harsh_week2_pipeline.ipynb` |
| Occasion embedding matrix (8 × 1536) + similarity heatmap | `artifacts/occasions/` |
| Raw fragrance embedding matrix (500 × 1536) | `artifacts/fragrance_raw/embeddings.npy` |
| Week 2 report — sections 1 and 5 complete | `results/week2_report.md` |

### Pending

| Task | Blocker |
|---|---|
| Full 35,889-row corpus embedding at 1024-dim | Needs full Kaggle session (pipeline is ready — hardware/time constraint) |
| `artifacts/fragrance_enriched/embeddings.npy` | Enrichment run on full Tier B must complete first |
| Raw vs enriched retrieval comparison | Needs enriched embeddings |
| `artifacts/benchmark_cases.json` (20 labels) | Needs API keys and pipeline run |
| `results/week2_report.md` sections 2, 3, 6, 7 | Needs above artifacts |
| Multimodal retrieval on 3+ outfit images | Requires Colab Pro A100 (~16 GB VRAM) |

---

## Key Engineering Decisions

**Tier selection strategy:** strict filter (all 4 note columns non-null) → relaxed filter (top_notes + main_accords) → hard minimum of 500 rows. Sorted by `rating_count` descending — popularity is a proxy for LLM knowledge of that fragrance.

**Enrichment fault tolerance:** dual-provider (Qwen local → Gemini API), prompt shrink on first failure, JSONL failure log on second failure. Checkpoint to CSV every 16 rows. 98% non-null `vibe_sentence` required before the stage is marked complete.

**Embedding checkpointing:** `embed_corpus_resume()` globs partial `.npy` checkpoint files sorted by batch index, concatenates them, returns `(partial_matrix, next_batch_idx)`. Safe to kill mid-run on Colab.

**Sanity check:** after embedding, sample 1,000 random pairs. If cosine similarity variance < 0.001, raise — catches collapsed embeddings before they corrupt downstream stages.

**Score fusion normalization:** min-max per signal before weighted sum. Without this, a signal with naturally higher scores dominates by scale, not quality. Weights are found by grid search against the 20-case benchmark.

---

## Verification Checklist

- [x] `uv run pytest` passes
- [x] `artifacts/occasions/similarity_heatmap.png` — formality gradient confirmed
- [x] `artifacts/fragrance_raw/embeddings.npy` — shape (500, 1536)
- [ ] `artifacts/fragrance_enriched/embeddings.npy` — pending full run
- [ ] Text retrieval: different top-5 for `casual_day` vs `black_tie`
- [ ] Enriched retrieval quantifiably better than raw
- [ ] Multimodal retrieval on 3+ outfit images
- [ ] `artifacts/benchmark_cases.json` — 20 schema-consistent labels
- [ ] `results/week2_report.md` — all 7 sections complete
- [ ] Full corpus embeddings pushed and committed

---

## Interfaces

**You depend on:**
- Darren → cleaned fragrance table (`vibescent_500.csv` / full corpus) ✓
- Neil → CNN checkpoint (`artifacts/colab_upload_bundle/checkpoints/cnn/best.pt`) ✓
- Karan → `display_text` generation (pending), cluster vibe mapping (partial)

**Others depend on you:**
- Neil → text embedding pipeline ✓
- Karan → `retrieval_text` schema and enriched corpus ✓
- Gavin → enriched corpus, embedding artifacts, benchmark labels
