# Harsh — Text Retrieval and Reranking Lead

Last updated: April 12, 2026

---

## Scope

You own:

- Occasion-text embeddings (query side)
- Fragrance-text embeddings (document side)
- Shared-space multimodal embedding retrieval (text + image → fragrance)
- Text-to-fragrance retrieval pipeline
- Fragrance enrichment pipeline (`enrich.py`) — extended scope, see below
- The research reranker

You do not own:

- Fragrance dataset sourcing (Darren)
- Image feature extraction (Neil)
- `display_text` generation (Karan)
- Final frontend polish

**Extended scope (April 12):** Harsh also owns the fragrance enrichment pipeline because Karan's raw `embedding_text` in `vibescent_unified.csv` was plain note concatenation (e.g. `bergamot, lemon | jasmine, rose | sandalwood, vanilla`), which is insufficient for semantic retrieval against occasion descriptions. `enrich.py` generates structured `retrieval_text` using `gemini-3-flash-preview`.

---

## Model Choices

| Role | Model | Provider | Key Metric |
|---|---|---|---|
| Text embedding | `voyage-3-large` | Voyage AI (`VOYAGE_API_KEY`) | MTEB 68.32 (#1 English) |
| Multimodal embedding | `Qwen3-VL-Embedding-8B` | Local GPU | MMEB-V2 77.8 (#1) |
| Enrichment LLM | `gemini-3-flash-preview` | Google | structured output, fast, 5 RPM free tier |
| Reranker | `gemini-3.1-pro-preview` | Google | multimodal input, structured output |
| Evaluation judge | `gemini-2.5-pro` | Google | separated from label generator |

Text embedding dimensionality: **1024** (voyage-3-large default).
Multimodal embedding dimensionality: up to 4096 — use **1536** for cross-modal compatibility with text embeddings.

---

## Current State (April 12)

### Done

| Artifact | Path |
|---|---|
| Text embedder (`VoyageEmbedder`) | `src/vibescents/embeddings.py` |
| Multimodal embedder (`Qwen3VLMultimodalEmbedder`) | `src/vibescents/embeddings.py` |
| Embedding pipelines | `src/vibescents/pipelines.py` |
| Reranker | `src/vibescents/reranker.py` |
| Benchmark generator | `src/vibescents/benchmark.py` |
| Enrichment pipeline | `src/vibescents/enrich.py` |
| Qwen3-VL embedder implementation | `src/vibescents/qwen3_vl_embedding.py` |
| Raw 500-row fragrance sample | `data/vibescent_500.csv` |
| Enriched sample (small) | `data/vibescent_enriched_sample.csv` |
| Retrieval comparison script | `scripts/compare_retrieval.py` |
| Colab multimodal notebook | `scripts/multimodal_colab.ipynb` |

### Pending (execution — no new code needed except Step 5)

**Step 1 — Occasion embeddings + heatmap**
```bash
uv run vibescents embed-occasions \
  --input-json examples/occasions.json \
  --output-dir artifacts/occasions
```
Expected output: `artifacts/occasions/similarity_heatmap.png` showing formality gradient (casual_day ↔ summer_party similar; black_tie ↔ wedding_guest similar; streetwear_night ↔ business_dinner dissimilar).

**Step 2 — Enrich full fragrance corpus**
```bash
uv run python -m vibescents.enrich \
  --input-csv data/vibescent_500.csv \
  --output-csv data/vibescent_enriched.csv
```
Note: `gemini-3-flash-preview` is rate-limited to 5 RPM on the free tier. The pipeline has checkpointing via `vibescent_enriched.csv.ckpt` — safe to restart on interruption.

**Step 3 — Embed fragrance corpus (both raw and enriched)**
```bash
# Raw baseline (voyage-3-large, CPU)
uv run vibescents embed-csv \
  --input-csv data/vibescent_500.csv \
  --id-column fragrance_id \
  --text-column embedding_text \
  --output-dir artifacts/fragrance_raw

# Enriched (voyage-3-large, CPU)
uv run vibescents embed-csv \
  --input-csv data/vibescent_enriched.csv \
  --id-column fragrance_id \
  --text-column retrieval_text \
  --output-dir artifacts/fragrance_enriched
```
Expected shape: `(500, 1024)` for each.

**Step 4 — RAW vs ENRICHED retrieval comparison**
```bash
uv run python scripts/compare_retrieval.py
```
Output: `artifacts/retrieval_comparison.txt` — top-5 fragrances per occasion for RAW and ENRICHED side by side.

**Step 5 — Multimodal retrieval (on Colab A100, 3+ outfit images)**
```bash
uv run vibescents multimodal-retrieve \
  --fragrance-csv data/vibescent_enriched.csv \
  --id-column fragrance_id \
  --text-column retrieval_text \
  --occasion-text "Black tie evening wedding in winter" \
  --image-path assets/tuxedo.jpg \
  --output-dir artifacts/multimodal_blacktie \
  --top-k 10
```
Requires `uv sync --extra gpu` and ~16 GB VRAM. Use `scripts/multimodal_colab.ipynb`.

**Step 6 — Generate benchmark labels**
```bash
uv run vibescents generate-benchmark \
  --briefs-json examples/benchmark_briefs.json \
  --output-json artifacts/benchmark_cases.json \
  --runs 3
```

**Step 7 — Write `results/week2_report.md`**

Must cover:
1. Occasion embedding heatmap — screenshot + formality gradient interpretation
2. Fragrance embedding sanity check — known-similar fragrances vs known-dissimilar cosine scores
3. Raw vs enriched retrieval — side-by-side top-5 for each occasion
4. Multimodal ablation — text-only vs text+image results
5. Model justification — why `voyage-3-large` and `Qwen3-VL-Embedding-8B`
6. Benchmark label quality — spot-check 5 cases
7. Failure analysis — where retrieval breaks and what Week 3 should fix

---

## Week 3 Plan

- Integrate text retrieval into the late-fusion baseline
- Build the reranker prompt and JSON schema
- Run baseline vs reranker comparison on the 20-case benchmark
- Ablation: with and without `Qwen3-VL-Embedding-8B`

---

## Verification Checklist

- [ ] `uv run pytest` passes
- [ ] `artifacts/occasions/similarity_heatmap.png` shows formality gradient
- [ ] `artifacts/fragrance_raw/embeddings.npy` shape `(500, 1024)`
- [ ] `artifacts/fragrance_enriched/embeddings.npy` shape `(500, 1024)`
- [ ] Text retrieval returns different top-5 for `casual_day` vs `black_tie`
- [ ] Enriched retrieval is noticeably better than raw (or documented why not)
- [ ] Multimodal retrieval tested on 3+ outfit images
- [ ] `artifacts/benchmark_cases.json` has 20 schema-consistent labels
- [ ] `results/week2_report.md` covers all 7 required sections
- [ ] All changes committed and pushed to `Text_Processing`

---

## Interfaces

**You depend on:**
- Darren → cleaned fragrance table (`vibescent_500.csv`) ✓
- Karan → `display_text` generation (pending), structured attribute alignment (pending)
- Gavin → benchmark case file, evaluation schema

**Others depend on you:**
- Karan → text embedding pipeline ✓
- Gavin → enriched fragrance corpus, embedding artifacts, benchmark labels

---

## Required Outputs

- `artifacts/occasions/embeddings.npy` + `similarity_heatmap.png`
- `artifacts/fragrance_raw/embeddings.npy` + `metadata.csv`
- `artifacts/fragrance_enriched/embeddings.npy` + `metadata.csv`
- `artifacts/retrieval_comparison.txt`
- `artifacts/benchmark_cases.json`
- `results/week2_report.md`

---

## Success Criteria

- Similar occasions cluster together in the heatmap with a readable formality gradient
- Enriched `retrieval_text` produces better retrieval than raw `embedding_text` (quantified in report)
- `Qwen3-VL-Embedding-8B` multimodal signal changes results meaningfully when an outfit image is added
- Reranking improves or matches the baseline on the 20-case benchmark
