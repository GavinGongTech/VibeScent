# VibeScent — Updated Plan

Supersedes: `plan-gavin-original.md`
Updated: April 12, 2026
Changes from original are marked **[UPDATED]** or **[NEW]**. Everything unmarked is unchanged from Gavin's original.

---

## Locked Decisions

- The project is optimizing for both research novelty and demo quality.
- The retrieval baseline is late fusion.
- The research layer is reranking only, not first-stage cross-attention.
- The text embedding baseline is `gemini-embedding-001`.
- **[UPDATED]** The primary shared-space multimodal production signal is `Qwen3-VL-Embedding-8B` (was `gemini-embedding-2`). MMEB-V2 score: 77.8 vs 68.9 — 9-point gap on the cross-modal retrieval benchmark directly relevant to this task. Runs locally on Colab Pro A100.
- The image branch must compare `CLIP-only`, `CNN-only`, and `hybrid`.
- The hybrid starts as score fusion.
- Every fragrance has `retrieval_text` and `display_text`.
- **[UPDATED]** Fragrance text is generated with LLM enrichment (`gemini-3-flash-preview`) using structured JSON schema, not free-form blurbs. The enrichment pipeline lives in `src/vibescents/enrich.py` on the `Text_Processing` branch.
- The benchmark uses 20 end-to-end cases.
- Benchmark labels are AI-assisted, but the primary score is metadata-based, not pure LLM judging.
- The primary multimodal reranker is `gemini-3.1-pro-preview`.
- **[NEW]** Occasions corpus is 8 descriptions, not 5. Covers the full formality spectrum: `casual_day`, `creative_office`, `business_dinner`, `wedding_guest`, `black_tie`, `streetwear_night`, `summer_party`, `editorial`.

## Reporting Rules

- Do not present five examples as validation.
- Do not present AI-generated benchmark labels as human ground truth.
- Do not say `top-p retrieval`.
- Do not add architecture complexity before the baseline is benchmarked.
- Do not treat preview-model behavior as guaranteed; keep the documented fallback path live.
- **[NEW]** Do not call raw note concatenation a retrieval text. `retrieval_text` must include enriched fields (season, occasion, formality, character, vibe).

---

## Project Goal

Build a competition-grade system that takes an outfit image and occasion text, retrieves fragrance recommendations, and supports a flashy final demo with defensible ML choices.

## Success Criteria

- returns fragrance recommendations that feel coherent
- beats simple text-only and image-only baselines
- produces stable outputs across repeated runs
- supports a polished frontend demo
- gives the team a research story that is more than prompt glue

---

## Delivery Strategy

### Baseline layer

The baseline must always work.

- retrieve fragrances from occasion text
- retrieve fragrances from outfit images
- retrieve shared-space similarities with a multimodal embedding model
- combine the signals with a transparent scoring rule
- return top candidates with explanations

### Research layer

The research layer is allowed to improve the baseline, but it is not allowed to replace the baseline until it clearly wins on the benchmark.

- rerank the shortlist with a stronger multimodal model
- compare baseline versus reranked results
- keep the baseline as the demo fallback

---

## Timeline

### Week 2 (deadline: April 12, 2026)

Deliver branch artifacts that can be integrated:

- fragrance dataset selected and cleaned ✓ (Darren/Karan — `vibescent_500.csv`)
- **[UPDATED]** fragrance descriptions enriched with LLM (Season, Occasion, Formality, Character, Vibe) ✓ (`enrich.py` on `Text_Processing`)
- occasion text retrieval working — **pending** (run `embed-occasions`)
- **[UPDATED]** `Qwen3-VL-Embedding-8B` multimodal retrieval working — **pending** (run `multimodal-retrieve` on Colab)
- image retrieval working — **[RISK]** `Image_Processing` branch is empty as of April 8. Neil has not delivered CLIP/CNN/hybrid. This blocks Week 3 fusion if unresolved.
- branch-level sanity checks completed — **pending**

### Week 3

Integrate the branches and benchmark the full system:

- late-fusion baseline working end to end
- 20-case benchmark created
- reranker added on top of baseline
- **[UPDATED]** ablation with and without `Qwen3-VL-Embedding-8B` completed (was `gemini-embedding-2`)
- baseline versus reranker comparison completed

### Week 4

Stabilize and present:

- final demo flow locked
- final benchmark numbers locked
- demo cases chosen
- presentation deck written around real results

---

## Artifact Contract

Every branch must produce:

- reproducible code entrypoint
- saved outputs under a predictable directory
- metadata file that maps rows to source items
- **[UPDATED]** one results write-up (`results/week2_report.md`) covering: heatmap interpretation, raw vs enriched retrieval comparison, multimodal ablation, benchmark label quality, failure analysis, model justification. A "short note" is not sufficient for competition evaluation.

Recommended artifact formats:

- embeddings: `.npy`
- tabular metadata: `.csv`
- configuration: `.json`
- figures: `.png`

## Required Shared Fields

Every fragrance row should contain:

- `fragrance_id`
- `brand`
- `name`
- `notes`
- `accords`
- `season_tags`
- `occasion_tags`
- `formality_score`
- `fresh_warm_score`
- `day_night_score`
- `retrieval_text`
- `display_text`

**[NEW]** The following enriched fields are required to build a valid `retrieval_text`. They are generated by `enrich.py` using `gemini-3-flash-preview`:

- `likely_season`
- `likely_occasion`
- `formality` (0.0–1.0)
- `fresh_warm` (0.0–1.0)
- `day_night` (0.0–1.0)
- `character_tags`
- `vibe_sentence`

## Core Risks

- fragrance datasets may be incomplete or inconsistent
- richer text generation may introduce noise if not schema-controlled
- AI-assisted evaluation can become circular if the same model both writes and grades the benchmark
- the reranker can become impressive-looking but unmeasured if the benchmark is weak
- **[UPDATED]** `Qwen3-VL-Embedding-8B` requires ~16 GB VRAM. All fragrance document embeddings must be pre-computed and cached as `.npy` before the demo — the model cannot run live on CPU.
- **[NEW]** Neil's `Image_Processing` branch is empty. If it remains empty at Week 3 integration, the fusion formula falls back to the 3-signal variant. Flag this to Gavin immediately.

## Non-Negotiables

- benchmark before polishing
- keep the baseline runnable at all times
- separate retrieval text from display text
- preserve interpretable fragrance metadata
- document every model choice and fallback

---

## Architecture

### System Overview

The system has three stages:

1. candidate retrieval
2. score fusion
3. reranking

### Stage 1: Candidate Retrieval

#### Occasion-text retrieval

Input: occasion text from the user
Target: fragrance `retrieval_text`
Primary model: `gemini-embedding-001`
Dimensionality: `1536`

#### Shared-space multimodal retrieval

Inputs: occasion text from the user, outfit image
Target: fragrance `retrieval_text`
**[UPDATED]** Primary model: `Qwen3-VL-Embedding-8B`

Why this model (replacing `gemini-embedding-2`):

- MMEB-V2 score: **77.8** (#1 overall) vs `gemini-embedding-2` at 68.9 — a 9-point gap on the cross-modal image+text retrieval benchmark directly relevant to this task
- Natively multimodal: maps text and images into a single shared embedding space using a unified VLM transformer (Qwen3-VL backbone), not a dual-encoder aligned via contrastive loss
- Runs locally on Colab Pro A100 (~16 GB VRAM in float16) — no API quota risk during demo
- Implementation: `src/vibescents/qwen3_vl_embedding.py` (vendored from HuggingFace model repo), `Qwen3VLMultimodalEmbedder` class in `embeddings.py`

Dimensionality: up to 4096 (configurable 64–4096; use 1536 for compatibility with existing text embeddings).

**[NEW]** GPU setup for Colab:
```bash
pip install torch transformers>=4.57.0 qwen-vl-utils>=0.0.14 accelerate
# or: uv sync --extra gpu
```

#### Image retrieval

Input: outfit image
Target: fragrance `retrieval_text`

Required branches: `CLIP-only`, `CNN-only`, `hybrid`
Recommended models: `OpenCLIP` for CLIP, `ResNet50` for CNN
Hybrid: `image_score = 0.70 * clip_score + 0.30 * cnn_score`

**[RISK]** Neil's `Image_Processing` branch has not delivered any of these as of April 8.

### Stage 2: Score Fusion

The first-stage ranking combines four signals:

- occasion-text score
- shared-space multimodal score
- image score
- structured attribute score

```
final_score = 0.30 * text_score + 0.25 * multimodal_score + 0.30 * image_score + 0.15 * structured_score
```

Structured score uses fragrance metadata: season fit, formality fit, day or night fit, fresh versus warm fit.

Fallback rule (if image branch or multimodal unavailable):

```
final_score = 0.45 * text_score + 0.40 * image_score + 0.15 * structured_score
```

**[NEW]** If Neil's image branch is still missing at Week 3:

```
final_score = 0.60 * text_score + 0.40 * multimodal_score
```

Until a proper 3- or 4-signal fallback is agreed with Gavin.

### Stage 3: Reranking

The reranker sees only the shortlist from Stage 2.
Primary reranker: `gemini-3.1-pro-preview`

Input per request: outfit image, occasion text, candidate fragrance `retrieval_text`, candidate fragrance structured metadata.

Output: per-candidate relevance score, short rationale, optional sub-scores for formality, season, freshness, and overall vibe fit.

The reranker is research scope. It does not replace the baseline unless it wins on the benchmark.

### Fragrance Representation

Every fragrance should have two text fields.

#### `retrieval_text`

Optimized for embeddings and ranking. Schema-controlled. Must include all enriched fields.

Format (produced by `enrich.py`):

```
Brand: {brand} | Name: {name}
Accords: {main_accords}
Top: {top_notes} | Heart: {middle_notes} | Base: {base_notes}
Season: {likely_season} | Best for: {likely_occasion}
Formality: {low/medium/high} | Character: {character_tags}
Vibe: {vibe_sentence}
```

Raw note concatenation (e.g. `bergamot, lemon | jasmine, rose | sandalwood, vanilla`) is **not** a valid `retrieval_text`. It is ingredient-level and cannot bridge the semantic gap to occasion descriptions.

#### `display_text`

Optimized for the demo UI. Can be more stylistic and varied, but must remain grounded in actual metadata.

### Multimodal Embedding Ablation

**[UPDATED]** `Qwen3-VL-Embedding-8B` is part of production scoring and still needs ablation.

Required comparisons:

- baseline without `Qwen3-VL-Embedding-8B` (text + image + structured only)
- baseline with `Qwen3-VL-Embedding-8B`
- reranked system without `Qwen3-VL-Embedding-8B`
- reranked system with `Qwen3-VL-Embedding-8B`

### Why This Architecture

- it supports both novelty and reliability
- it keeps the retrieval path debuggable
- it gives Harsh a meaningful research reranker to own
- it preserves Neil's required CLIP, CNN, and hybrid comparison
- **[UPDATED]** `Qwen3-VL-Embedding-8B` provides state-of-the-art cross-modal retrieval (MMEB-V2 #1) and runs fully locally, removing API dependency from the multimodal signal
- it still preserves a fallback path if model inference fails

### Official References

- Qwen3-VL-Embedding-8B HuggingFace: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
- Qwen3-VL-Embedding GitHub: https://github.com/QwenLM/Qwen3-VL-Embedding
- MMEB Leaderboard: https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard
- Gemini 3.1 Pro Preview: https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview
- Gemini embeddings guide: https://ai.google.dev/gemini-api/docs/embeddings
- Structured outputs: https://ai.google.dev/gemini-api/docs/structured-output
- Batch API: https://ai.google.dev/gemini-api/docs/batch-api

---

## Evaluation Plan

### Goal

Measure whether the system is actually improving recommendation quality without pretending that AI-written labels are the same thing as human ground truth.

### Benchmark Structure

The benchmark contains 20 end-to-end cases.

Each case includes: one outfit image, one occasion description, target style attributes, acceptable fragrance neighborhoods, disallowed traits.

### Label Generation

The benchmark is AI-assisted.
Primary label generator: `gemini-3.1-pro-preview`

Generation method: batch or looped structured-output requests; three independent generations per case; keep only cases with strong agreement across runs.

Required output schema per case:

- `case_id`
- `occasion_text`
- `target_formality`
- `target_season`
- `target_day_night`
- `target_fresh_warm`
- `acceptable_accords`
- `acceptable_note_families`
- `disallowed_traits`
- `example_good_fragrances`
- `confidence`

### Primary Scoring

Do not use the same model as both label writer and primary judge. Primary score is metadata-based.

For each retrieved fragrance, score: accord match, note-family match, season match, formality match, day or night match, fresh versus warm match.

Compute:

- `attribute_match@3`
- `attribute_match@5`
- `neighborhood_hit@3`
- `neighborhood_hit@5`

### Secondary Scoring

Recommended judge: `gemini-2.5-pro` (different from label generator — avoids self-confirmation bias).

Judge inputs: outfit image, occasion text, top candidates from baseline, top candidates from reranker.
Judge outputs: preferred shortlist, short rationale, confidence.

### Required Reporting

Report all of the following:

- text-only retrieval performance
- image-only retrieval performance
- **[UPDATED]** `Qwen3-VL-Embedding-8B` multimodal retrieval performance
- late-fusion baseline performance
- reranked performance
- **[UPDATED]** with and without `Qwen3-VL-Embedding-8B`
- benchmark cases where reranking helps
- benchmark cases where reranking hurts
- **[NEW]** raw `embedding_text` vs enriched `retrieval_text` retrieval comparison

### Presentation Rule

Call this what it is: an AI-assisted benchmark with metadata-based scoring. Do not call it human-labeled ground truth.

---

## Team Roles

### Gavin — PM and Integration Lead

**Scope:** branch contracts, benchmark files, integration deadlines, final result reporting.

**Week 2:** lock the fragrance dataset choice ✓; lock the shared schema ✓; lock output paths and file formats ✓.

**Week 3:** publish the 20-case benchmark; collect branch outputs; integrate the late-fusion baseline; make sure reranker comparisons are reproducible; **[UPDATED]** publish ablation results with and without `Qwen3-VL-Embedding-8B`.

**Week 4:** lock demo cases; lock final metrics; keep the final story consistent with the actual benchmark.

**[NEW] Immediate action required:** Neil's `Image_Processing` branch is empty as of April 8. This must be escalated now. If Neil cannot deliver CLIP/CNN/hybrid by Week 3 start, Gavin needs to either (a) assign image retrieval to another branch or (b) officially adopt the 2-signal fallback formula for integration.

---

### Darren — Fragrance Data Lead

**Scope:** fragrance dataset selection, schema definition, data loading, normalization.

**Week 2:** ✓ Delivered `vibescent_500.csv` (500-row cleaned dataset). Schema documented.

**Week 3:** improve coverage only if it does not break the canonical schema; support targeted enrichment where the benchmark exposes gaps.

**Required fields:** `fragrance_id`, `brand`, `name`, `notes`, `accords`, any available season, occasion, and gender metadata.

**Constraints:** do not merge multiple messy datasets before one clean table exists; do not let schema drift across branches.

---

### Karan — Fragrance Representation Lead

**Scope:** fragrance text generation, structured fragrance attributes, note-family and accord mapping.

**[UPDATED] Status:** The `retrieval_text` and enrichment pipeline was built by Harsh on the `Text_Processing` branch (`src/vibescents/enrich.py`) because Karan's `embedding_text` in `vibescent_unified.csv` was raw note concatenation, which is insufficient for semantic retrieval. Karan and Harsh need to align on which CSV is the canonical source for `retrieval_text` going into Week 3 integration.

**Karan's remaining scope for Week 2:**

- `display_text` generation — still Karan's responsibility, not built yet
- structured attribute table (formality scores, season tags, occasion tags) — partially overlaps with enriched fields in `enrich.py`; needs deduplication with Harsh

**Week 3:** refine representation quality based on benchmark failures; support reranker inputs with cleaner structured fields.

---

### Neil — Image Retrieval Lead

**Scope:** image-to-fragrance retrieval, image preprocessing, CLIP-only / CNN-only / hybrid comparison.

**[RISK] Status:** `Image_Processing` branch is empty as of April 8. None of the Week 2 deliverables have been completed.

**Week 2 (still pending):**
- CLIP branch with OpenCLIP
- CNN branch with ResNet50
- Hybrid: `image_score = 0.70 * clip_score + 0.30 * cnn_score`
- Image embedding artifacts and score tables

**If Neil does not deliver by Week 3 start:** the 4-signal fusion formula cannot run. Gavin must decide whether to use the 2-signal fallback or delay integration.

**[UPDATED]** Week 3 comparison must include the new `Qwen3-VL-Embedding-8B` multimodal signal (not `gemini-embedding-2`) when benchmarking how much unique value the image branch adds on top of the multimodal embedding.

---

### Harsh — Text Retrieval and Reranking Lead

**Scope:** occasion-text embeddings, fragrance-text embeddings, shared-space multimodal embedding retrieval, text-to-fragrance retrieval, the research reranker.

**[UPDATED] Extended scope:** Harsh also owns the fragrance enrichment pipeline (`enrich.py`) because Karan's raw `embedding_text` was insufficient for retrieval quality.

**Model choices:**

| Role | Model | Benchmark |
|---|---|---|
| Text embedding | `gemini-embedding-001` | MTEB 68.32 (#1 English) |
| Multimodal embedding | `Qwen3-VL-Embedding-8B` | MMEB-V2 77.8 (#1) |
| Enrichment | `gemini-3-flash-preview` | — (structured output, fast) |
| Reranker | `gemini-3.1-pro-preview` | — |
| Evaluation judge | `gemini-2.5-pro` | — |

**Current state (April 12):**

Done:
- `src/vibescents/embeddings.py` — `GeminiEmbedder` (text), `Qwen3VLMultimodalEmbedder` (multimodal)
- `src/vibescents/pipelines.py` — `embed_text_frame`, `embed_occasions`, `retrieve_with_multimodal_query`
- `src/vibescents/reranker.py` — `GeminiReranker`
- `src/vibescents/benchmark.py` — `BenchmarkGenerator`, `consolidate_case_drafts`
- `src/vibescents/enrich.py` — LLM enrichment pipeline
- `data/vibescent_500.csv` — raw 500-row fragrance sample
- `data/vibescent_enriched_sample.csv` — enriched sample with `retrieval_text`

**Pending (all execution — no new code needed except Step 5):**

**Step 1 — Occasion embeddings + heatmap**
```bash
uv run vibescents embed-occasions \
  --input-json examples/occasions.json \
  --output-dir artifacts/occasions
```
Expected output: `artifacts/occasions/similarity_heatmap.png` showing formality gradient (casual ↔ summer party similar; black_tie ↔ wedding_guest similar; streetwear ↔ business_dinner dissimilar).

**Step 2 — Enrich full fragrance corpus (if not done)**
```bash
uv run python -m vibescents.enrich \
  --input-csv data/vibescent_500.csv \
  --output-csv data/vibescent_enriched.csv
```

**Step 3 — Embed fragrance corpus (run twice, on Colab for enriched)**
```bash
# Raw baseline (CPU, Gemini API)
uv run vibescents embed-csv \
  --input-csv data/vibescent_500.csv \
  --id-column fragrance_id \
  --text-column embedding_text \
  --output-dir artifacts/fragrance_raw

# Enriched (GPU required — run on Colab Pro)
uv run vibescents embed-csv \
  --input-csv data/vibescent_enriched.csv \
  --id-column fragrance_id \
  --text-column retrieval_text \
  --output-dir artifacts/fragrance_enriched
```

**Step 4 — Text-to-fragrance retrieval comparison (new ~30-line script)**
```python
from vibescents.similarity import cosine_similarity_matrix, top_k_indices
import numpy as np, pandas as pd

occasion_emb = np.load("artifacts/occasions/embeddings.npy")
for label, emb_path, meta_path in [
    ("RAW",      "artifacts/fragrance_raw/embeddings.npy",      "artifacts/fragrance_raw/metadata.csv"),
    ("ENRICHED", "artifacts/fragrance_enriched/embeddings.npy", "artifacts/fragrance_enriched/metadata.csv"),
]:
    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    scores = cosine_similarity_matrix(occasion_emb, emb)
    occasions = pd.read_csv("artifacts/occasions/metadata.csv")
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    for i, row in occasions.iterrows():
        top5 = top_k_indices(scores[i], 5)
        print(f"\n--- {row['occasion_id']} ---")
        for j in top5:
            print(f"  {scores[i,j]:.3f}  {meta.iloc[j]['name'] if 'name' in meta.columns else meta.iloc[j].iloc[1]}")
```

**Step 5 — Multimodal retrieval (on Colab, 3+ outfit images)**
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
5. Model justification — why `Qwen3-VL-Embedding-8B` over the original
6. Benchmark label quality — spot-check 5 cases
7. Failure analysis — where does retrieval break and what should Week 3 fix

**Verification checklist:**

- [ ] `uv run pytest` passes
- [ ] `artifacts/occasions/similarity_heatmap.png` shows formality gradient
- [ ] `artifacts/fragrance_enriched/embeddings.npy` exists with shape (N, ≤4096)
- [ ] Text retrieval returns different top-5 for `casual_day` vs `black_tie`
- [ ] Enriched retrieval is noticeably better than raw (or documented why not)
- [ ] Multimodal retrieval tested on 3+ outfit images
- [ ] `artifacts/benchmark_cases.json` has 20 schema-consistent labels
- [ ] `results/week2_report.md` covers heatmap, retrieval, multimodal, and failures
- [ ] All changes committed and pushed to `Text_Processing`

**Required outputs:** occasion embedding matrix, fragrance embedding matrix, multimodal embedding score table, text retrieval score table, reranker score table, `results/week2_report.md`.

**Success criteria:** similar occasions cluster together; enriched `retrieval_text` produces better retrieval than raw `embedding_text`; the `Qwen3-VL-Embedding-8B` multimodal signal changes results meaningfully when an image is added; reranking improves or matches the baseline on the benchmark.
ine on the benchmark.
