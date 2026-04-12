# VibeScent — Full Project Reference

Last updated: April 12, 2026
Source of truth for architecture, evaluation, timeline, team roles, and locked decisions.

---

## Project Goal

Build a competition-grade system that takes an outfit image and occasion text, retrieves fragrance recommendations, and supports a polished final demo with defensible ML choices.

### Success Criteria

- Returns fragrance recommendations that feel coherent and explainable
- Beats simple text-only and image-only baselines on the 20-case benchmark
- Produces stable outputs across repeated runs
- Supports a polished frontend demo
- Delivers a research story that goes beyond prompt glue (the reranker must be benchmarked, not just added)

---

## Team and Ownership

| Role | Owner | Branch |
|---|---|---|
| PM and Integration Lead | Gavin | `main` |
| Text Retrieval and Reranking Lead | Harsh | `Text_Processing` |
| Image Retrieval Lead | Neil | `Image_Processing` |
| Fragrance Representation Lead | Karan | `Text_Processing` (aligned with Harsh) |
| Fragrance Data Lead | Darren | `main` / shared |

### Gavin — PM and Integration Lead

Gavin owns branch contracts, benchmark files, integration deadlines, and final result reporting.

**Week 2:** lock dataset choice ✓, lock shared schema ✓, lock output paths and file formats ✓.

**Week 3:** publish the 20-case benchmark; collect branch outputs; integrate the late-fusion baseline; ensure reranker comparisons are reproducible; publish ablation results with and without `Qwen3-VL-Embedding-8B`.

**Week 4:** lock demo cases; lock final metrics; keep the final story consistent with the actual benchmark.

**Required outputs:** benchmark case file, evaluation rubric, integration checklist, final result summary.

**Guardrails:**
- Do not accept vague progress reports
- Do not accept incompatible file formats
- Do not present AI-assisted labels as human labels
- Do not let the reranker replace the baseline without benchmark evidence
- Do not let the preview embedding model stay in production unless ablation justifies it

**Immediate action required (April 12):** Neil's `Image_Processing` branch is empty. If Neil cannot deliver CLIP/CNN/hybrid by Week 3 start, Gavin must either assign image retrieval to another member or officially adopt the 2-signal fallback formula for integration.

---

## Architecture

### System Overview

Three-stage pipeline:

1. **Candidate retrieval** — text, image, and multimodal signals produce ranked candidates independently
2. **Score fusion** — signals are combined with a weighted rule into a single shortlist
3. **Reranking** — a stronger multimodal model reorders the shortlist and provides rationale

### Stage 1: Candidate Retrieval

#### Text Retrieval (Occasion text → fragrance `retrieval_text`)

| Field | Value |
|---|---|
| Model | `voyage-3-large` |
| Provider | Voyage AI (`VOYAGE_API_KEY`) |
| Dimensionality | 1024 |
| MTEB English score | 68.32 (#1) |
| Input type (query) | `"query"` |
| Input type (document) | `"document"` |

Why voyage-3-large over gemini-embedding-001: same MTEB top position, no Google API quota dependency for the text path, simpler billing separation from the Gemini reranker usage.

#### Multimodal Retrieval (Occasion text + outfit image → fragrance `retrieval_text`)

| Field | Value |
|---|---|
| Model | `Qwen3-VL-Embedding-8B` |
| Provider | Local GPU (HuggingFace) |
| Dimensionality | up to 4096 (use 1536 for cross-modal compatibility) |
| MMEB-V2 score | 77.8 (#1 overall) |
| VRAM required | ~16 GB float16 (Colab Pro A100) |

Why Qwen3-VL over gemini-embedding-2: 9-point gap on MMEB-V2 (77.8 vs 68.9), which is the cross-modal retrieval benchmark directly relevant to this task. Natively multimodal via a unified VLM transformer (Qwen3-VL backbone), not dual-encoder contrastive alignment. Runs fully locally — no API quota risk during demo.

All fragrance document embeddings must be pre-computed and cached as `.npy` before the demo. The model cannot run live on CPU.

GPU setup:
```bash
pip install torch transformers>=4.57.0 qwen-vl-utils>=0.0.14 accelerate
# or: uv sync --extra gpu
```

#### Image Retrieval (Outfit image → fragrance `retrieval_text`)

| Branch | Model |
|---|---|
| CLIP-only | OpenCLIP |
| CNN-only | ResNet50 |
| Hybrid | `image_score = 0.70 * clip_score + 0.30 * cnn_score` |

**Status: BLOCKED.** Neil's `Image_Processing` branch is empty as of April 8. None of the Week 2 deliverables have been completed. This blocks the 4-signal fusion formula.

### Stage 2: Score Fusion

Primary formula (4 signals):
```
final_score = 0.30 * text_score + 0.25 * multimodal_score + 0.30 * image_score + 0.15 * structured_score
```

Structured score is derived from fragrance metadata: season fit, formality fit, day/night fit, fresh vs warm fit.

Fallback formula if image branch is missing:
```
final_score = 0.60 * text_score + 0.40 * multimodal_score
```

Fallback formula if multimodal is also unavailable:
```
final_score = 0.70 * text_score + 0.30 * structured_score
```

This layer must stay simple and inspectable. Do not add learned weights until the benchmark is running.

### Stage 3: Reranking

| Field | Value |
|---|---|
| Model | `gemini-3.1-pro-preview` |
| Scope | Research — does not replace baseline unless it wins on benchmark |
| Input | outfit image, occasion text, candidate `retrieval_text`, candidate structured metadata |
| Output | per-candidate relevance score, short rationale, optional sub-scores (formality, season, freshness, vibe) |

The reranker is wired via `src/vibescents/reranker.py` (`GeminiReranker`). It sees only the shortlist from Stage 2 (top-k, default 10).

---

## Fragrance Representation

### Two Required Text Fields

Every fragrance has two distinct text fields:

**`retrieval_text`** — optimized for embedding and ranking. Schema-controlled. Generated by `enrich.py`.

Format:
```
Brand: {brand} | Name: {name}
Accords: {main_accords}
Top: {top_notes} | Heart: {middle_notes} | Base: {base_notes}
Season: {likely_season} | Best for: {likely_occasion}
Formality: {low/medium/high} | Character: {character_tags}
Vibe: {vibe_sentence}
```

Raw note concatenation (e.g. `bergamot, lemon | jasmine, rose | sandalwood, vanilla`) is **not** a valid `retrieval_text`. It operates at ingredient level and cannot bridge the semantic gap to occasion descriptions.

**`display_text`** — optimized for the demo UI. Can be more expressive and stylistic, but must remain grounded in actual metadata. Still Karan's responsibility to generate.

### Required Shared Schema

Every fragrance row must contain:

| Field | Source |
|---|---|
| `fragrance_id` | Darren |
| `brand` | Darren |
| `name` | Darren |
| `notes` | Darren |
| `accords` | Darren |
| `season_tags` | Darren / enriched |
| `occasion_tags` | Darren / enriched |
| `formality_score` | Karan / enriched |
| `fresh_warm_score` | Karan / enriched |
| `day_night_score` | Karan / enriched |
| `retrieval_text` | Harsh (enrich.py) |
| `display_text` | Karan |

Enriched fields generated by `enrich.py` using `gemini-3-flash-preview`:

- `likely_season`
- `likely_occasion`
- `formality` (0.0–1.0)
- `fresh_warm` (0.0–1.0)
- `day_night` (0.0–1.0)
- `character_tags`
- `vibe_sentence`

---

## Model Table

| Role | Model | Provider | Key Metric |
|---|---|---|---|
| Text embedding | `voyage-3-large` | Voyage AI | MTEB 68.32 (#1 English) |
| Multimodal embedding | `Qwen3-VL-Embedding-8B` | Local GPU | MMEB-V2 77.8 (#1) |
| Enrichment LLM | `gemini-3-flash-preview` | Google | structured output, fast, 5 RPM free |
| Reranker | `gemini-3.1-pro-preview` | Google | multimodal, batch API, structured output |
| Evaluation judge | `gemini-2.5-pro` | Google | separated from label generator |

---

## Evaluation Plan

### Benchmark Structure

20 end-to-end cases. Each case contains:

- one outfit image
- one occasion description
- target style attributes
- acceptable fragrance neighborhoods
- disallowed traits

### Label Generation

AI-assisted. Primary label generator: `gemini-3.1-pro-preview`.

Method: looped structured-output requests, 3 independent generations per case, keep only cases with strong agreement across runs.

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

### Primary Scoring (metadata-based)

Do not use the same model as both label writer and primary judge.

For each retrieved fragrance score: accord match, note-family match, season match, formality match, day/night match, fresh vs warm match.

Metrics:

- `attribute_match@3`
- `attribute_match@5`
- `neighborhood_hit@3`
- `neighborhood_hit@5`

### Secondary Scoring (LLM judge)

Judge: `gemini-2.5-pro` (different from `gemini-3.1-pro-preview` label generator — avoids self-confirmation bias).

Judge inputs: outfit image, occasion text, top candidates from baseline, top candidates from reranker.
Judge outputs: preferred shortlist, short rationale, confidence.

### Required Reporting

- Text-only retrieval performance
- Image-only retrieval performance
- `Qwen3-VL-Embedding-8B` multimodal retrieval performance
- Late-fusion baseline performance (all signals)
- Reranked performance
- Ablation: with and without `Qwen3-VL-Embedding-8B`
- Raw `embedding_text` vs enriched `retrieval_text` retrieval comparison
- Benchmark cases where reranking helps
- Benchmark cases where reranking hurts

### Presentation Rules

- Call this what it is: an AI-assisted benchmark with metadata-based scoring
- Do not call it human-labeled ground truth
- Do not present five examples as validation
- Do not present AI-generated benchmark labels as human ground truth
- Do not say `top-p retrieval`
- Do not add architecture complexity before the baseline is benchmarked
- Do not treat preview-model behavior as guaranteed; keep the documented fallback path live
- Do not call raw note concatenation a retrieval text

### Failure Conditions

The evaluation is not credible if:

- the same model generates labels and acts as the only judge
- the benchmark cases are not schema-consistent
- the system is only shown on hand-picked wins
- there is no baseline versus reranker comparison

---

## Timeline

### Week 2 (deadline: April 12, 2026)

| Deliverable | Owner | Status |
|---|---|---|
| Fragrance dataset selected and cleaned | Darren | ✓ `vibescent_500.csv` |
| Shared schema locked | Gavin | ✓ |
| Fragrance enrichment pipeline | Harsh | ✓ `enrich.py` |
| Occasions embeddings + heatmap | Harsh | pending |
| Raw fragrance embeddings | Harsh | pending |
| Enriched fragrance embeddings | Harsh | pending |
| Benchmark labels (20 cases) | Harsh / Gavin | pending |
| RAW vs ENRICHED retrieval comparison | Harsh | pending |
| `display_text` generation | Karan | pending |
| CLIP branch | Neil | **MISSING** |
| CNN branch | Neil | **MISSING** |
| Hybrid branch | Neil | **MISSING** |
| `results/week2_report.md` | Harsh | in progress |

### Week 3

- Late-fusion baseline working end to end
- 20-case benchmark published by Gavin
- Reranker added on top of baseline
- Ablation: with and without `Qwen3-VL-Embedding-8B`
- Baseline vs reranker comparison completed
- Karan refines representation based on benchmark failures

### Week 4

- Final demo flow locked
- Final benchmark numbers locked
- Demo cases chosen
- Presentation deck written around real results

---

## Artifact Contract

Every branch must produce:

- Reproducible code entrypoint (CLI command or notebook)
- Saved outputs under a predictable directory path
- Metadata file that maps rows to source items
- `results/week2_report.md` covering: heatmap interpretation, retrieval comparison, multimodal ablation, benchmark label quality, failure analysis, model justification

Formats:

- Embeddings: `.npy`
- Tabular metadata: `.csv`
- Configuration: `.json`
- Figures: `.png`

---

## Occasions Corpus

8 occasion descriptions covering the full formality spectrum:

| ID | Description |
|---|---|
| `casual_day` | Relaxed daytime — weekend errands, lunch with friends |
| `creative_office` | Creative industry office — expressive but professional |
| `business_dinner` | Business dinner — polished, confident, evening |
| `wedding_guest` | Wedding guest — elegant, celebratory, occasion-appropriate |
| `black_tie` | Black tie evening — formal, refined, commanding |
| `streetwear_night` | Streetwear night out — bold, urban, youthful |
| `summer_party` | Summer outdoor party — fresh, playful, social |
| `editorial` | Fashion editorial — avant-garde, artistic, attention-commanding |

Expected heatmap: formality gradient visible (casual_day ↔ summer_party similar; black_tie ↔ wedding_guest similar; streetwear_night ↔ business_dinner dissimilar).

---

## References

- Voyage AI: https://docs.voyageai.com
- Qwen3-VL-Embedding-8B HuggingFace: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
- MMEB Leaderboard: https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard
- Gemini 3.1 Pro Preview: https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview
- Gemini structured outputs: https://ai.google.dev/gemini-api/docs/structured-output
- Gemini Batch API: https://ai.google.dev/gemini-api/docs/batch-api
