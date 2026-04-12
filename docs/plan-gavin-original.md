# VibeScent — Gavin's Original Plan

Merged from: `project-plan.md`, `architecture.md`, `evaluation.md`, `README.md`, `team/*.md`
Last updated: April 7, 2026
This document is the untouched original. Do not edit it.

---

## Locked Decisions

- The project is optimizing for both research novelty and demo quality.
- The retrieval baseline is late fusion.
- The research layer is reranking only, not first-stage cross-attention.
- The text embedding baseline is `gemini-embedding-001`.
- The primary shared-space multimodal production signal is `gemini-embedding-2`.
- The image branch must compare `CLIP-only`, `CNN-only`, and `hybrid`.
- The hybrid starts as score fusion.
- Every fragrance has `retrieval_text` and `display_text`.
- Fragrance text is generated with LLM-assisted templates, not free-form blurbs.
- The benchmark uses 20 end-to-end cases.
- Benchmark labels are AI-assisted, but the primary score is metadata-based, not pure LLM judging.
- The primary multimodal reranker is `gemini-3.1-pro-preview`.

## Reporting Rules

- Do not present five examples as validation.
- Do not present AI-generated benchmark labels as human ground truth.
- Do not say `top-p retrieval`.
- Do not add architecture complexity before the baseline is benchmarked.
- Do not treat preview-model behavior as guaranteed; keep the documented fallback path live.

---

## Project Goal

Build a competition-grade system that takes an outfit image and occasion text, retrieves fragrance recommendations, and supports a flashy final demo with defensible ML choices.

## Success Criteria

The project succeeds if it does all of the following:

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

### Week 2

Deliver branch artifacts that can be integrated:

- fragrance dataset selected and cleaned
- fragrance descriptions generated
- occasion text retrieval working
- image retrieval working
- `gemini-embedding-2` retrieval working
- branch-level sanity checks completed

### Week 3

Integrate the branches and benchmark the full system:

- late-fusion baseline working end to end
- 20-case benchmark created
- reranker added on top of baseline
- ablation with and without `gemini-embedding-2` completed
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
- one short note describing assumptions and failure modes

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

Some of these may be inferred if the raw dataset does not provide them directly.

## Core Risks

- fragrance datasets may be incomplete or inconsistent
- richer text generation may introduce noise if not schema-controlled
- AI-assisted evaluation can become circular if the same model both writes and grades the benchmark
- the reranker can become impressive-looking but unmeasured if the benchmark is weak
- preview-model behavior or quotas may shift, so fallback paths must remain runnable

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
Recommended dimensionality: `1536`
Reason: it is the current official text embedding model in the Gemini API; it is available on the free tier; it is a better default for text-only semantic retrieval than older small sentence-transformer baselines.

#### Shared-space multimodal retrieval

Inputs: occasion text from the user, outfit image
Target: fragrance `retrieval_text`
Primary model: `gemini-embedding-2`
Recommended dimensionality: `1536`

Role in the stack: this is part of the primary production scoring pipeline; it provides a unified embedding-space signal across text and image inputs.

Why this model: it is Google's multimodal embedding model; it maps text and images into one embedding space; Google positions it for cross-modal semantic search and recommendation systems.

#### Image retrieval

Input: outfit image
Target: fragrance `retrieval_text`

Required branches: `CLIP-only`, `CNN-only`, `hybrid`
Recommended models: `OpenCLIP` for the CLIP branch, `ResNet50` for the CNN branch
Hybrid definition: score fusion first

Recommended initial rule:

```
hybrid_image_score = 0.70 * clip_score + 0.30 * cnn_score
```

Reason: CLIP should carry most of the semantic image-text matching load; CNN features can add useful visual structure without destabilizing the pipeline.

### Stage 2: Score Fusion

The first-stage ranking combines four signals:

- occasion-text score
- shared-space multimodal score
- image score
- structured attribute score

Recommended initial rule:

```
final_score = 0.30 * text_score + 0.25 * multimodal_score + 0.30 * image_score + 0.15 * structured_score
```

Structured score uses fragrance metadata such as: season fit, formality fit, day or night fit, fresh versus warm fit.

This layer should stay simple and inspectable.

Fallback rule (if `gemini-embedding-2` is unavailable):

```
final_score = 0.45 * text_score + 0.40 * image_score + 0.15 * structured_score
```

### Stage 3: Reranking

The reranker sees only the shortlist from Stage 2.
Primary reranker: `gemini-3.1-pro-preview`

Input per request: outfit image, occasion text, candidate fragrance `retrieval_text`, candidate fragrance structured metadata.

Output: per-candidate relevance score, short rationale, optional sub-scores for formality, season, freshness, and overall vibe fit.

The reranker is research scope. It does not replace the baseline unless it wins on the benchmark.

Why this model: it supports image, text, video, audio, and PDF inputs; it supports Batch API and structured outputs; Google positions it as stronger on thinking, reliability, and grounded multi-step workflows.

### Fragrance Representation

Every fragrance should have two text fields.

#### `retrieval_text`

Optimized for embeddings and ranking. Should be rich, but schema-controlled.

Recommended sections: brand and name, fragrance family, key accords, top/middle/base notes when available, likely season, likely occasion, likely formality, fresh versus warm, day versus night.

#### `display_text`

Optimized for the demo UI. Can be more stylistic and varied than `retrieval_text`, but should still be grounded in actual metadata.

### Multimodal Embedding Ablation

`gemini-embedding-2` is part of production scoring, but it still needs ablation.

Required comparisons:

- baseline without `gemini-embedding-2`
- baseline with `gemini-embedding-2`
- reranked system without `gemini-embedding-2`
- reranked system with `gemini-embedding-2`

The preview model stays in production only if it improves the benchmark enough to justify the added dependency.

### Why This Architecture

- it supports both novelty and reliability
- it keeps the retrieval path debuggable
- it gives Harsh a meaningful research reranker to own
- it preserves Neil's required CLIP, CNN, and hybrid comparison
- it makes the multimodal embedding model a real production signal instead of a side experiment
- it still preserves a fallback path if preview behavior becomes unstable

### Official References

- Gemini 3.1 Pro Preview: https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview
- Gemini Embedding 2 Preview: https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2
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

Why this model: it supports Batch API; it supports structured outputs; it accepts multimodal inputs, which matters for outfit-image benchmark cases.

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

Do not use the same model as both label writer and primary judge. Primary score should be metadata-based.

For each retrieved fragrance, score: accord match, note-family match, season match, formality match, day or night match, fresh versus warm match.

From this, compute:

- `attribute_match@3`
- `attribute_match@5`
- `neighborhood_hit@3`
- `neighborhood_hit@5`

### Secondary Scoring

Use an LLM judge only as a secondary evaluator.
Recommended judge: `gemini-2.5-pro`

Reason: it separates the judge from the label generator; it reduces direct self-confirmation bias.

Judge inputs: outfit image, occasion text, top candidates from the baseline, top candidates from the reranker.
Judge outputs: preferred shortlist, short rationale, confidence.

### Required Reporting

Report all of the following:

- text-only retrieval performance
- image-only retrieval performance
- multimodal embedding retrieval performance
- late-fusion baseline performance
- reranked performance
- with and without `gemini-embedding-2`
- benchmark cases where reranking helps
- benchmark cases where reranking hurts

### Presentation Rule

Call this what it is: an AI-assisted benchmark with metadata-based scoring. Do not call it human-labeled ground truth.

### Failure Conditions

The evaluation is not credible if:

- the same model generates labels and acts as the only judge
- the benchmark cases are not schema-consistent
- the system is only shown on hand-picked wins
- there is no baseline versus reranker comparison

---

## Team Roles

### Gavin — PM and Integration Lead

**Scope:** branch contracts, benchmark files, integration deadlines, final result reporting.

**Week 2:** lock the fragrance dataset choice; lock the shared schema; lock output paths and file formats.

**Week 3:** publish the 20-case benchmark; collect branch outputs; integrate the late-fusion baseline; make sure reranker comparisons are reproducible; publish ablation results with and without `gemini-embedding-2`.

**Week 4:** lock demo cases; lock final metrics; keep the final story consistent with the actual benchmark.

**Required outputs:** benchmark case file, evaluation rubric, integration checklist, final result summary.

**Guardrails:** do not accept vague progress reports; do not accept incompatible file formats; do not present AI-assisted labels as human labels; do not let the reranker replace the baseline without benchmark evidence; do not let the preview embedding model stay in production unless the ablation justifies it.

---

### Darren — Fragrance Data Lead

**Scope:** fragrance dataset selection, schema definition, data loading, normalization and missing-value handling.

**Week 2:** choose one primary fragrance dataset; document the schema; clean core fields; export the canonical fragrance table.

**Week 3:** improve coverage only if it does not break the canonical schema; support targeted enrichment where the benchmark exposes gaps.

**Required fields:** `fragrance_id`, `brand`, `name`, `notes`, `accords`, any available season, occasion, and gender metadata.

**Required outputs:** source dataset decision note, canonical fragrance table, data quality summary, missingness summary.

**Constraints:** do not merge multiple messy datasets before one clean table exists; do not let schema drift across branches.

**Success criteria:** the team has one stable fragrance table; downstream branches can consume it without manual cleanup.

---

### Karan — Fragrance Representation Lead

**Scope:** fragrance text generation, structured fragrance attributes, note-family and accord mapping.

**Week 2:** generate `retrieval_text`; generate `display_text`; compute structured fragrance attributes; validate that similar fragrances cluster together.

**Week 3:** refine representation quality based on benchmark failures; support reranker inputs with cleaner structured fields.

**Text strategy:**
- `retrieval_text`: rich, schema-controlled, optimized for ranking.
- `display_text`: more expressive, optimized for the demo.
- Generation method: LLM-assisted templates.

**Required outputs:** fragrance text generation script or notebook, final `retrieval_text`, final `display_text`, structured attribute table.

**Interfaces you depend on:** from Darren — canonical fragrance table; from Harsh — text embedding pipeline.

**Success criteria:** fragrance text is rich without becoming noisy; structured attributes improve scoring stability; the representation supports both retrieval and polished UI copy.

---

### Neil — Image Retrieval Lead

**Scope:** image-to-fragrance retrieval, image preprocessing, CLIP-only / CNN-only / hybrid comparison.

**Week 2:** build the CLIP branch with OpenCLIP; build the CNN branch with ResNet50; build the hybrid branch with score fusion; save image embeddings and score outputs; produce similarity and nearest-neighbor artifacts.

**Week 3:** integrate the winning image branch into the late-fusion baseline; compare image-only versus fused retrieval; compare CLIP/CNN/hybrid behavior against the shared-space `gemini-embedding-2` signal.

**Model choices:**
- CLIP branch: OpenCLIP
- CNN branch: ResNet50
- Hybrid: `image_score = 0.70 * clip_score + 0.30 * cnn_score`

**Required outputs:** image embedding artifacts, image-to-fragrance score table, CLIP versus CNN versus hybrid comparison, one short note on failure modes.

**Interfaces you depend on:** from Darren — cleaned fragrance table; from Karan — `retrieval_text` and structured fragrance attributes; from Gavin — benchmark case file.

**Success criteria:** visually similar outfits score similarly; image-only retrieval produces plausible fragrance candidates; the hybrid is clearly defined and benchmarked; the image branch still contributes unique value even when the multimodal embedding signal is added.

---

### Harsh — Text Retrieval and Reranking Lead

**Scope:** occasion-text embeddings, fragrance-text embeddings, shared-space multimodal embedding retrieval, text-to-fragrance retrieval, the research reranker.

Does not own: fragrance dataset sourcing, image feature extraction, final frontend polish.

**Week 2:**
- encode occasion descriptions with `gemini-embedding-001`
- encode fragrance `retrieval_text`
- build `gemini-embedding-2` retrieval for occasion text and outfit-image inputs
- compute text-to-fragrance similarity
- save embeddings and metadata
- produce similarity heatmaps and nearest-neighbor checks

**Week 3:**
- integrate text retrieval into the late-fusion baseline
- build the reranker prompt and JSON schema
- run baseline versus reranker comparison on the 20-case benchmark

**Model choices:**
- Primary text embedding: `gemini-embedding-001`
- Primary shared-space multimodal: `gemini-embedding-2`
- Reranker: `gemini-3.1-pro-preview`

**Required outputs:** occasion embedding matrix, fragrance embedding matrix, multimodal embedding score table, text retrieval score table, reranker score table, one short note describing wins, misses, and unstable cases.

**Interfaces you depend on:** from Darren — cleaned fragrance table; from Karan — `retrieval_text` and structured fragrance attributes; from Gavin — benchmark case file and evaluation schema.

**Success criteria:** similar occasions cluster together; text-only retrieval produces plausible fragrance candidates; the multimodal embedding signal improves the fused ranking or earns its place through ablation; reranking improves or matches the baseline on the benchmark.
nking or earns its place through ablation; reranking improves or matches the baseline on the benchmark.
