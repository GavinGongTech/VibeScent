# VibeScent Project Plan

Last updated: April 7, 2026

## Goal

Build a competition-grade system that takes an outfit image and occasion text, retrieves fragrance recommendations, and supports a flashy final demo with defensible ML choices.

## Success Criteria

The project succeeds if it does all of the following:

- returns fragrance recommendations that feel coherent
- beats simple text-only and image-only baselines
- produces stable outputs across repeated runs
- supports a polished frontend demo
- gives the team a research story that is more than prompt glue

## Delivery Strategy

The project is split into two layers:

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

## Timeline

### Week 2

Deliver branch artifacts that can be integrated:

- fragrance dataset selected and cleaned
- fragrance descriptions generated
- occasion text retrieval working
- image retrieval working
- `gemini-embedding-2-preview` retrieval working
- branch-level sanity checks completed

### Week 3

Integrate the branches and benchmark the full system:

- late-fusion baseline working end to end
- 20-case benchmark created
- reranker added on top of baseline
- ablation with and without `gemini-embedding-2-preview` completed
- baseline versus reranker comparison completed

### Week 4

Stabilize and present:

- final demo flow locked
- final benchmark numbers locked
- demo cases chosen
- presentation deck written around real results

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
