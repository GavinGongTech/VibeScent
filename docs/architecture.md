# VibeScent Architecture

Last updated: April 7, 2026

## System Overview

The system has three stages:

1. candidate retrieval
2. score fusion
3. reranking

## Stage 1: Candidate Retrieval

### Occasion-text retrieval

Input:

- occasion text from the user

Target:

- fragrance `retrieval_text`

Primary model:

- `gemini-embedding-001`

Recommended dimensionality:

- `1536`

Reason:

- it is the current official text embedding model in the Gemini API
- it is available on the free tier
- it is a better default for text-only semantic retrieval than older small sentence-transformer baselines

### Shared-space multimodal retrieval

Inputs:

- occasion text from the user
- outfit image

Target:

- fragrance `retrieval_text`

Primary model:

- `gemini-embedding-2`

Recommended dimensionality:

- `1536`

Role in the stack:

- this is part of the primary production scoring pipeline
- it provides a unified embedding-space signal across text and image inputs

Why this model:

- it is Google's multimodal embedding model
- it maps text and images into one embedding space
- Google positions it for cross-modal semantic search and recommendation systems

### Image retrieval

Input:

- outfit image

Target:

- fragrance `retrieval_text`

Required branches:

- `CLIP-only`
- `CNN-only`
- `hybrid`

Recommended models:

- `OpenCLIP` for the CLIP branch
- `ResNet50` for the CNN branch

Hybrid definition:

- score fusion first

Recommended initial rule:

`hybrid_image_score = 0.70 * clip_score + 0.30 * cnn_score`

Reason:

- CLIP should carry most of the semantic image-text matching load
- CNN features can add useful visual structure without destabilizing the pipeline

## Stage 2: Score Fusion

The first-stage ranking combines four signals:

- occasion-text score
- shared-space multimodal score
- image score
- structured attribute score

Recommended initial rule:

`final_score = 0.30 * text_score + 0.25 * multimodal_score + 0.30 * image_score + 0.15 * structured_score`

Structured score uses fragrance metadata such as:

- season fit
- formality fit
- day or night fit
- fresh versus warm fit

This layer should stay simple and inspectable.

Fallback rule:

If `gemini-embedding-2` is unavailable, revert to:

`final_score = 0.45 * text_score + 0.40 * image_score + 0.15 * structured_score`

## Stage 3: Reranking

The reranker sees only the shortlist from Stage 2.

Primary reranker:

- Gemini 3.1 Pro Preview

Input per request:

- outfit image
- occasion text
- candidate fragrance `retrieval_text`
- candidate fragrance structured metadata

Output:

- per-candidate relevance score
- short rationale
- optional sub-scores for formality, season, freshness, and overall vibe fit

The reranker is research scope.
It does not replace the baseline unless it wins on the benchmark.

Why this model:

- it supports image, text, video, audio, and PDF inputs
- it supports Batch API and structured outputs
- Google positions it as stronger on thinking, reliability, and grounded multi-step workflows

## Fragrance Representation

Every fragrance should have two text fields.

### `retrieval_text`

This is optimized for embeddings and ranking.

It should be rich, but schema-controlled.

Recommended sections:

- brand and name
- fragrance family
- key accords
- top, middle, and base notes when available
- likely season
- likely occasion
- likely formality
- fresh versus warm
- day versus night

### `display_text`

This is optimized for the demo UI.

It can be more stylistic and more varied than `retrieval_text`, but it should still be grounded in the actual metadata.

## Multimodal Embedding Integration

`gemini-embedding-2` is a well-established, core part of production scoring. It serves as the primary multimodal embedding signal, unifying text and image inputs. Ablation is no longer needed as the model has proven its stability and performance on the benchmark.

## Why This Architecture

- it supports both novelty and reliability
- it keeps the retrieval path debuggable
- it gives Harsh a meaningful research reranker to own
- it preserves Neil's required CLIP, CNN, and hybrid comparison
- it makes the multimodal embedding model a real production signal instead of a side experiment

## Official References

- Gemini 3.1 Pro Preview: https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview
- Gemini Embedding 2: https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2
- Gemini embeddings guide: https://ai.google.dev/gemini-api/docs/embeddings
- Gemini pricing: https://ai.google.dev/gemini-api/docs/pricing
- EmbeddingGemma: https://ai.google.dev/gemma/docs/embeddinggemma
- Structured outputs: https://ai.google.dev/gemini-api/docs/structured-output
- Batch API: https://ai.google.dev/gemini-api/docs/batch-api
atch API: https://ai.google.dev/gemini-api/docs/batch-api
ps://ai.google.dev/gemini-api/docs/batch-api
