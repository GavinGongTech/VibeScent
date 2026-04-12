# Harsh

Role: Text Retrieval And Reranking Lead

Last updated: April 7, 2026

## Scope

You own:

- occasion-text embeddings
- fragrance-text embeddings
- shared-space multimodal embedding retrieval
- text-to-fragrance retrieval
- the research reranker

You do not own:

- fragrance dataset sourcing
- image feature extraction
- final frontend polish

## Main Deliverables

### Week 2

- encode occasion descriptions with `gemini-embedding-001`
- encode fragrance `retrieval_text`
- build `gemini-embedding-2` retrieval for occasion text and outfit-image inputs
- compute text-to-fragrance similarity
- save embeddings and metadata
- produce similarity heatmaps and nearest-neighbor checks

### Week 3

- integrate text retrieval into the late-fusion baseline
- build the reranker prompt and JSON schema
- run baseline versus reranker comparison on the 20-case benchmark

## Model Choices

Primary text embedding model:

- `gemini-embedding-001`

Primary shared-space multimodal model:

- `gemini-embedding-2`

Reranker:

- Gemini 3.1 Pro Preview

## Required Outputs

- occasion embedding matrix
- fragrance embedding matrix
- multimodal embedding score table
- text retrieval score table
- reranker score table
- one short note describing wins, misses, and unstable cases

## Interfaces You Depend On

From Darren:

- cleaned fragrance table

From Karan:

- `retrieval_text`
- structured fragrance attributes

From Gavin:

- benchmark case file
- evaluation schema

## Success Criteria

- similar occasions cluster together
- text-only retrieval produces plausible fragrance candidates
- the multimodal embedding signal provides a strong, established baseline for the fused ranking
- reranking improves or matches the baseline on the benchmark
ark
