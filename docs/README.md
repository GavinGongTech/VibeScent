# VibeScent Docs

Last updated: April 7, 2026

This directory is the source of truth for planning.

## Core Docs

- `docs/project-plan.md`
- `docs/architecture.md`
- `docs/evaluation.md`

## Team Docs

- `docs/team/harsh.md`
- `docs/team/neil.md`
- `docs/team/darren.md`
- `docs/team/karan.md`
- `docs/team/gavin.md`

## Locked Decisions

- The project is optimizing for both research novelty and demo quality.
- The retrieval baseline is late fusion.
- The research layer is reranking only, not first-stage cross-attention.
- The text embedding baseline is `gemini-embedding-001`.
- The primary shared-space multimodal production signal is `gemini-embedding-2-preview`.
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
