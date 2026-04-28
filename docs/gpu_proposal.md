# GPU Resource Proposal — ScentAI Multimodal Recommender

**Prepared by:** Harsh Agarwal  
**Submitted to:** Vayun Malik, ML Co-Director  
**Date:** April 24, 2026  
**Project:** ScentAI — Outfit-to-Fragrance Multimodal Recommendation System

---

## Executive Summary

ScentAI's recommendation quality is currently bottlenecked by an **embedding space mismatch**: the fragrance corpus was pre-embedded with one model family, while inference-time queries use a different model family (Gemini API). Cosine similarity computed across two different vector spaces is geometrically meaningless, which directly degrades retrieval precision. GPU access resolves this by running a single unified OSS model stack — the Qwen3-VL framework — for both corpus embedding and query embedding, eliminating the mismatch entirely and replacing a text-only LLM reranker with a multimodal one that can actually see the outfit image during ranking.

---

## Current Architecture and Its Bottleneck

The live system uses a 4-channel late fusion pipeline:

| Channel | Current implementation | Problem |
|---------|------------------------|---------|
| Text retrieval | Gemini `embedding-001` query vs Qwen3-Embedding-8B corpus | **Cross-space mismatch** |
| Multimodal retrieval | Gemini Vision → text description → embed → vs Qwen corpus | **Two lossy steps + mismatch** |
| Image classification | CLIP ViT-B/32 (needs GPU) | Currently bypassed on CPU |
| LLM reranker | Gemini Pro (text-only) | Cannot see outfit image during reranking |

The root problem is that the fragrance corpus (35k rows) was embedded offline with `Qwen3-Embedding-8B`, but at inference time, query vectors come from `gemini-embedding-001`. These are two independently trained models with no shared embedding space — their cosine similarities are not comparable. The codebase itself acknowledges this: *"Cross-model retrieval is imperfect but functional at demo scale"* (`src/vibescents/engine.py`, line 52).

---

## Proposed Solution: Qwen3-VL Unified Stack

Alibaba Research released a unified multimodal retrieval framework in January 2026 (arXiv:2601.04720): **Qwen3-VL-Embedding-8B** paired with **Qwen3-VL-Reranker-8B**. Both models share the same architecture and embedding space, designed to work together.

### Why this stack specifically

**Qwen3-VL-Embedding-8B**
- MMEB-V2 score: **77.8% — #1 on the leaderboard** as of January 2026
- Embeds images and text in the same vector space natively — no intermediate text description step
- Already in the codebase (`src/vibescents/embeddings.py:185`) but not yet used at inference because there is no GPU available during query time

**Qwen3-VL-Reranker-8B**
- Purpose-built companion to the embedding model, trained on the same architecture
- Accepts outfit image + fragrance candidate list as joint input
- Ranks candidates based on actual visual features, not a text proxy
- This replaces the current Gemini Pro reranker, which is text-only and cannot observe the outfit

**Revised architecture with GPU:**

```
Outfit image + context
        ↓
Qwen3-VL-Embedding-8B  [GPU]
  ├── embed(image + context) → multimodal query vector
  └── embed(context text)   → text query vector
        ↓
cosine similarity vs Qwen3-VL-re-embedded corpus  [same vector space]
        ↓
top-20 candidates
        ↓
Qwen3-VL-Reranker-8B  [GPU]
  └── rerank(image, top-20 fragrances) → top-3 with visual reasoning
        ↓
CLIP ViT-B/32  [GPU, already implemented]
  └── formality / season / time classification → structured channel
```

Zero API calls at inference. Fully local. Reproducible.

---

## GPU Requirements

### Memory breakdown (all in bfloat16)

| Model | VRAM |
|-------|------|
| Qwen3-VL-Embedding-8B | ~16 GB |
| Qwen3-VL-Reranker-8B | ~16 GB |
| CLIP ViT-B/32 | ~151 MB |
| **Total** | **~34 GB** |

Any GPU with ≥ 40 GB VRAM satisfies the inference requirement. An A100 40 GB (standard on Colab Pro+) is the minimum. An A100 80 GB or equivalent gives headroom for larger batch sizes during the one-time corpus re-embedding job.

### Tasks and time estimates

| Task | Type | Estimated time | GPU needed |
|------|------|----------------|------------|
| Corpus re-embed with Qwen3-VL-Embedding-8B (35k rows, batch 64) | **One-time offline** | ~8 minutes on A100 | Yes |
| CLIP image scoring per request | Per-inference | ~0.2s | Yes |
| Qwen3-VL-Embedding query encoding per request | Per-inference | ~0.5s | Yes |
| Qwen3-VL-Reranker over top-20 candidates | Per-inference | ~1–2s | Yes |

The corpus re-embedding is a **one-time job**. After it runs, the resulting `.npy` file is committed to the repo and never needs to run again unless the dataset changes.

---

## Expected Quality Improvements

| Metric | Before (Gemini API + mismatch) | After (Qwen3-VL unified stack) |
|--------|-------------------------------|-------------------------------|
| Embedding consistency | Cross-space (meaningless cosine) | Single space (meaningful cosine) |
| Multimodal channel | Image → text → embed (lossy) | Direct image embedding (lossless) |
| Reranker input | Text description only | Raw image + text (multimodal) |
| API dependency at inference | Gemini Vision + Gemini Embed | None |
| MMEB-V2 benchmark coverage | Not evaluated (Gemini not on MMEB) | 77.8% (#1 open-source) |

---

## Budget Request

| Item | Cost estimate |
|------|--------------|
| Google Colab Pro+ (A100 40 GB, ~20 GPU hours) | ~$15–25 one-time |
| If university cluster is available | $0 |

Note: I currently have Colab Pro GPU credits that cover the one-time corpus re-embedding job. The per-inference GPU cost during the demo and presentation is negligible (single-digit requests, not production traffic). If cluster access is available for the demo session, API costs drop to zero.

**If funds are needed beyond existing credits:** $25–50 covers the full project lifecycle including buffer for experimentation.

---

## What Happens Without GPU

The system falls back to text-only retrieval (`_TEXT_ONLY_WEIGHTS = {"text": 0.80, "structured": 0.20}`) using Gemini embeddings against a mismatched corpus. This path works but produces noticeably weaker recommendations — semantically correct on occasion type and mood but often misses on fragrance character and visual aesthetic fit. The multimodal and image channels (which carry 55% of the fusion weight when active) are silently dropped.

---

## References

- Qwen3-VL-Embedding-8B: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
- Qwen3-VL-Reranker-8B: https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B
- arXiv:2601.04720 — Unified Framework paper: https://arxiv.org/abs/2601.04720
- MMEB Leaderboard: https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard
