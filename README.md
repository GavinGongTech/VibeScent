# VibeScent

A multimodal fragrance recommendation system that matches outfit images to luxury fragrances using text retrieval, zero-shot image classification, score fusion, and LLM pointwise reranking.

## Architecture

```
Outfit Image ─┬─► CLIP zero-shot classifier ─► formality / season / time attributes
               │
               └─► Context (event, mood, time) ─┬─► all-MiniLM-L6-v2 (text)
                                                  └─► Structured attribute scoring
                                                          │
                                               Score Fusion (weighted sum)
                                                          │
                                                   Top-20 Candidates
                                                          │
                                           Gemma-3-27B Pointwise Reranker
                                                          │
                                                  Top-3 Fragrances
                                                          │
                              FastAPI ML backend (port 8000) ◄── Next.js frontend (port 3000)
                                                          │
                              FastAPI Scraper API (port 8001) ─► Pricing + purchase links
```

## Components

| Directory | Owner | Description |
|-----------|-------|-------------|
| `src/vibescents/` | Harsh | Core Python package — backend API, engine, embeddings, fusion, image scoring, reranker |
| `app/`, `components/`, `lib/` | Darren | Next.js 14 frontend — luxury editorial UI |
| `notebooks/` | Harsh | Colab/Kaggle pipelines for offline preprocessing |
| `docs/` | All | Project planning and architecture documentation |
| `tests/` | Harsh | pytest suite (16 test files) |
| `data/` | All | Fragrance datasets, enriched CSV, and LTR labeled datasets |
| `artifacts/` | Harsh | Pre-computed embeddings and pipeline outputs |

## Quick Start

### Prerequisites

- Python 3.11+ (managed via `uv`)
- Bun (frontend package manager)
- No local GPU required — CPU-native pipeline. Uses NVIDIA NIM API for reranking.

### Setup

```bash
# Python dependencies
uv sync --extra dev

# Frontend dependencies
bun install
```

Create a `.env` file in the project root:

```bash
# Optional — enables live pricing and purchase links on result cards
SERPAPI_KEY=<your_serpapi_key>

# Required for Reranking and Enrichment — uses build.nvidia.com
NVIDIA_API_KEY=<your_nvidia_api_key>

# Optional — faster HuggingFace model downloads
HF_TOKEN=<your_hf_token>
```

### Generate corpus embeddings (one-time)

Before first run, the MiniLM corpus embeddings must be generated:

1. Run the offline embedding pipeline: `uv run artifacts/re_embed_corpus_cpu.py`
2. Ensure `artifacts/minilm_corpus/embeddings.npy` is created.

The engine will not start without these files. 

### Run everything

```bash
./start.sh
```

This starts all three services in order and waits for each to be healthy:
- ML backend → `http://localhost:8000`
- Scraper API → `http://localhost:8001`
- Next.js frontend → `http://localhost:3000`

Open `http://localhost:3000/demo` to use the app.

To run services individually:

```bash
uv run uvicorn "vibescents.backend_app:create_configured_app" --factory --host 0.0.0.0 --port 8000
uv run uvicorn vibescents.scraper_app:app --host 0.0.0.0 --port 8001
bun run dev:web
```

## Inference Pipeline

### Online (per request, ~1–4 seconds)

| Channel | Weight | Model | Coverage |
|---|---|---|---|
| Image | 0.450 | CLIP ViT-L/14 | Full corpus (36K) |
| Text | 0.275 | all-MiniLM-L6-v2 | Full corpus (36K) |
| Structured | 0.275 | Arithmetic (no model) | Full corpus (36K) |

Each channel produces a per-fragrance score array. All three are min-max normalized to [0, 1] independently, then combined via weighted sum. The exact weights were learned via Logistic Regression on a synthetic Learning-to-Rank (LTR) dataset, anchored by a strong domain prior for the image channel.

**Reranking Phase**: The top 20 candidates from score fusion are sent concurrently to the `google/gemma-3-27b-it` model via the Nvidia NIM API. Gemma acts as a pointwise zero-shot judge, assessing the match and providing an explanation before re-sorting the final top 3.

### Offline (one-time)

- **Enrichment**: LLM (Qwen3.5-8B or Gemini flash fallback) generates `formality`, `day_night`, `fresh_warm`, `vibe_sentence`, and other fields for each fragrance
- **Corpus embedding**: `all-MiniLM-L6-v2` embeds all enriched `retrieval_text` strings → `artifacts/minilm_corpus/embeddings.npy`

## Project Layout

```text
src/vibescents/
  backend_app.py          # FastAPI server (port 8000) — /healthz, /recommend
  engine.py               # VibeScoreEngine — 3-channel fusion, lazy model loading
  embeddings.py           # SentenceTransformerEmbedder
  fusion.py               # min_max_normalize + weighted fuse_scores + LTR weights
  image_scoring.py        # CLIPImageScorer — zero-shot outfit classification
  structured_scorer.py    # compute_structured_scores — context → attribute → distance
  query.py                # context_to_query_string — expands event/mood/time to rich phrases
  reranker.py             # NvidiaNIMReranker — Async Gemma-3-27B pointwise judge
  scraper_app.py          # FastAPI server (port 8001) — /search
  perfume_scraper.py      # SerpAPI Google Shopping scraper
  schemas.py              # Pydantic models: RecommendRequest, FragranceRecommendation, etc.
  settings.py             # Configuration (artifact paths, model names)

app/
  page.tsx                # Landing page (/)
  demo/page.tsx           # Demo page (/demo) — two-column: inputs + results
  model/page.tsx          # How It Works (/model) — pipeline + methodology
  api/recommend/route.ts  # POST /api/recommend — orchestrates ML backend + scraper

components/
  demo/                   # OutfitUploader, ContextForm, SubmitButton, ResultsPanel, FragranceCard
  ...

notebooks/
  ...

data/                     # vibescent_enriched.csv — 36K fragrances with all enrichment fields
  ltr_labels.csv          # Synthetic LTR labels used to learn fusion weights
artifacts/
  minilm_corpus/          # embeddings.npy
```

## Fusion Formula

```
fused = 0.450 × norm(image_scores)
      + 0.275 × norm(text_scores)
      + 0.275 × norm(structured_scores)
```

`norm()` is min-max normalization over the full candidate array. When a channel is unavailable, the engine uses dynamic weights that re-normalize the remaining channels to sum to 1.0.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `NVIDIA_API_KEY` | Yes | Required for Gemma-3 pointwise reranking and offline data enrichment. |
| `SERPAPI_KEY` | No | Google Shopping scraper for live pricing. Without it, price shows as "Price Unavailable" |
| `HF_TOKEN` | No | Faster downloads of models from HuggingFace Hub |
