# VibeScent

A multimodal fragrance recommendation system that matches outfit images to luxury fragrances using text retrieval, zero-shot image classification, and score fusion.

## Architecture

```
Outfit Image ─┬─► CLIP zero-shot classifier ─► formality / season / time attributes
               │
               ├─► Qwen3-VL-Embedding-8B (multimodal) ─► joint image+text embedding
               │
               └─► Context (event, mood, time) ─┬─► Qwen3-VL-Embedding-8B (text)
                                                  └─► Structured attribute scoring
                                                          │
                                               Score Fusion (weighted sum)
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
| `src/vibescents/` | Harsh | Core Python package — backend API, engine, embeddings, fusion, image scoring |
| `app/`, `components/`, `lib/` | Darren | Next.js 14 frontend — luxury editorial UI |
| `notebooks/` | Harsh | Colab/Kaggle pipelines for offline preprocessing |
| `docs/` | All | Project planning and architecture documentation |
| `tests/` | Harsh | pytest suite (16 test files) |
| `data/` | All | Fragrance datasets and enriched CSV |
| `artifacts/` | Harsh | Pre-computed embeddings and pipeline outputs |

## Quick Start

### Prerequisites

- Python 3.11+ (managed via `uv`)
- Bun (frontend package manager)
- GPU recommended — 16 GB VRAM for Qwen3-VL text/multimodal channels; CPU fallback runs CLIP + structured channels only

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

# Optional — faster HuggingFace model downloads
HF_TOKEN=<your_hf_token>
```

### Generate corpus embeddings (one-time, GPU required)

Before first run, the Qwen3-VL corpus embeddings must be generated:

1. Open `notebooks/harsh_offline_pipeline.ipynb` in Colab or Kaggle (A100 recommended)
2. Run all cells — this embeds ~36K enriched fragrances with Qwen3-VL-Embedding-8B
3. Download `artifacts/qwen3vl_corpus/embeddings.npy` and the paired `metadata.csv` into the project

The engine will not start without these files. The offline pipeline also generates the enriched metadata (`data/vibescent_enriched.csv`) if it doesn't exist.

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
| Text | 0.30 | Qwen3-VL-Embedding-8B | Full corpus (36K) |
| Multimodal | 0.25 | Qwen3-VL-Embedding-8B | Tier B only (2K enriched) |
| Image | 0.30 | CLIP (openai/clip-vit-base-patch32) | Tier B only |
| Structured | 0.15 | Arithmetic (no model) | Tier B only |

Each channel produces a per-fragrance score array. All four are min-max normalized to [0, 1] independently, then combined via weighted sum. Missing channels (e.g. no GPU for Qwen3-VL) are redistributed with pre-tuned fallback weight sets.

### Offline (one-time, GPU required)

- **Enrichment**: LLM (Qwen3.5-8B or Gemini flash fallback) generates `formality`, `day_night`, `fresh_warm`, `vibe_sentence`, and other fields for each fragrance
- **Corpus embedding**: Qwen3-VL-Embedding-8B embeds all enriched `retrieval_text` strings → `artifacts/qwen3vl_corpus/embeddings.npy`

## Project Layout

```text
src/vibescents/
  backend_app.py          # FastAPI server (port 8000) — /healthz, /recommend
  engine.py               # VibeScoreEngine — 4-channel fusion, lazy model loading
  embeddings.py           # Qwen3VLMultimodalEmbedder
  fusion.py               # min_max_normalize + weighted fuse_scores
  image_scoring.py         # CLIPImageScorer — zero-shot outfit classification
  image_scoring.py        # ImageHeadProbabilities dataclass
  structured_scorer.py    # compute_structured_scores — context → attribute → distance
  query.py                # context_to_query_string — expands event/mood/time to rich phrases
  reranker.py             # Cross-encoder reranker (research, not in production path)
  scraper_app.py          # FastAPI server (port 8001) — /search
  perfume_scraper.py      # SerpAPI Google Shopping scraper
  schemas.py              # Pydantic models: RecommendRequest, FragranceRecommendation, etc.
  settings.py             # Configuration (artifact paths, model names)
  enrich.py               # LLM enrichment pipeline
  pipelines.py            # Offline pipeline orchestration

app/
  page.tsx                # Landing page (/)
  demo/page.tsx           # Demo page (/demo) — two-column: inputs + results
  model/page.tsx          # How It Works (/model) — pipeline + methodology
  api/recommend/route.ts  # POST /api/recommend — orchestrates ML backend + scraper

components/
  demo/                   # OutfitUploader, ContextForm, SubmitButton, ResultsPanel, FragranceCard
  landing/                # Hero, AboutSection
  model/                  # PipelineVisual, DataVisual
  layout/                 # Navbar, Footer
  ui/                     # Button, Tag, GoldDivider

lib/
  types.ts                # Shared TypeScript types (RecommendRequest, FragranceRecommendation, ContextInput)
  recommend.ts            # Client-side /api/recommend fetch wrapper

notebooks/
  harsh_offline_pipeline.ipynb   # Main preprocessing pipeline — enrichment + embedding (GPU)
  harsh_week5_qwen3vl.ipynb      # Week 5 Qwen3-VL multimodal embedding experiments

tests/                    # pytest suite — 16 test files
data/                     # vibescent_enriched.csv — 36K fragrances with all enrichment fields
artifacts/
  qwen3vl_corpus/         # embeddings.npy + metadata.csv (generated by offline pipeline)
  occasions/              # Week 2 occasion embeddings and similarity heatmap
```

## Fusion Formula

```
fused = 0.30 × norm(text_scores)
      + 0.25 × norm(multimodal_scores)
      + 0.30 × norm(image_scores)
      + 0.15 × norm(structured_scores)
```

`norm()` is min-max normalization over the full candidate array. When a channel is unavailable, the engine falls back to pre-tuned weight sets that re-normalize the remaining channels to sum to 1.0.

## API Reference

### ML Backend (`localhost:8000`)

| Method | Path | Body | Description |
|--------|------|------|-------------|
| GET | `/healthz` | — | Liveness probe |
| POST | `/recommend` | `RecommendRequest` | Outfit image + context → ranked fragrances |

### Scraper API (`localhost:8001`)

| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/search` | `{perfumes: string[], budget: float}` | Perfume names + budget → pricing + links |

## Testing

```bash
uv run python -m pytest tests/ -v
```

If the venv has a Python version conflict:

```bash
mv .venv .venv_old
UV_PROJECT_ENVIRONMENT=.venv_new uv sync --extra dev
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `SERPAPI_KEY` | No | Google Shopping scraper for live pricing. Without it, price shows as "Price Unavailable" |
| `HF_TOKEN` | No | Faster downloads of Qwen3-VL models from HuggingFace Hub |
