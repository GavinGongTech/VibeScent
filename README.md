# VibeScent

A multimodal fragrance recommendation system that matches outfit images to fragrances using text retrieval, image classification, and score fusion.

## Architecture

```
Outfit Image ─┬─► Image Classifier (CNN/CLIP) ─► vibe labels + 512-d embedding
               │
               └─► Text Pipeline (Gemini/Voyage) ─► fragrance retrieval + reranking
                                                          │
               Context (event, mood, time) ───────────────┘
                                                          │
                                                    Score Fusion ─► Top-K Fragrances
                                                          │
                                                    FastAPI Backend ◄── Next.js Frontend
```

## Components

| Directory | Owner | Description |
|-----------|-------|-------------|
| `src/vibescents/` | Harsh | Core Python package — backend API, embeddings, fusion, image scoring, reranking |
| `models/`, `experiments/`, `src/inference.py` | Neil | CNN/CLIP outfit classifiers (3 architectures) |
| `embed_fragrances.py`, `merge_datasets.py`, `scent_clusters.py` | Karan | Fragrance data unification, embedding, clustering |
| `app/`, `components/`, `lib/` | Darren | Next.js frontend with luxury editorial UI |
| `notebooks/` | Harsh | Colab pipelines for Weeks 2–4 |
| `docs/` | All | Project planning and member documentation |

## Quick Start

### Backend (Python)

```bash
uv sync --extra dev
```

Set environment variables:
```bash
GEMINI_API_KEY=...      # Required for text pipeline
VOYAGE_API_KEY=...      # Optional
HF_TOKEN=...            # Optional (Hugging Face)
```

Run the API server:
```bash
uv run python -m vibescents.backend_app
```

### Frontend (Next.js)

```bash
cd app/  # or wherever the Next.js root is
bun install
bun run dev
```

### Image Processing

```bash
pip install -r requirements.txt

# Generate pseudo-labels
python src/clip_zero_shot.py

# Run inference
python src/inference.py --model hybrid --checkpoint checkpoints/hybrid/best.pt --input path/to/outfit.jpg
```

### Fragrance Pipeline

```bash
python merge_datasets.py        # Unify 5 data sources
python embed_fragrances.py      # Generate embeddings (requires GPU)
python scent_clusters.py        # K-Means clustering
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Liveness probe |
| POST | `/recommend` | Accept outfit image (base64) + context, return ranked fragrances |

## Testing

```bash
uv run python -m pytest tests/ -v
```
