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
| `notebooks/` | Harsh | Colab pipelines for Weeks 2-4 |
| `docs/` | All | Project planning and member documentation |

## Quick Start

### Backend (Python)

```bash
uv sync --extra dev
```

Set environment variables:
```bash
GEMINI_API_KEY=...      # Required for text pipeline
GOOGLE_API_KEY=...      # Alternative to GEMINI_API_KEY
VOYAGE_API_KEY=...      # Optional
HF_TOKEN=...            # Optional (Hugging Face)
NGROK_AUTH_TOKEN=...    # Optional (tunnel for demo)
```

For Colab notebooks (GPU runtime):
```bash
pip install -r notebooks/requirements.colab.txt
```

Run the API server:
```bash
uv run python -m vibescents.backend_app
```

### Frontend (Next.js)

```bash
cd frontend/
bun install
bun run dev
```

### Image Processing

```bash
pip install -r requirements.txt

# Generate pseudo-labels via CLIP zero-shot
python src/clip_zero_shot.py

# Train a model
python experiments/cnn_baseline/train.py --epochs 15 --batch_size 64

# Run inference
python src/inference.py --model hybrid --checkpoint checkpoints/hybrid/best.pt --input path/to/outfit.jpg
```

### Fragrance Pipeline

```bash
python merge_datasets.py        # Unify 5 data sources
python embed_fragrances.py      # Generate embeddings (requires GPU)
python scent_clusters.py        # K-Means clustering
```

## Project Layout

```text
src/vibescents/           # Core Python package
  backend_app.py          # FastAPI server
  cli.py                  # CLI entry point
  embeddings.py           # Text/multimodal embedding generation
  fusion.py               # 4-signal score fusion
  image_scoring.py        # CNN probability-to-fragrance matching
  image_preprocess.py     # Base64 -> tensor preprocessing
  reranker.py             # Cross-encoder reranking
  schemas.py              # Pydantic models
  settings.py             # Configuration

models/                   # Image classification models (Neil)
  cnn_baseline.py
  clip_standalone.py
  cnn_clip_hybrid.py

experiments/              # Training/eval scripts per model
app/                      # Next.js frontend (Darren)
notebooks/                # Colab pipelines
tests/                    # pytest suite
docs/                     # Project docs
data/                     # Fragrance datasets
```

## CLI Workflows

```bash
# Embed occasion descriptions
uv run vibescents embed-occasions --input-json examples/occasions.json --output-dir artifacts/occasions

# Embed fragrance corpus
uv run vibescents embed-csv --input-csv data/fragrances.csv --id-column fragrance_id --text-column retrieval_text --output-dir artifacts/fragrance_text

# Multimodal retrieval
uv run vibescents multimodal-retrieve --fragrance-csv data/fragrances.csv --occasion-text "Black tie evening wedding" --image-path assets/tuxedo.jpg --output-dir artifacts/multimodal_query

# Rerank candidates
uv run vibescents rerank --candidate-json artifacts/multimodal_query/top_candidates.json --occasion-text "Black tie evening wedding" --output-json artifacts/reranked_candidates.json
```

## Fusion Formula

```
final_score = 0.30 * text_score + 0.25 * multimodal_score + 0.30 * image_score + 0.15 * structured_score
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
