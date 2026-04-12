# VibeScent Text Processing

Harsh's branch owns:

- text embeddings with `gemini-embedding-001`
- multimodal embedding retrieval with `gemini-embedding-2`
- reranking with `gemini-3.1-pro-preview`
- benchmark generation utilities

## Setup

```bash
uv sync --extra dev
```

Set one of:

- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

## Project Layout

```text
src/vibescents/
  cli.py
  benchmark.py
  embeddings.py
  io_utils.py
  reranker.py
  schemas.py
  settings.py
  similarity.py

examples/
  benchmark_briefs.json
  occasions.json

tests/
```

## Common Workflows

Embed occasion descriptions and generate a similarity heatmap:

```bash
uv run vibescents embed-occasions \
  --input-json examples/occasions.json \
  --output-dir artifacts/occasions
```

Embed a fragrance corpus from `retrieval_text`:

```bash
uv run vibescents embed-csv \
  --input-csv data/fragrances.csv \
  --id-column fragrance_id \
  --text-column retrieval_text \
  --output-dir artifacts/fragrance_text
```

Run shared-space multimodal retrieval with occasion text plus image:

```bash
uv run vibescents multimodal-retrieve \
  --fragrance-csv data/fragrances.csv \
  --id-column fragrance_id \
  --text-column retrieval_text \
  --occasion-text "Black tie evening wedding in winter" \
  --image-path assets/tuxedo.jpg \
  --output-dir artifacts/multimodal_query
```

Rerank candidate fragrances:

```bash
uv run vibescents rerank \
  --candidate-json artifacts/multimodal_query/top_candidates.json \
  --occasion-text "Black tie evening wedding in winter" \
  --image-path assets/tuxedo.jpg \
  --output-json artifacts/reranked_candidates.json
```

Generate AI-assisted benchmark labels from case briefs:

```bash
uv run vibescents generate-benchmark \
  --briefs-json examples/benchmark_briefs.json \
  --output-json artifacts/benchmark_cases.json
```

## Production Stack

Primary retrieval signals:

- `gemini-embedding-001` text score
- `gemini-embedding-2` shared-space multimodal score
- Neil's CLIP/CNN/hybrid image score
- structured metadata score

Initial fusion rule:

```text
final_score =
  0.30 * text_score +
  0.25 * multimodal_score +
  0.30 * image_score +
  0.15 * structured_score
```

## Evaluation Rules

- Do not present five examples as validation.
- Keep the 20-case benchmark reproducible.
- Use metadata-based scoring as the primary metric.
- Treat LLM judging as secondary analysis, not primary ground truth.

## Verification

Run local tests for the non-API utilities:

```bash
uv run pytest
```
 pytest
```
