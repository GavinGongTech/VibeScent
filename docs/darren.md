# Darren — Fragrance Data Lead + Frontend Lead

Last updated: April 18, 2026

---

## Scope

You own:

- Fragrance dataset sourcing and scraping
- Schema definition — canonical field names, types, and meaning
- Data loading and normalization
- Missing-value handling
- Canonical fragrance table export
- Next.js frontend (ScentAI)
- Web scraper integration

---

## Current State (April 25, 2026)

**Data delivery: complete.**
The full 35,889-row dataset is enriched and stored in `data/vibescent_enriched.csv`. This unified dataset is now used by all retrieval channels (Text, Multimodal, Image, Structured).

**Frontend: complete.**
Next.js app with Fragrantica scraper integrated. Commit `cf444a8 Integrated webscraper to front end`.

---

## Dataset

**Primary source:** Fragrantica (`data/raw/fragrantica_clean.csv`) — scraped with semicolon-delimited CSV, ~35,000+ rows.
**Secondary source:** Parfumo (`data/raw/parfumo_data_clean.csv`) — merged into unified dataset.
**Merged corpus:** `data/vibescent_enriched.csv` — 35,889 rows, fully enriched with LLM-generated attributes, deduplicated, canonical schema applied.

### Canonical Schema

| Field | Type | Description |
|---|---|---|
| `fragrance_id` | string | Unique identifier — stable across all branches, never recycled |
| `brand` | string | Fragrance house or brand |
| `name` | string | Fragrance name |
| `top_notes` | string | Top note list (evaporates first, 5–15 min) |
| `middle_notes` | string | Heart note list (main body, 20–60 min) |
| `base_notes` | string | Base note list (dry-down, hours) |
| `main_accords` | string | Comma-separated genre descriptors (e.g. woody, floral, spicy) |
| `rating_count` | float | Number of Fragrantica reviews |
| `rating_value` | float | Average rating score |
| `gender` | string | Gender metadata if present |
| `concentration` | string | EDP / EDT / Parfum etc. |
| `year` | float | Release year |
| `source` | string | `fragrantica` or `parfumo` |
| `likely_season` | enum | Enriched attribute: spring/summer/fall/winter |
| `likely_occasion`| string | Enriched attribute: e.g. "Formal evening event" |
| `formality` | float | Enriched attribute: 0-1 scale |
| `fresh_warm` | float | Enriched attribute: 0-1 scale |
| `day_night` | float | Enriched attribute: 0-1 scale |
| `character_tags` | list | Enriched attribute: [luminous, opulent, ...] |
| `vibe_sentence` | string | Enriched attribute: descriptive summary |
| `retrieval_text` | string | Merged raw + enriched text for embedding |

The enriched fields are added **by `enrich.py` (Harsh)** — the pipeline has been run on the entire 35,889-row corpus.

---

## Frontend

**Stack:** Next.js 14 (App Router), Tailwind CSS, Framer Motion, TypeScript.

**Pages:**
- `/` — Landing page with hero, about strip, footer
- `/demo` — Two-column layout: image upload + context form (left), fragrance results (right)
- `/model` — Pipeline explainer page

**Scraper integration:** `scraper/` directory — scrapes Fragrantica fragrance pages and returns structured JSON. Integrated into the frontend as of `cf444a8`.

**API stub:** `app/api/recommend/route.ts` — POST `/api/recommend` returns mock `FragranceRecommendation[]`. Replace this block when the real model backend is ready.

---

## Key Constraints

- `fragrance_id` must be stable and never recycled — downstream `.npy` embedding matrices reference row order. If row order changes, all embedding artifacts are invalidated.
- Do not let schema drift — field names must be consistent across all branches. Any new fields go through this doc or are documented as enrichment fields.
- Do not merge a second dataset unless one fully clean table already exists.

---

## Required Outputs

| Artifact | Status |
|---|---|
| `data/vibescent_500.csv` — canonical 500-row sample | ✅ |
| `data/processed/vibescent_unified.csv` — full 35,889-row merged corpus | ✅ |
| `data/raw/fragrantica_clean.csv` — raw Fragrantica scrape | ✅ |
| `data/raw/parfumo_data_clean.csv` — raw Parfumo data | ✅ |
| Next.js frontend | ✅ |
| Scraper integration | ✅ |
| Data quality summary (missingness rates per field) | ❌ Pending |

---

## Interfaces

**Others depend on you:**
- Harsh → `vibescent_500.csv` for enrichment and embedding ✓
- Karan → canonical fragrance table for text generation ✓
- Neil → cleaned fragrance table for image retrieval ✓
- All branches → stable `fragrance_id` guarantee ✓
