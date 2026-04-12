# Darren — Fragrance Data Lead

Last updated: April 12, 2026

---

## Scope

You own:

- Fragrance dataset selection and sourcing
- Schema definition — canonical field names, types, and meaning
- Data loading and normalization
- Missing-value handling
- Canonical fragrance table export

---

## Status (April 12)

**Week 2 delivery: complete.** `data/vibescent_500.csv` is the canonical 500-row fragrance table. Schema is documented. Downstream branches are consuming it without manual cleanup.

---

## Required Schema

Every row in the canonical fragrance table must contain:

| Field | Type | Description |
|---|---|---|
| `fragrance_id` | string | Unique identifier, stable across all branches |
| `brand` | string | Fragrance house or brand |
| `name` | string | Fragrance name |
| `notes` | string | Raw note string (top / heart / base) |
| `accords` | string | Comma-separated main accords |
| `season_tags` | string | Available season metadata if present in source |
| `occasion_tags` | string | Available occasion metadata if present in source |
| `gender` | string | Gender metadata if present in source |
| `embedding_text` | string | Raw note concatenation (Karan's original field, kept for RAW baseline comparison) |

The following enriched fields are added downstream by `enrich.py` (Harsh) — Darren does not generate these:

- `likely_season`, `likely_occasion`, `formality`, `fresh_warm`, `day_night`, `character_tags`, `vibe_sentence`, `retrieval_text`, `display_text`

---

## Week 3 Plan

- Improve coverage only if it does not break the canonical schema (e.g. filling high-missingness fields for fragrances that appear in benchmark cases)
- Support targeted enrichment where the benchmark exposes gaps
- Do not merge a second dataset unless one fully clean table already exists

---

## Constraints

- Do not merge multiple messy datasets before one clean table exists
- Do not let schema drift — field names must be consistent across all branches
- `fragrance_id` must be stable and never recycled — downstream `.npy` matrices reference row order

---

## Required Outputs

- `data/vibescent_500.csv` ✓
- Source dataset decision note (which dataset, why, what was dropped)
- Data quality summary (missingness rates per field)
- Confirmed `fragrance_id` stability guarantee

---

## Interfaces

**Others depend on you:**
- Harsh → `vibescent_500.csv` for enrichment and embedding ✓
- Karan → canonical fragrance table for text generation ✓
- Neil → cleaned fragrance table for image retrieval

---

## Success Criteria

- The team has one stable, versioned fragrance table
- `fragrance_id` is unique and stable
- Downstream branches consume the table without manual cleanup
- Schema does not drift across branches — any additions go through you or are documented as enrichment fields
