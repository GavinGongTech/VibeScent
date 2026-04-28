---
marp: true
theme: default
class: invert
paginate: true
style: |
  section {
    background: #0a0a0a;
    color: #f5f0e8;
    font-family: 'Georgia', serif;
    padding: 48px 60px;
  }
  h1 {
    color: #c9a96e;
    font-size: 2.2em;
    font-weight: 300;
    letter-spacing: 0.04em;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 12px;
  }
  h2 { color: #c9a96e; font-size: 1.5em; font-weight: 400; }
  h3 { color: #f5f0e8; font-size: 1.1em; font-weight: 400; }
  strong { color: #c9a96e; }
  code {
    background: #111111;
    color: #c9a96e;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.82em;
  }
  pre {
    background: #111111;
    border-left: 3px solid #c9a96e;
    padding: 16px 20px;
    font-size: 0.72em;
    line-height: 1.6;
  }
  table { width: 100%; border-collapse: collapse; }
  th {
    color: #c9a96e;
    border-bottom: 1px solid #2a2a2a;
    padding: 8px 12px;
    text-align: left;
    font-weight: 400;
    letter-spacing: 0.06em;
    font-size: 0.82em;
    text-transform: uppercase;
  }
  td {
    padding: 8px 12px;
    border-bottom: 1px solid #1a1a1a;
    font-size: 0.86em;
  }
  li { margin: 7px 0; }
  section.title {
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
  }
  section.title h1 {
    font-size: 3.2em;
    border: none;
    margin-bottom: 4px;
  }
  section.title p { color: #888070; }
---

<!-- _class: title -->

# VIBESCENT

### AI-Powered Fragrance Recommendations from Outfit Images

<br>

**Harsh Agarwal · Neil Bhattacharyya · Karan Singh · Darren Nguyen**

*April 2026*

---

# The Problem

**You want to smell right for the occasion. But fragrance databases speak the wrong language.**

<br>

A user uploads an outfit photo and types: *"Gala, Evening, Mysterious."*

The database has **35,889 fragrances**. Every entry looks like this:

```
bergamot, lemon | jasmine, rose | sandalwood, vanilla
```

<br>

| Approach | Why it fails |
|---|---|
| Keyword search | Zero lexical overlap between "mysterious gala" and ingredient names |
| Raw semantic embedding | Ingredient chemistry and occasion context are different semantic spaces |
| Manual curation | Doesn't scale to 35,889 items or millions of possible outfit photos |

---

# The Core Insight

**Fragrances must be translated from chemistry vocabulary into human experience vocabulary before semantic matching works.**

<br>

**What a fragrance database has:**
```
bergamot, lemon, jasmine, rose, sandalwood, vanilla
```

**What our enrichment pipeline produces:**
```
Season: spring/fall  |  Best for: formal evening event
Formality: high      |  Character: sophisticated, radiant, elegant
Mood: confident, seductive  |  Palette: gold, ivory, amber
Vibe: A warm luminous floral that announces a polished evening presence.
```

<br>

The second form matches how users describe occasions — because both are written in human vocabulary about human experience.
Everything else in the system follows from this single insight.

---

# System Architecture — Three Stages

```
User: outfit image  +  occasion text ("Gala, Evening, Mysterious")
                         │
          ┌──────────────▼──────────────┐
          │     Stage 1: Retrieval      │
          │  Four signals independently │
          │  score all 35,889 frags     │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     Stage 2: Fusion         │
          │  Normalize + weighted sum   │
          │  → one score per fragrance  │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     Stage 3: Reranking      │
          │  Gemini holistically scores │
          │  the top-10 shortlist       │
          └──────────────┬──────────────┘
                         │
              Top 3 recommendations
              with explanation text
```

---

# The Data Foundation

**35,889 fragrances scraped from Fragrantica**
*Darren Nguyen — Data Lead*

<br>

| Field | Example |
|---|---|
| Brand + Name | Maison Francis Kurkdjian — Baccarat Rouge 540 |
| Top / Middle / Base notes | jasmine, saffron / amberwood / fir resin, cedar |
| Main accords | sweet, woody, floral, oriental |
| **Rating count** | **47,832 reviews** |

<br>

**Why rating count is the selection key:**
Fragrances with tens of thousands of reviews have been written about extensively — in reviews, editorial copy, and fragrance culture writing.
The LLM enricher has absorbed that writing during training.
It can accurately predict the vibe of Chanel No. 5.
It cannot reliably predict the vibe of an obscure niche fragrance with 8 reviews.

High rating count = the LLM knows this fragrance culturally.

---

# Offline Pipeline — Runs Once, Before Any User Request

```
Raw CSV (35,889 rows from Fragrantica + Parfumo)
    │
    ├── LLM Enrichment (All 35,889) ──────────────────────────
    │     35,889 × LLM call → 11 structured attributes per fragrance
    │     ~12–18 hours on local GPU cluster
    │
    ├── Build retrieval_text ──────────────────────────────────
    │     Merge raw note fields + enriched attributes
    │     → one rich string per fragrance, ready for embedding
    │
    ├── Text Embedding (All 35,889) ───────────────────────────
    │     Qwen3-Embedding-8B → 35,889 × 4096 matrix
    │
    └── Multimodal Embedding (All 35,889) ─────────────────────
          Qwen3-VL-Embedding-8B → 35,889 × 4096 matrix
          (cross-modal space: comparable to outfit image embeddings)
```

**Why keep all 35,889?** Full coverage — every fragrance in the database is enriched and searchable via all four channels.

---

# The Enrichment Pipeline

**Per-fragrance LLM translation — the most important preprocessing step**
*Harsh Agarwal*

<br>

For each fragrance, the LLM reads raw metadata and generates:

| Attribute | Type | Example (Baccarat Rouge 540) |
|---|---|---|
| `formality` | float 0–1 | `0.88` |
| `fresh_warm` | float 0–1 | `0.75` — warm, amber-heavy |
| `day_night` | float 0–1 | `0.90` — night wear |
| `likely_season` | enum | `fall` |
| `likely_occasion` | string | `Black tie evening event` |
| `character_tags` | list | `[luminous, opulent, crystalline]` |
| `mood_tags` | list | `[romantic, confident, mysterious]` |
| `color_palette` | list | `[amber, gold, ivory]` |
| `vibe_sentence` | string | *A luminous amber that bridges elegance and sensuality.* |

<br>

Schema enforced via Pydantic — malformed outputs are rejected and retried.
All 35,889 fragrances have been successfully enriched.

---

# The Four Retrieval Signals

**Stage 1: each branch asks a fundamentally different question**

<br>

```
"Gala, Evening, Mysterious"  +  outfit photo
          │
    ┌─────┼──────────────┬──────────────────┐
    │     │              │                  │
  Text  Multi-        Image             Structured
  Branch modal         CNN               Branch
    │     Branch        Branch               │
    │     │              │                  │
  35,889  35,889       35,889             35,889
  scores  scores       scores             scores
```

- **Text:** *Is the fragrance's enriched description semantically close to the occasion?*
- **Multimodal:** *Does the joint embedding of the outfit image + text match the fragrance?*
- **Image CNN:** *Do the outfit's visual attributes — formality, season, time of day — match?*
- **Structured:** *Do the raw numeric fields align with what this occasion requires?*

---

# Text Retrieval Branch

**The backbone — covers all 35,889 fragrances**
*Harsh Agarwal*

<br>

**Offline:** embed every fragrance's `retrieval_text` → 35,889 × 4096 matrix saved to disk.

**Online:**
```python
query = Qwen3_Embedding_8B("Gala, Evening, Mysterious")   # 4096-d vector
scores = FRAG_EMBEDDINGS @ query.T                         # 35,889 scores, ~50ms
```

<br>

**Why Qwen3-Embedding-8B?**
- #1 MTEB English leaderboard, score 68.32 — beats OpenAI text-embedding-3-large
- Matryoshka-trained: using full 4,096 dims for maximum precision
- Runs locally — no API quota risk during live demo

**What it catches:**
"Mysterious evening" matches "romantic, mysterious... Black tie evening" because those phrases co-occur in fragrance culture writing the model was trained on.
The enriched `retrieval_text` bridges the gap that raw ingredient lists cannot.

---

# Multimodal Retrieval Branch

**The joint image-text signal — All 35,889 fragrances**
*Harsh Agarwal + Neil Bhattacharyya*

<br>

**Offline:** `Qwen3-VL-Embedding-8B` embeds each fragrance's `retrieval_text` in the cross-modal space.

**Online:**
```python
query = Qwen3VL(text="Gala, Evening, Mysterious", image=outfit_photo)
scores = MM_EMBEDDINGS @ query.T   # 35,889 scores
```

<br>

**What makes this different from the text branch:**
The joint embedding shifts based on the outfit's visual character.
A dark structured blazer pushes the query toward "formal, evening" — even if the user typed "casual."
A bright yellow sundress pulls toward "fresh, summer."
The image *overrides* the text signal when there is a mismatch.

**Why Qwen3-VL-Embedding-8B?**
MMEB-V2 cross-modal benchmark: **77.8** — #1 overall.
`gemini-embedding-2` (original plan) scored 68.9 — we switched mid-project for the 9-point gain.

---

# Image CNN Branch

**Explicit outfit attribute prediction**
*Neil Bhattacharyya — Image Retrieval Lead*

<br>

**Architecture:**
```
outfit image (224 × 224)
    ↓
ResNet-50 → 2048-d features
CLIP ViT-L/14 → 768-d features
    ↓  concatenate + linear projection
256-d shared trunk
    ├── formal_head  → [casual: 0.05 / semi: 0.12 / formal: 0.83]
    ├── season_head  → [spring: 0.07 / summer: 0.04 / fall: 0.71 / winter: 0.18]
    └── time_head    → [day: 0.15 / night: 0.85]
```

<br>

**Why a CNN instead of just using CLIP's embedding?**
CLIP gives an opaque vector. The CNN gives **explicit probabilities**.
You can say: *"this outfit is 83% formal, 71% fall, 85% night."*
Those numbers align directly with the enrichment attributes already on every fragrance.
The CNN bridges visual signals to the same attribute space the fragrances live in.

---

# Image CNN Scoring

**Matching outfit attributes to fragrance attributes**

<br>

CNN predicts: `P(formal)=0.83`, `P(fall)=0.71`, `P(night)=0.85`

For each fragrance, look up its enriched attributes and compute:

```python
score = P_formal[fragrance_formal_class]
      × P_season[fragrance_season_class]
      × P_time[fragrance_time_class]
```

<br>

**Baccarat Rouge 540** — `formality=0.88 (formal)`, `season=fall`, `day_night=0.90 (night)`:
```
0.83 × 0.71 × 0.85 = 0.50   ✓  correctly scored high
```

**A fresh citrus aquatic** — `formality=0.15 (casual)`, `season=summer`, `day_night=0.20 (day)`:
```
0.05 × 0.04 × 0.15 = 0.0003   ✓  correctly scored near-zero
```

<br>

This creates sharp contrast between fragrances that fit the outfit and those that don't —
something text similarity alone cannot achieve.

---

# Score Fusion

**Combining four signals into one ranking**
*Harsh Agarwal*

<br>

**The problem:** each branch produces scores on a different scale. Text cosine similarities cluster 0.3–0.7. CNN probability products span 0.0001–0.8. Combining them raw would let scale dominate, not quality.

**Solution:** min-max normalize each signal independently to [0, 1] first.

```python
fused = (0.30 × text_norm           # wide recall — covers all 35,889
       + 0.25 × multimodal_norm     # cross-modal alignment — all 35k
       + 0.30 × image_norm          # explicit formality/season/time — all 35k
       + 0.15 × structured_norm)    # numeric attribute match — tiebreaker
```

<br>

**Weights are not guesses.** A grid search over all valid weight combinations (step=0.05) runs against 20 benchmark cases. The configuration that maximizes retrieval accuracy on the benchmark is the one we ship.

**Selection:** `argpartition(-fused, 50)[:50]` — O(N) top-50, then sort those 50.

---

# The Reranker

**Holistic cultural reasoning on the shortlist**
*Harsh Agarwal*

<br>

Fusion finds fragrances that *score well numerically*.
The reranker asks whether they *actually make sense for this specific outfit and occasion*.

<br>

**Qwen3-VL-Reranker-8B receives:**
- The outfit image
- Occasion text ("Gala, Evening, Mysterious")
- Top 10 candidates — name, brand, `retrieval_text`, baseline score, all structured metadata

**Returns for each candidate:**
```json
{
  "overall_score": 0.91,
  "formality_score": 0.91,
  "season_score": 0.91,
  "freshness_score": 0.91,
  "explanation": "The crystalline amber character aligns with the outfit's
                  structured, high-contrast aesthetic and rewards close
                  contact — right for a gala where impressions are made."
}
```

**Note:** Sub-scores are currently mirrored from the overall score in the Week 5 implementation.

---

# End-to-End at Inference Time

```
User uploads tuxedo photo  +  "Gala, Evening, Mysterious"
  │
  ├─ 1. Occasion text: "Gala, Evening, Mysterious"
  │
  ├─ 2. Text branch
  │       embed query (4096-d)  →  dot product with 35,889 × 4096 matrix
  │       → scores_text[35,889]   ~50ms
  │
  ├─ 3. Multimodal branch
  │       embed(image + text)   →  dot product with 35,889 × 4096 matrix
  │       → scores_mm[35,889]
  │
  ├─ 4. Image CNN
  │       ResNet-50 + CLIP forward pass  →  P(formal, season, time)
  │       → score each fragrance         →  scores_img[35,889]
  │
  ├─ 5. Structured: map query context to numeric targets → scores_struct[35,889]
  │
  ├─ 6. Normalize all four to [0,1]  →  weighted sum  →  fused[35,889]
  │
  ├─ 7. argpartition top 50  →  sort  →  shortlist
  │
  ├─ 8. Qwen3-VL-Reranker-8B reranks top 10 with outfit image
  │
  └─ 9. Return top 3 with name, house, score, notes, occasion, rationale
```

---

# Model Choices — Justified

| Role | Model | Key Reason |
|---|---|---|
| Text embedding | Qwen3-Embedding-8B | #1 MTEB English — Matryoshka 4096-d |
| Multimodal embedding | Qwen3-VL-Embedding-8B | #1 MMEB-V2 cross-modal (77.8 vs 68.9) |
| Enrichment | Qwen3.5-27B local / Gemini Flash | Structured JSON, schema enforcement |
| Image classifier | ResNet-50 + CLIP ViT-L/14 | Interpretable 3-head attribute output |
| Reranker | Qwen3-VL-Reranker-8B | Local multimodal reranking, mirrors sub-scores |
| Evaluation judge | Gemini 2.5 Pro | Different from label generator — no self-grading |

<br>

**One deliberate decision worth calling out:**
The evaluation judge (Gemini 2.5 Pro) is a *different model* from the label generator (Qwen3-VL-Reranker-8B).
If the same model generates benchmark labels and evaluates results against them,
it will grade its own outputs favorably.
Separating them is standard practice in LLM evaluation — it's not an accident.

---

# What We Built

| Artifact | Status | Owner |
|---|---|---|
| 35,889-row fragrance dataset (Fragrantica + Parfumo) | ✅ | Darren |
| Next.js frontend + scraper | ✅ | Darren |
| CNN-CLIP hybrid image classifier (3 heads) | ✅ | Neil |
| Qwen3-VL multimodal embedder | ✅ | Neil |
| Enrichment pipeline — dual-LLM, checkpointed | ✅ | Harsh |
| 21-stage orchestration notebook (Kaggle T4) | ✅ | Harsh |
| Occasion embedding matrix + similarity heatmap | ✅ | Harsh |
| Raw fragrance embedding matrix (500 × 1536) | ✅ | Harsh |
| Score fusion with weight grid search | ✅ | Harsh |
| Gemini reranker with image context | ✅ | Harsh |
| FastAPI backend + Pydantic schemas | ✅ | Harsh |
| K-means cluster vibe pipeline | ✅ | Karan |

---

# Team

| Person | Role | Core Contribution |
|---|---|---|
| **Harsh Agarwal** | Text Retrieval & Reranking Lead | Enrichment pipeline, 21-stage GPU orchestration, score fusion, reranker, all ML backend infrastructure |
| **Neil Bhattacharyya** | Image Retrieval Lead | CNN-CLIP hybrid classifier, Qwen3-VL integration |
| **Karan Singh** | Fragrance Representation Lead | K-means cluster pipeline, vibe space mapping |
| **Darren Nguyen** | Data + Frontend Lead | Fragrantica scraper, canonical dataset, Next.js frontend |
| **Gavin Gong** | PM + Integration Lead | System design, benchmarking, team coordination |

<br>

**How the pieces fit:**
Neil's model and Karan's embeddings are inputs to Harsh's pipeline.
The pipeline is the connective tissue that makes all contributions work together at inference time.

---

<!-- _class: title -->

# Thank You

### Questions?

<br>

*The recommendation you see is the product of 35,889 fragrances,*
*an LLM translation step, four independent retrieval signals,*
*and a holistic multimodal reranker — running in under a second.*
