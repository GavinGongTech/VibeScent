# VibeScent — Full Project Reference

> Last updated: April 27, 2026. Incorporates all architecture decisions, ML design rationale, and implementation details through Week 5.
>
> **Current implementation status (Week 5):** All four scoring channels and the full frontend are shipped. The corpus consists of **35,889 fragrances**, all of which are fully enriched and embedded—the "Tier B" quality gap has been eliminated. The image branch uses CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) with **5 classification heads** (formal, season, time, gender, frequency). The structured branch is fully query-aware. The reranker is **Qwen3-VL-Reranker-8B** (Local GPU, zero API keys), with sub-scores mirrored from the overall score. The corpus is embedded with **Qwen3-VL-Embedding-8B at full 4096-d**: `artifacts/qwen3vl_corpus/embeddings.npy` (35,889 × 4096, L2-normalized). Text and multimodal channels share this single unified matrix. BM25 post-fusion blend and a hard context filter are live. MMR diversification is the reranker fallback.

---

## Project Goal

Build a competition-grade system that takes an outfit image and occasion text, retrieves fragrance recommendations, and supports a polished final demo with defensible ML choices.

### Success Criteria

- Returns fragrance recommendations that feel coherent and explainable
- Beats simple text-only and image-only baselines on the 20-case benchmark
- Produces stable outputs across repeated runs
- Supports a polished frontend demo
- Delivers a research story that goes beyond prompt glue — the reranker must be benchmarked, not just added

---

## Team and Ownership

| Role | Owner | Branch |
|---|---|---|
| PM and Integration Lead | Gavin | `main` |
| Text Retrieval and Reranking Lead | Harsh | `Text_Processing` |
| Image Retrieval Lead | Neil | `Image_Processing` |
| Fragrance Representation Lead | Karan | `Text_Processing` (aligned with Harsh) |
| Fragrance Data Lead | Darren | `main` / shared |

---

## System Architecture — Three Stages

The full system is a three-stage pipeline that runs at inference time:

```
User input: outfit image + occasion context
        ↓
Stage 1: Candidate Retrieval
  Four independent signals score every fragrance in the corpus
        ↓
Stage 2: Score Fusion
  Signals are min-max normalized and combined via weighted sum
        ↓
Stage 3: Reranking (optional)
  A multimodal LLM reorders the top-10 shortlist and provides rationale
        ↓
Output: Top 3–5 ranked fragrance recommendations with explanations
```

Each stage is explained in deep technical detail below.

---

## How the Pipeline Works — End to End

This section is a narrative walkthrough of the full system. Read this before the technical sections. It explains what happens to each input at each step, why each transformation was designed that way, and what the alternatives were.

---

### The Fundamental Problem

The system receives two inputs: an outfit photo and a free-text occasion description. It must return a ranked list of fragrances from a 35,889-item database.

The naive solution — keyword search — fails immediately. A user typing "mysterious evening gala" and a fragrance described as "jasmine, saffron, amberwood, cedar" have zero lexical overlap. There is no word in common. Standard search engines return nothing or return random results.

The slightly less naive solution — semantic embedding — partially works but has a different failure mode. If you embed "mysterious evening gala" and compare it to embeddings of raw fragrance descriptions, you find fragrances that mention evening or occasion language. But most fragrance database entries look like `"bergamot, lemon | jasmine, rose | sandalwood, vanilla"` — ingredient lists, not occasion descriptions. The semantic space of ingredient chemistry and the semantic space of occasion context do not overlap even with a good embedding model.

The core architectural insight driving all design decisions: **you must translate the fragrance from chemistry vocabulary into human experience vocabulary before you can do meaningful semantic matching.** This translation is the enrichment step. Everything else follows from it.

---

### What Happens Before Any User Request — Offline Preprocessing

The pipeline has two phases: offline (runs once on GPU, takes hours) and online (runs per request, takes seconds). The offline phase builds the indexes. The online phase queries them.

**Step 1 — You have 35,889 fragrances in a raw CSV.**

Each row has: brand, name, top/middle/base notes, main accords, gender, concentration, rating count. This is what Fragrantica stores.

**Step 2 — You enrich the full corpus of 35,889 fragrances.**

Every row in the dataset now undergoes LLM enrichment. We have moved beyond the historical 2,000-fragrance "Tier B" limit. Every fragrance in the dataset is now enriched, ensuring that long-tail niche fragrances carry the same rich semantic signals as global bestsellers.

**Step 3 — You translate fragrances from chemistry to experience vocabulary (Enrichment).**

For each of the 35,889 fragrances, an LLM (**Qwen3-8B** locally) reads the raw metadata and generates:

```
formality: 0.88          (how dressed-up is this fragrance?)
day_night: 0.82          (day or night appropriate?)
fresh_warm: 0.75         (fresh/citrus/light vs warm/amber/heavy?)
likely_season: fall
likely_occasion: Black tie evening
character_tags: [luminous, opulent, crystalline, warm, modern]
mood_tags: [romantic, confident, mysterious]
color_palette: [amber, gold, ivory]
vibe_sentence: "A luminous amber that bridges elegance and sensuality with crystalline precision."
```

This translation works because the LLM has absorbed fragrance culture from reviews, editorial writing, and marketing copy during training. It knows that saffron + amber reads as "luxury" and "formal" in a cultural context that raw ingredient names don't encode.

The output is constrained via `outlines` — a library that restricts token generation to valid JSON matching the exact schema. You cannot get a malformed or missing field. This is not optional: at 35,889 inference calls, even a 1% failure rate means ~358 broken fragrance embeddings.

**Step 4 — You build a rich retrieval string for each fragrance.**

The enrichment fields are concatenated with the raw fields into a single `retrieval_text` string:

```
Brand: MFK | Name: Baccarat Rouge 540 | Accords: sweet, woody |
Top: jasmine, saffron | Heart: amberwood | Base: fir resin |
Season: fall | Best for: Black tie evening | Formality: high |
Character: luminous, opulent, crystalline | Mood: romantic, mysterious |
Vibe: A luminous amber that bridges elegance and sensuality.
```

This string now lives in the same semantic space as how users describe occasions. "Mysterious evening" will match "romantic, mysterious... Black tie evening" in embedding space because those phrases co-occur in the cultural writing the embedding model was trained on.

**Step 5 — You embed every fragrance into a 4096-dimensional vector (Unified Corpus).**

Qwen3-VL-Embedding-8B reads each `retrieval_text` string and outputs a 4096-dimensional vector. All 35,889 rows carry enriched fields. The vectors are L2-normalized and saved as a single unified matrix: `artifacts/qwen3vl_corpus/embeddings.npy` (shape 35,889 × 4096, float32, ~560 MB).

Week 5 Reality: The pipeline uses Qwen3-VL-Embedding-8B at full 4096-d. No Matryoshka truncation or 4096-d projection is applied. The VL model's superior cross-modal alignment is leveraged at its native resolution, providing maximum semantic density for retrieval.

**Step 6 — Unified text+multimodal corpus: one matrix for both channels.**

Both the text channel query (text-only embedding of the occasion phrase) and the multimodal channel query (joint image + text embedding) query against the same `artifacts/qwen3vl_corpus/embeddings.npy` matrix. Historical "Tier" distinctions are obsolete for retrieval—the full 35,889-row corpus is reachable by all signals.

This is architecturally valid because all three artifacts — corpus documents, text queries, and multimodal queries — are produced by the same model (Qwen3-VL-Embedding-8B) and therefore live in the same embedding space. Cosine similarity between them is meaningful. The prior cross-model mismatch (Qwen3-Embedding text vectors vs. Qwen3-VL query vectors) that existed in Week 4 is fully resolved.

**Step 7 — CLIP ViT-L/14 zero-shot classifier produces outfit attribute scores (Image Branch).**

This runs online on each user's outfit image. `CLIPImageScorer` (`openai/clip-vit-large-patch14`) takes the outfit photo and produces **five** independent probability distributions:
- `formal`: P ∈ ℝ³ — [casual, smart-casual, formal]
- `season`: P ∈ ℝ⁴ — [spring, summer, fall, winter]
- `time`: P ∈ ℝ² — [day, night]
- `gender`: P ∈ ℝ³ — [male, female, neutral]
- `frequency`: P ∈ ℝ² — [occasional, everyday]

For each class, 3 natural-language prompts are encoded and their similarities to the image embedding are averaged before softmax. This multi-prompt averaging is Neil's original approach from `backend/clip_zero_shot.py` — it reduces sensitivity to any single prompt phrasing.

The fragrance database has pre-computed attributes from enrichment (Step 3). The CLIP scorer bridges the gap: it maps the outfit image into the same 5-attribute space. `NeilCNNWrapper` is also available for loading Neil's trained CNN checkpoint if a checkpoint is provided — the scoring interface (`ImageHeadProbabilities`) is identical.

---

### What Happens at Request Time — Online Inference

A user uploads an outfit photo and selects: Event = "Gala", Time = "Evening", Mood = "Mysterious".

**Step 1 — Build the occasion query string.**

`context_to_query_string(ctx)` expands each context field into a rich descriptive phrase from a lookup table, then joins them with `|`:

```
"black tie gala formal event elegant luxury oriental, chypre, oud, amber, incense, opulent, sillage |
 evening twilight dinner sophisticated |
 mysterious dark seductive complex smoky oud"
```

If `ctx.customNotes` is set (e.g. `"prefer something with oud and leather"`), it is appended verbatim. The expanded string lives in the same human-experience vocabulary space as the enriched fragrance descriptions — increasing semantic overlap for the embedding comparison. A bare `"Gala"` would embed near formal-context text; the expanded phrase embeds directly into the fragrance chemistry-culture space where retrieval lives.

**Step 2 — Text Branch produces 35,889 scores.**

`q_text = Qwen3-VL-Embedding-8B(occasion_phrase)` → 4096-d vector, L2-normalized.

`sig_text = CORPUS_EMBEDDINGS @ q_text.T` → 35,889 scalar scores.

This is a matrix multiply: the pre-computed 35,889 × 4096 matrix (`artifacts/qwen3vl_corpus/embeddings.npy`) dotted against the 4096-d query vector. Each scalar is the cosine similarity between that fragrance's enriched text embedding and the expanded occasion query.

Time: ~50ms on CPU, ~5ms on GPU. Single BLAS matrix multiply.

What it finds: fragrances whose retrieval_text is semantically close to the expanded occasion phrase. Because both corpus and query are produced by Qwen3-VL-Embedding-8B, there is no cross-model embedding space mismatch. All 35,889 rows have enriched `retrieval_text` (source: `vibescent_enriched.csv`).

**Step 3 — Multimodal Branch produces 35,889 scores.**

`q_mm = Qwen3-VL-Embedding-8B(text=occasion_phrase, image=outfit_photo)` → 4096-d vector.

The model reads both the image and the text simultaneously and produces a single joint embedding. This encodes: what does this outfit look like AND what is the user saying about the occasion?

`sig_mm = CORPUS_EMBEDDINGS @ q_mm.T` → 35,889 scores (same unified matrix as the text branch).

Week 5 Reality: the multimodal branch queries the full 35,889-row corpus at full 4096-d. This means the visual signal from the outfit photo now influences scores for every fragrance in the database. There is no truncation or limited candidate pool.

Why this is different from the text branch: the joint embedding shifts based on the outfit's visual character. A dark, structured blazer in the outfit photo pushes the query vector toward "formal, evening" associations even if the user typed "casual." The image cannot be ignored; the text branch cannot see it.

**Step 4 — Image Zero-Shot Branch produces scores over enriched corpus.**

The outfit photo is decoded from base64, opened as a PIL RGB image, and processed by `CLIPImageScorer`:

```
CLIP.get_image_features(outfit) → image_embedding  (1, D), L2-normalized
cosine_sim(image_emb, formal_text_embs) → logits → softmax → [0.05, 0.12, 0.83]
cosine_sim(image_emb, season_text_embs) → logits → softmax → [0.07, 0.04, 0.71, 0.18]
cosine_sim(image_emb, time_text_embs)   → logits → softmax → [0.15, 0.85]
```

CLIP predicts: this outfit is 83% formal, 71% fall-appropriate, 85% evening.

For each corpus row, the engine looks up 5 enrichment attributes and computes a joint likelihood score across all 5 heads:

```
score = P_formal[formal_class]
      × P_season[season_class]
      × P_time[time_class]
      × P_gender[gender_class]
      × P_frequency[frequency_class]
```

A black-tie formal evening fragrance (formality=0.88, day_night=0.82, season=fall, gender=neutral, frequency=occasional) scored against a tuxedo photo:
```
0.83 × 0.71 × 0.85 × 0.55 × 0.78 ≈ 0.21
```

A casual daytime unisex everyday fragrance scored against the same tuxedo photo:
```
0.05 × 0.04 × 0.15 × 0.40 × 0.92 ≈ 0.0001
```

The 5-head product creates a sharper distinction than the 3-head version — adding gender and frequency axes reduces false positives where a fragrance matched on formality/season/time but was culturally misaligned (e.g. a masculine aftershave for a feminine editorial look).

**Step 5 — Structured Branch produces scores across the full corpus.**

`compute_structured_scores(ctx, corpus_df)` maps each context field to a float target in enrichment space, then scores every fragrance by its proximity to those targets:

```python
# "Gala" → formality_target=0.90
# "Evening" → day_night_target=0.70
# "Mysterious" → fresh_warm_target=0.75 (warm end)
score = mean(1.0 - |fragrance_attr - target|) over active dimensions
```

If no context is provided, all fragrances receive 0.5 (neutral). The branch never fails — it's pure arithmetic on pre-computed enrichment attributes, no GPU required.

**Step 6 — Normalize and fuse.**

Each of the four score arrays is independently min-max normalized to [0, 1]:
```
sig_text_norm = (sig_text - min) / (max - min)    # shape (35889,)
sig_mm_norm   = (sig_mm   - min) / (max - min)    # shape (35889,)
sig_img_norm  = (sig_img  - min) / (max - min)    # shape (35889,)
sig_s_norm    = (sig_s    - min) / (max - min)    # shape (35889,)
```

In Week 5, all four arrays cover the full 35,889-row corpus — no zero-padding for multimodal. Normalization makes the weight the actual control dial: without it, a branch with naturally higher raw scores would dominate regardless of assigned weight.

Fusion:
```
fused = 0.30 × sig_text_norm + 0.25 × sig_mm_norm + 0.30 × sig_img_norm + 0.15 × sig_s_norm
```

**Step 6a — Hard filter (post-fusion, pre-top-k).**

`_hard_filter(ctx)` builds a boolean mask that zeroes out structurally incompatible fragrances before top-k selection:

```python
if ctx.eventType in ('Gala', 'Wedding'):  mask &= formality >= 0.3   # exclude very casual
elif ctx.eventType == 'Casual':           mask &= formality <= 0.8   # exclude black-tie
if ctx.mood == 'Fresh':                   mask &= fresh_warm <= 0.7  # exclude heavy orientals
elif ctx.mood == 'Warm':                  mask &= fresh_warm >= 0.3  # exclude cold/aquatic
```

If fewer than 3 fragrances pass the filter, it is bypassed entirely (safety guard). The filter prevents obviously wrong results from surfacing in the top-3 regardless of how other channels scored them.

**Step 6b — BM25 post-fusion blend.**

After hard-filtering, a BM25 lexical score is blended in:
```python
bm25_norm = min_max_normalize(BM25Okapi.get_scores(query_tokens))
fused_filtered = 0.9 × fused_filtered + 0.1 × bm25_norm
```

BM25 is built from the `retrieval_text` column at engine startup. The 10% weight is a small correction signal that boosts fragrances with direct keyword overlap (e.g. a query containing "oud" matching a fragrance with "oud" in its retrieval_text). This catches lexical matches that the dense embedding might rank lower due to embedding space geometry. BM25 is a no-op if `rank_bm25` is not installed.

**Step 7 — Select top candidates.**

`top_20_indices = argpartition(-fused_filtered, 20)[:20]` — the 20 fragrances with the highest blended scores. O(N) introselect, then sort those 20.

Shortlist size reduced from 50 (Week 4) to 20: the reranker is a local 8B model (not a massive API call), so a smaller, higher-precision shortlist is more appropriate. Fewer irrelevant candidates reduces the reranker's task difficulty.

**Step 8 — Rerank or MMR fallback.**

The 20 candidates go to `Qwen3VLReranker` (Qwen3-VL-Reranker-8B, local GPU). The reranker receives:
- The outfit image (temp file path)
- The occasion query string
- Each candidate's `build_candidate_text()` output: `"Name by Brand | vibe_sentence | Occasion: X | Notes: Y"`

The reranker returns per-candidate `overall_score`, `formality_score`, `season_score`, `freshness_score`, and `explanation`. In the current implementation, **the sub-scores (formality, season, freshness) are mirrored from the overall score**, representing the model's confidence in the total match. Top 3 by `overall_score` are selected.

If the reranker is unavailable (no GPU, load failure), MMR diversification runs instead:
```python
top3 = mmr_select(q_emb[0], corpus_emb, top_20, lambda_param=0.5, top_k=3)
```

MMR (Maximal Marginal Relevance, λ=0.5) balances relevance against diversity — the second pick maximizes `0.5 × relevance - 0.5 × max_sim_to_selected`, preventing the top-3 from being near-duplicate fragrances from the same accord family.

**Step 9 — Build the response.**

For each of the top 3–5 fragrances, the engine constructs a `FragranceRecommendation`:
- `name`, `house`: from the DataFrame
- `score`: the fused or reranker score
- `notes`: top/middle/base notes joined, deduplicated, capped at 8
- `reasoning`: the `vibe_sentence` from enrichment (or reranker rationale if shipped)
- `occasion`: `likely_occasion` from enrichment

This is what the frontend displays.

---

### Why This Architecture and Not Something Else

**Why retrieval + fusion instead of a single generative model?**

A single model (e.g., ask GPT-4V: "given this outfit and occasion, recommend three fragrances") could work but has critical problems for this use case:

1. **Hallucination.** A generative model will confidently recommend fragrances that don't exist, or describe existing fragrances inaccurately. A retrieval system can only return fragrances that are in the database — no hallucination is possible.

2. **No control over the candidate pool.** A generative model picks based on training data frequency. Famous fragrances (Chanel No. 5, Dior Sauvage) will be recommended disproportionately regardless of fit. Retrieval scores every fragrance in the corpus, so an obscure niche fragrance can win if it's the best match.

3. **Latency.** A full GPT-4V inference for a detailed multi-fragrance recommendation takes 5-15 seconds. The retrieval pipeline (without reranker) takes 2-4 seconds. With 35K fragrances pre-indexed, the search is fast.

4. **Benchmarkability.** You can measure retrieval quality (attribute_match@3, neighborhood_hit@5) with structured metrics. Evaluating generative output quality requires LLM judges, which have their own biases.

**Why four signals instead of one or two?**

Each signal covers a blind spot of the others:
- Text alone: can't see the outfit, biased by vocabulary overlap
- Multimodal alone: reaches the full corpus, black-box
- Image CNN alone: 9 values total, no semantic nuance
- Structured alone (ideal): no neural understanding of similarity, only attribute arithmetic—but **fully query-aware** as it sets targets based on user context.

Together, they triangulate: if all four agree a fragrance is the best match, confidence is high. If they disagree, the weighted sum finds the centroid, which is still better than any single signal alone.

**Why not train a single end-to-end model?**

An end-to-end model (outfit image + occasion text → fragrance embedding, trained to minimize distance to correct fragrances) would be ideal. It would learn the optimal representation jointly. But it requires:
- Thousands of labeled `(outfit, occasion, correct_fragrance)` training pairs
- A training loop (GPU time, hyperparameter tuning, evaluation)
- A dataset that doesn't exist for this domain

The retrieval + fusion approach achieves the same goal without labeled training data: it leverages models pre-trained on related tasks (general text retrieval, cross-modal retrieval, image classification) and combines their outputs. This is transfer learning at the architecture level rather than the parameter level.

**Why rerank instead of using a better retrieval model?**

Retrieval is fundamentally a recall problem: find every fragrance that might be good. Reranking is a precision problem: from those candidates, find the best one. Optimizing for both simultaneously in a single model is hard. Separating retrieval (fast, recall-optimized, runs over 35K fragrances) from reranking (slower, precision-optimized, runs over 10 candidates) lets each component specialize.

The reranker (Qwen3-VL-Reranker-8B) has far more compute budget per candidate than the retrieval system. It can reason holistically about the outfit image and the fragrance profile in a way that matrix multiply cannot. But it can't run on 35,889 candidates — the latency would be minutes. Running it on 20 candidates is ~2-5 seconds.

This is the standard two-stage retrieval architecture used in production search systems (dense retrieval → cross-encoder reranking). We've adapted it for multimodal fragrance retrieval.

---

## Corpus Structure — Enrichment of the Full Dataset

In Week 5, the "Tier" concept has evolved. While the dataset was originally split into Tiers to manage API costs, the entire 35,889-row corpus (`data/vibescent_enriched.csv`) has now been enriched with LLM-generated attributes and re-embedded.

### The Full Enriched Corpus (35,889 fragrances)

Every fragrance in the database now contains: brand, name, top/middle/base notes, main accords, gender, concentration, and all **LLM-enriched fields** (formality, vibe sentence, mood tags, season inference).

**Why enrich everything?** Coverage. Recommending only the top 2,000 fragrances would systematically exclude niche houses. Week 5's scale-up ensures that any fragrance in the database is reachable with full signal quality. "Tier B" now refers legacy-wise to the high-priority subset used during development, but retrieval and fusion now run over the full enriched 35k rows.

---

## Pre-Processing — The Enrichment Pipeline

This is the most important preprocessing step in the entire system. Everything downstream depends on it.

### Why Raw Notes Fail

Raw Fragrantica data looks like this:

```
Name: Baccarat Rouge 540
Top notes: jasmine, saffron
Middle notes: amberwood, ambergris
Base notes: fir resin, cedar
Accords: sweet, woody, warm spicy, floral
```

A user query looks like this:

```
"I'm wearing a structured black blazer to a gala, feeling mysterious"
```

If you embed both strings and compute cosine similarity, you fail. The fragrance text lives in **chemistry vocabulary space** — ingredient names, accord descriptors. The query lives in **human experience vocabulary space** — occasion, mood, aesthetic intent. These spaces have near-zero lexical overlap and minimal semantic overlap. "Jasmine" is not similar to "mysterious." "Fir resin" is not similar to "gala."

The only way to connect them is to translate the fragrance from chemistry vocabulary into human experience vocabulary. That is what enrichment does.

### What Enrichment Does

For every fragrance in the corpus, an LLM (**Qwen3-8B** locally) receives the raw metadata as a prompt and acts as a fragrance expert:

**Input prompt:**
```
Name: Baccarat Rouge 540
Brand: Maison Francis Kurkdjian
Top notes: jasmine, saffron
Middle notes: amberwood, ambergris
Base notes: fir resin, cedar
Accords: sweet, woody, warm spicy, floral
Gender: unisex
Concentration: EDP
```

**Output (`EnrichmentSchemaV2`):**
```json
{
  "likely_season": "fall",
  "likely_occasion": "Black tie evening",
  "formality": 0.88,
  "fresh_warm": 0.75,
  "day_night": 0.82,
  "character_tags": ["luminous", "opulent", "crystalline", "warm", "modern"],
  "vibe_sentence": "A luminous amber that bridges elegance and sensuality with crystalline precision.",
  "longevity": "long",
  "projection": "strong",
  "mood_tags": ["romantic", "confident", "mysterious"],
  "color_palette": ["amber", "gold", "ivory"]
}
```

The LLM is doing **knowledge distillation** — it has seen thousands of fragrance reviews, editorial pieces, and cultural associations. It can infer that BR540's amber-saffron accord reads as "luxurious" and "formal" in the cultural context of how people actually talk about fragrance.

### Constrained Decoding — Why `outlines` and `vLLM` Are Critical

The enrichment pipeline uses guided decoding (via `outlines` or `vLLM`) for structured output generation. This is not just JSON prompting. These tools intercept the token logit distribution at each decoding step and zero out any token that would produce output incompatible with `EnrichmentSchemaV2`'s JSON schema.

Concretely:
- `likely_season` can only be one of `["spring", "summer", "fall", "winter", "all-season"]` — enforced at the token level.
- `formality` must be a float between 0.0 and 1.0.
- `character_tags` must have 3–5 items.

This matters because enrichment runs on all **35,889 rows**. Guided decoding makes malformed output structurally impossible, ensuring the integrity of the entire corpus without manual cleanup. By Week 5, this allows us to scale beyond the historical 2,000-row "Tier B" limit to the full dataset.

### The `retrieval_text` Construction

After enrichment, all fields — raw and enriched — are concatenated into a single `retrieval_text` string. This is what gets embedded.

```python
# From enrich.py _build_retrieval_text()
"Brand: MFK | Name: Baccarat Rouge 540 | Accords: sweet, woody, warm spicy |
Top: jasmine, saffron | Heart: amberwood, ambergris | Base: fir resin, cedar |
Season: fall | Best for: Black tie evening | Formality: high |
Character: luminous, opulent, crystalline, warm, modern |
Mood: romantic, confident, mysterious | Palette: amber, gold, ivory |
Longevity: long | Projection: strong |
Vibe: A luminous amber that bridges elegance and sensuality with crystalline precision."
```

The `vibe_sentence` is the most semantically rich component — it's the LLM's synthesis of all the metadata into natural language that directly overlaps with how users describe occasions and moods. When someone queries "mysterious evening gala," the embedding of that query is close to the embedding of "luminous amber... elegance and sensuality... crystalline precision" in the shared semantic space.

The design principle: **raw notes describe what a fragrance smells like chemically; retrieval_text describes what a fragrance means culturally.**

### Quality Gates

Two validation layers ensure enrichment quality before artifacts are shipped:

1. **Smoke test** (`smoke_test_enrichment()`): runs a single-row enrichment and validates the result before committing to the full batch. Checks that `vibe_sentence` is non-empty, `likely_season` is set, and `character_tags` has ≥3 items. Aborts early if the LLM is broken.

2. **Success rate gate** (`validate_enrichment()`): after full enrichment, asserts that ≥98% of rows have a non-null `vibe_sentence`. If more than 2% failed, the pipeline aborts — a silently degraded corpus is worse than a failed run because you wouldn't know which rows are broken.

Failure handling within the enrichment loop has two layers:
- First attempt: full prompt
- Second attempt: 70% truncated prompt (handles context-length issues on very verbose fragrance entries)
- Failure: row is logged to a JSONL file and skipped; the pipeline continues

---

## Stage 1 — The Four Retrieval Signals

Each signal independently scores every fragrance in its active corpus and produces a 1-D array of raw scores. These arrays feed into the fusion stage.

### Signal 1 — Text Branch (weight: 0.30)

**The question it answers:** Does the semantic description of this fragrance match the occasion the user described?

**Model:** Qwen3-VL-Embedding-8B (Week 5 upgrade from Qwen3-Embedding-8B at truncated 1024-d — now full 4096-d, same model as multimodal channel)

**How it works:**

1. The expanded occasion phrase is embedded: `embedder.embed_multimodal_documents([query_str])` → vector `q ∈ ℝ⁴⁰⁹⁶`, L2-normalized
2. The pre-computed corpus matrix `CORPUS_EMBEDDINGS ∈ ℝ^(35889 × 4096)` is loaded from `artifacts/qwen3vl_corpus/embeddings.npy`
3. Cosine similarity via matrix multiply (valid because both sides are L2-normalized): `sig_text = CORPUS_EMBEDDINGS @ q.T` → shape `(35889,)`
4. Each scalar is the cosine similarity between that fragrance's enriched text embedding and the query

**Full 4096-d (no truncation):** In Week 4, the pipeline truncated to 1024-d (`raw[:, :1024]`) using Qwen3-Embedding-8B's Matryoshka property. In Week 5, no truncation is applied — Qwen3-VL-Embedding-8B produces 4096-d vectors that are stored and queried at full dimension. The Matryoshka truncation workaround is no longer needed because the VL model is used for both corpus and query embedding.

**L2 normalization:** every embedding is normalized to unit length before storage. Cosine similarity = dot product when `|a|=|b|=1`. Single matrix multiply — O(N × D).

**Why this branch is irreplaceable:**
- It covers all 35,889 fragrances (same as Week 4, but all rows now have enriched embeddings).
- It captures semantic overlap between the expanded occasion phrase and fragrance descriptions.
- CPU fallback available: if Qwen3-VL is unavailable, `SentenceTransformerEmbedder` (configurable via `settings.text_embedding_model`) handles text embedding — the multimodal channel is disabled in this mode.

**Why the text branch alone is insufficient:**
- It cannot see the outfit image — text and multimodal queries from the same occasion context differ only in the visual signal.
- Two fragrances with near-identical vibe sentences but different accord profiles may score identically; the image and structured branches break the tie.

### Signal 2 — Multimodal Branch (weight: 0.25)

**The question it answers:** Does the visual character of this outfit, combined with the occasion context, match the semantic profile of this fragrance?

**Model:** Qwen3-VL-Embedding-8B (MMEB-V2 score 77.8, #1 overall on the cross-modal retrieval benchmark)

**Why this model was chosen over alternatives:**

| Model | MMEB-V2 Score | Deployment |
|---|---|---|
| Qwen3-VL-Embedding-8B | **77.8** (#1) | Local GPU, ~16GB VRAM |
| gemini-embedding-2 | 68.9 | API, quota risk |
| CLIP ViT-L/14 | ~64.x | Local, image-only |

The 9-point gap on MMEB-V2 directly corresponds to cross-modal retrieval quality — the exact task of matching outfit images to fragrance text descriptions. This gap justifies the GPU dependency. Running locally also eliminates API quota risk during the demo.

**How it works:**

1. **Document embedding (offline, pre-computed):** All 35,889 fragrances are embedded as text-only documents with Qwen3-VL-Embedding-8B: `CORPUS_EMBEDDINGS ∈ ℝ^(35889 × 4096)`, saved to `artifacts/qwen3vl_corpus/embeddings.npy`. This is the same matrix used by the text branch — no separate multimodal corpus.

2. **Query embedding (online, at inference):** At query time, outfit image + occasion phrase are jointly embedded: `q_mm = embedder.embed_multimodal_query(text=occasion_phrase, image_path=tmp_img)`. The joint query vector encodes both visual outfit character and semantic occasion intent.

3. **Similarity:** `sig_mm = CORPUS_EMBEDDINGS @ q_mm.T` → shape `(35889,)`. Full coverage, no padding.

**The key insight — what the image adds:** A user typing "evening event" and a user wearing a linen shirt produce the same text query — but different multimodal queries. The linen shirt shifts the query vector away from heavy oriental fragrances toward fresh, light profiles without any text change. The degree to which the image shifts the joint query embedding is the unique contribution of this branch.

**Why the multimodal branch is non-redundant with the image CLIP branch:**
- Multimodal uses open-vocabulary visual-semantic understanding in a continuous 4096-d space — subtle signals like fabric weight, color saturation, and silhouette formality are captured holistically.
- The CLIP branch classifies the outfit into 14 discrete values (3+4+2+3+2 from 5 heads). These two signals are architecturally independent and may disagree.
- Multimodal similarity is holistic and fuzzy. CLIP-to-attribute NLL matching is explicit and interpretable.
- When they agree, confidence is high. When they disagree, the weighted fusion finds the centroid.

**Operational constraint:** Qwen3-VL-Embedding-8B requires ~16 GB VRAM (float16). On insufficient GPU, the multimodal channel is skipped and `sig_mm` is zeroed — the engine degrades gracefully to a 3-signal fusion with `_NO_IMAGE_WEIGHTS` or `_TEXT_ONLY_WEIGHTS` depending on what else is available.

### Signal 3 — Image Zero-Shot Branch (weight: 0.30)

**The question it answers:** What does the outfit's visual character predict about formality, season, and time-of-day — and which fragrances' enrichment attributes match those predictions?

**Model:** CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) — zero-shot classifier matching Neil's original `backend/clip_zero_shot.py` approach. No outfit-specific training required.

**Why CLIP ViT-L/14 instead of CLIP:**
- Alignment with Neil's work: `CLIPImageScorer` uses the same scoring approach as Neil's backend (`score_classification` with multi-prompt averaging), ensuring consistent behavior across both codepaths.
- `NeilCNNWrapper` is provided for backward compatibility: if Neil's trained checkpoint is available, it can be loaded and its output is consumed through the same `ImageHeadProbabilities` interface.
- CLIP ViT-L/14 is GPU-available but also CPU-runnable (slower). The 400 MB model weight advantage of CLIP-base is less relevant now that the system runs on A100/L4.

**Text prompts per class:** 3 prompts per class, averaged before softmax — this matches Neil's approach and reduces sensitivity to any single prompt phrasing:

```python
# Formality — 3 classes × 3 prompts each
FORMAL_PROMPTS = [
    ["a person in ripped jeans and a graphic tee...", ...],   # casual
    ["a person in chinos and a button down shirt...", ...],   # smart-casual
    ["a person in a tailored suit and tie...", ...],          # formal
]
# Season — 4 classes × 3 prompts each
# Time — 2 classes × 3 prompts each
# Gender — 3 classes × 3 prompts each   [NEW in Week 5]
# Frequency — 2 classes × 3 prompts each [NEW in Week 5]
```

All prompt embeddings are computed once at `CLIPImageScorer.__init__()` and cached. At inference, only the image embedding needs to be computed.

#### Five Classification Heads

```
formal:    P ∈ ℝ³   → [P(casual), P(smart-casual), P(formal)]
season:    P ∈ ℝ⁴   → [P(spring), P(summer), P(fall), P(winter)]
time:      P ∈ ℝ²   → [P(day), P(night)]
gender:    P ∈ ℝ³   → [P(male), P(female), P(neutral)]       ← new
frequency: P ∈ ℝ²   → [P(occasional), P(everyday)]           ← new
```

For each class in each head, CLIP computes cosine similarities between the image embedding and all 3 prompts for that class, takes their mean, and stacks the class means. Softmax is applied over the stacked means for each head. This gives a final probability vector per head.

**Why gender and frequency were added:**
- **Gender**: an outfit with clear masculine or feminine signals should not recommend fragrances that are strongly gendered in the opposite direction. Without this, a strongly feminine look could return "brut" or "polo" style masculine fragrances that share the right formality/season.
- **Frequency**: special-occasion outfits (gala gown, tuxedo) should prefer `occasional` fragrances (statement sillage, complex). Everyday outfits should prefer `everyday` fragrances (skin-close, approachable). Without this, the same formal fragrance could surface for both a gala and a "smart casual" office look.

#### Fragrance Scoring via Negative Log-Likelihood

This is the most important architectural decision in the image branch: **it does not do vector similarity.** Instead, it uses the CNN's probability distributions as a probabilistic model of outfit attributes, then scores each fragrance by how well its enrichment attributes match those predictions.

For each fragrance, three enrichment attributes are discretized into class indices:

```python
# formality: float [0,1] → class {0,1,2}
if formality < 0.33:  return 0   # casual
if formality < 0.67:  return 1   # semi-formal
else:                 return 2   # formal

# day_night: float [0,1] → class {0,1}
return 1 if day_night >= 0.5 else 0   # night

# season: string → index {0,1,2,3}
# "all-season" → argmax(season_probs) — takes CNN's most confident prediction
```

The NLL score for a fragrance is:
```
NLL = -(log P_formal[formal_target]
      + log P_season[season_target]
      + log P_time[time_target]
      + log P_gender[gender_target]
      + log P_frequency[frequency_target])

score = exp(-NLL) = P_formal × P_season × P_time × P_gender × P_frequency
```

The score is the **joint probability** that CLIP assigns the outfit to the fragrance's exact attribute bucket across all five dimensions. Maximizing this score = finding fragrances whose enrichment attributes match what CLIP predicts about the outfit.

Per-head weights (`DEFAULT_HEAD_WEIGHTS = {all: 1.0}`) are configurable — setting `gender=0` or `frequency=0` disables individual heads without code changes.

**Why NLL and not L2 distance over probabilities:**
- NLL is the proper probabilistic loss. It is calibrated to the softmax output: a model that outputs `[0.95, 0.03, 0.02]` for the correct class gets low NLL; one that outputs `[0.35, 0.33, 0.32]` gets high NLL. L2 distance would not distinguish these as sharply.
- Using NLL also means the score degrades gracefully when the model is uncertain — a uniform distribution over formal classes gives lower score than a peaked one, correctly expressing lower confidence.

**Known calibration gap:** neural network classifiers are often overconfident — they output `P=0.95` when empirical accuracy is 80%. If the CNN is overconfident, scores will cluster near 0 and 1, reducing discrimination between fragrances. Temperature scaling (`logits / T` before softmax) is the standard fix. It is not currently applied. This is a known limitation.

**Why this branch is non-redundant with multimodal:**
- Multimodal produces a continuous 4096-d embedding — interpretability is zero. You can't inspect what visual feature drove the score.
- The CNN branch is fully interpretable: you can read off `P_formal=[0.05, 0.12, 0.83]` and understand exactly what the model predicts.
- The CNN trained specifically on outfit classification may be more precise on garment-specific visual features (cut, collar, lapel, hem) than a general vision-language model whose training is diluted across all visual domains.
- Most importantly: the CNN connects outfit visual features to fragrance enrichment attributes via a probabilistic bridge. The multimodal branch connects outfit features to fragrance text descriptions via a semantic embedding bridge. These are different paths through different intermediate representations.

**Why this branch is non-redundant with text:**
- Text never sees the image. A user can type "casual event" but wear a tuxedo. Text believes the words. CLIP believes the pixels. These are independent sources of evidence.

#### Score Normalization

```python
likelihoods = [P_formal × P_season × P_time × P_gender × P_frequency for each row]
return min_max_normalize(np.array(likelihoods, dtype=float32))
```

Min-max normalization across the candidate pool means the image branch contributes a **relative ranking signal**, not an absolute probability. The fragrance with the highest joint probability gets score 1.0; the worst gets 0.0. This normalization is applied independently per branch in the fusion stage.

### Signal 4 — Structured Branch (weight: 0.15)

**The question it answers:** How well do this fragrance's pre-computed enrichment attributes align with the user's explicit occasion context?

**Implementation** (`structured_scorer.py`, `compute_structured_scores()`):

Each of the three context fields maps to a float target in the enrichment attribute space:

```python
# Formality targets per event type
_FORMALITY_TARGETS = {
    "Gala": 0.90, "Business": 0.75, "Wedding": 0.85,
    "Date Night": 0.60, "Casual": 0.15, "Festival": 0.25,
}
# Day/night targets per time of day
_DAY_NIGHT_TARGETS = {
    "Morning": 0.10, "Afternoon": 0.30, "Evening": 0.70, "Night": 0.90,
}
# Fresh/warm targets per mood
_FRESH_WARM_TARGETS = {
    "Bold": 0.65, "Subtle": 0.35, "Fresh": 0.05, "Warm": 0.90, "Mysterious": 0.75,
}
```

Each active dimension contributes `1.0 - |fragrance_attribute - target|` to the per-fragrance score. The contributions are averaged over the number of active dimensions. If no context is provided, all fragrances receive 0.5 (neutral). The output is float32 in [0, 1].

**Why this branch is non-redundant:**

The three neural branches (text, multimodal, image) all involve learned inference — embedding geometry, trained weights, probabilistic classifiers. All three can be wrong in the same direction if training data is biased. The structured branch uses **ground truth attributes** computed by an LLM fragrance expert reasoning explicitly about each fragrance's cultural role, note profile, and occasion appropriateness.

`formality=0.88` for Baccarat Rouge 540 is not inferred from pixels or embedding distance. It was produced by a model with fragrance domain knowledge, instructed specifically to rate formality on a 0–1 scale. This is a different epistemic source — reasoned from explicit domain knowledge rather than learned from distributional statistics.

The structured branch is the only signal that directly compares user intent to fragrance attributes without any neural intermediary — pure attribute arithmetic. It is fully query-aware: the score for a given fragrance changes with every query, not a static field average.

---

## Stage 2 — Score Fusion

### Min-Max Normalization (Per Branch)

Before combining signals, each branch's raw score array is independently min-max normalized to [0, 1]:

```python
def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    minimum, maximum = np.min(scores), np.max(scores)
    if maximum == minimum:
        return np.zeros_like(scores)   # degenerate: all scores equal
    return (scores - minimum) / (maximum - minimum)
```

This normalization step is critical and non-obvious. The four signals have completely different ranges:
- Text cosine similarity: typically 0.3–0.8 (all fragrances have some semantic overlap with any query)
- Image NLL-derived score: can range from near-0 to 1.0 depending on CNN confidence
- Structured score: 0.0–1.0 by construction

Without normalization, a branch with naturally higher scores would dominate the fusion regardless of its weight. With normalization, each branch contributes proportionally to its weight, with its highest-scoring fragrance always mapped to 1.0 and its lowest to 0.0.

**Implication:** the branch weights are not sensitivity-equivalent. A branch with a flat score distribution (e.g. the CNN gives 0.5 ± 0.02 for all fragrances) contributes almost no signal even at full weight — because after normalization, all its scores are near 0.5 and only the weight multiplier differentiates them. Branches with higher score variance contribute more discriminative signal per unit of weight.

### Weighted Sum

```python
fused = (wt  * _mm_norm(sig_text)
       + wm  * _mm_norm(sig_mm)
       + wi  * _mm_norm(sig_img)
       + ws  * _mm_norm(sig_s))

# where wt=0.30, wm=0.25, wi=0.30, ws=0.15
```

Weights are loaded from `week3/best_weights.json`. If that file doesn't exist (Week 3 was skipped), equal weights (0.25 each) are used as fallback.

### Weight Selection — Heuristic vs. Learned

The default weights in `fusion.py` are heuristic:

```python
DEFAULT_FUSION_WEIGHTS = {
    "text":        0.30,
    "multimodal":  0.25,
    "image":       0.30,
    "structured":  0.15,
}
```

**Why these values:** text and image each get 0.30 because they represent the two primary modalities (language and vision). Multimodal gets 0.25 because it already partially covers both modalities — it would be double-counted if weighted equally. Structured gets 0.15 because it currently doesn't use query context (see above gap), reducing its discriminative value.

**The correct approach — grid search:** `fusion.py` implements `build_weight_grid()` which generates all valid combinations with a configurable step size (default 0.05), and `grid_search_weights()` which evaluates all combinations against a scoring function. With step=0.05, there are 1771 valid 4-channel weight combinations (all tuples that sum to 1.0). Optimizing over the 20-case benchmark via grid search is the Week 3 deliverable that replaces the heuristic weights.

```python
grid = build_weight_grid(step=0.05)
result = grid_search_weights(score_map, scorer=my_metric, weight_grid=grid)
# result.weights has the empirically best combination
```

The `best_weights.json` artifact from Week 3 stores the grid search result and is loaded at startup in Week 4.

### Top-K Selection

After fusion + hard filter + BM25 blend, `top_k_indices(fused_filtered, 20)` selects the top 20 candidates efficiently via `np.argpartition` (O(N) average). The shortlist was reduced from 50 (Week 4) to 20 (Week 5) because the local Qwen3-VL-Reranker-8B is a precision-first model that works best on a high-quality, tight candidate set.

---

## Stage 3 — Reranking

The reranker is a research component — it does not replace the fusion baseline unless it wins on the 20-case benchmark.

**Model:** `Qwen3VLReranker` (Qwen3-VL-Reranker-8B, local GPU, zero API keys)

**Week 5 change from Week 4:** the reranker was previously `GeminiReranker` (Gemini 3.1 Pro Preview via API). This was replaced with a local Qwen3-VL-Reranker-8B for two reasons: (1) eliminates API quota risk and network latency for the demo; (2) the reranker model natively sees the outfit image — it can score candidates holistically based on visual outfit character, not just text.

**Input to reranker:** top-20 shortlist from fusion (post-filter, post-BM25), each candidate formatted via `build_candidate_text()`:
```
"Baccarat Rouge 540 by MFK | A luminous amber that bridges elegance and sensuality | Occasion: Black tie evening | Notes: jasmine, saffron | amberwood | fir resin"
```
Plus the occasion query string and the outfit image temp file path.

**Output schema:**
The reranker returns a `RerankResult` containing `overall_score`, `formality_score`, `season_score`, `freshness_score`, and `explanation`. In the current Week 5 implementation, the model populates the three attribute scores by mirroring the `overall_score` (the model's confidence in the total match).

**MMR fallback when reranker unavailable:**
```python
top3 = mmr_select(q_emb[0], corpus_emb, top_20_indices, lambda_param=0.5, top_k=3)
```
MMR (Maximal Marginal Relevance) balances relevance vs. diversity. λ=0.5 weights them equally. Without MMR, the top-3 in a dense embedding space often cluster in the same accord family (e.g. three oriental amber fragrances). MMR ensures each pick adds maximally new information relative to what was already selected.

**Why a reranker on top of fusion:** fusion is a weighted average of independent signals. It cannot reason about whether a combination of signals makes sense together, or catch cases where one branch is confidently wrong. The Qwen3-VL-Reranker-8B sees the outfit image directly and scores candidates with holistic visual-semantic reasoning — it can flag "this is a beach outfit and BR540 is a black tie fragrance" in a way that no matrix multiply can.

**Evaluation design — avoiding self-confirmation bias:** the label generator is `gemini-1.5-flash` and the evaluation judge is `gemini-1.5-pro`. Using the same model as both label writer and judge causes the judge to prefer its own generation patterns, not outputs that are actually better for users.

---

## Why Each Branch is Non-Redundant — Summary

The four branches answer four different questions from four different information sources:

| Branch | Primary question | Sees image? | Sees text? | Covers full corpus? | Neural? |
|---|---|---|---|---|---|
| text | Does the fragrance description semantically match the occasion? | No | Yes | Yes (35,889) | Yes |
| multimodal | Does the joint visual-semantic character of the outfit match the fragrance? | Yes | Yes | Yes (35,889) | Yes |
| image (CLIP) | Do the outfit's inferred attributes (formality/season/time/gender/frequency) match the fragrance's enrichment fields? | Yes | No (prompts only) | Yes (all enriched rows) | Yes |
| structured | Do the fragrance's pre-computed enrichment attributes match the user's explicit context? | No | No | Yes (all enriched rows) | No |

No two branches share the same profile. Removing any one creates a systematic blind spot:
- **Remove text:** text-only semantic similarity is lost — the signal that maps occasion language directly to fragrance vocabulary
- **Remove multimodal:** visual signal is lost from the dense retrieval step. CLIP's 14-value discrete classification still carries image signal, but the continuous, open-vocabulary visual-semantic association is gone
- **Remove image (CLIP):** the only signal that reads outfit pixels and maps them to fragrance enrichment attributes disappears. Formality, season, time, gender, and frequency signals from the outfit itself are gone
- **Remove structured:** the only non-neural, ground-truth attribute signal disappears. All remaining signals are neural inferences that can be biased in the same direction by the same distributional assumptions in their training data

---

## 21-Stage Offline Pipeline (Week 2 Notebook)

The preprocessing pipeline that generates all artifacts runs as 21 notebook stages:

| Stage | Purpose | Output |
|---|---|---|
| 1 | Setup: GPU tier detection, disk check, deps | GPU tier string (A100/L4/T4) |
| 2–4 | Load and preprocess full 35,889-row corpus | `vibescent_enriched.csv` DataFrame |
| 5 | Embed corpus text (Qwen3-VL-Embedding-8B) | `artifacts/qwen3vl_corpus/embeddings.npy` (35889 × 4096) |
| 6 | Load enrichment LLM (Qwen3-8B or Gemini fallback) | LLM client in memory |
| 7 | Smoke test enrichment (single row) | Pass/fail gate |
| 8–10 | Enrich full 35,889-row corpus | `vibescent_enriched_full.csv` |
| 11 | Rebuild `retrieval_text` for enriched corpus | Updated DataFrames |
| 12 | Generate `display_text` (optional) | Skipped in production runs |
| 13–15 | Quality validation: raw vs enriched comparison | Quality validation report |
| 16 | Multimodal query probes (text-only queries) | Baseline similarity matrices |
| 17 | Embed documents multimodally (Qwen3-VL) | Part of unified unified matrix |
| 18 | Multimodal query probes (text + image queries) | Quality validation report |
| 19 | Embedding sanity checks (variance test) | Pass/fail gate |
| 20 | Report writer | `results/week5_report.md` sections |
| 21 | Artifact sink: final 4096-d unified matrix | `embeddings.npy` (35889 × 4096) |

**Week 5 implementation:** the active inference corpus is `artifacts/qwen3vl_corpus/embeddings.npy` (35,889 × 4096), generated by Qwen3-VL-Embedding-8B at full dimension. This single unified matrix serves both text and multimodal retrieval channels. All 35,889 fragrances in the dataset carry full enrichment metadata, eliminating the historical "Tier B" quality gap.

**Stage gate mechanism:** each expensive stage (embedding, enrichment) writes a `manifest.json` on completion. `stage_complete(stage_id, artifacts_dir, pipeline_version)` checks for this manifest and skips the stage if it exists. Re-running the notebook after a crash resumes from the last completed stage without re-embedding.

**GPU tier routing:** GPU VRAM detection at Stage 1 determines which stages run:
- A100 (≥35 GB): all stages run at full capacity
- L4 (≥20 GB): Qwen3-VL embedding runs, but at smaller batch sizes
- T4 (<20 GB): multimodal embedding is skipped (Stage 17 no-ops); enrichment falls back to Gemini API

**Checkpoint-based recovery:** `embed_corpus()` writes partial embedding matrices every 100 batches. On crash, `embed_corpus_resume()` globs for checkpoint files, concatenates them in order, and returns the partial result with the next batch index. This prevents losing hours of GPU work to Colab disconnects.

---

## Inference Engine — `VibeScoreEngine`

`src/vibescents/engine.py` implements `VibeScoreEngine`, which satisfies the `RecommendationEngine` protocol and is injected into the FastAPI app via `create_configured_app()`. All heavy models are lazy-loaded on first request and cached for the lifetime of the process.

### Startup

`VibeScoreEngine.from_artifacts()` loads:
- `artifacts/qwen3vl_corpus/embeddings.npy` → L2-normalized float32 matrix, shape `(N, 4096)`
- Corpus metadata CSV → corpus DataFrame (N rows, all enrichment fields)
- Builds `BM25CorpusScorer` from `retrieval_text` column if `rank_bm25` is installed

Both artifacts must be present before the server starts. The `/healthz` endpoint returns `{"status": "ok"}` once loaded.

### Inference Path

1. **Build `query_str`** via `context_to_query_string()` — expands event/mood/time to rich phrases joined by `|`; appends `customNotes` if set
2. **Text channel** — `embedder.embed_multimodal_documents([query_str])` → cosine similarity over full corpus matrix (35,889 × 4096). Skipped if no GPU; falls back to `SentenceTransformerEmbedder` (CPU-compatible) if available
3. **Multimodal channel** — write image bytes to temp file → `embedder.embed_multimodal_query(text, image_path)` → cosine similarity over same corpus matrix. Skipped if no GPU or no image
4. **Image channel** — `CLIPImageScorer.score_image(image_bytes)` → `ImageHeadProbabilities` (5 heads) → vectorised 5-head NLL over corpus. Skipped if torch/transformers unavailable
5. **Structured channel** — `compute_structured_scores(context, corpus_df)` — always available, pure arithmetic
6. `_fuse()` selects the appropriate fallback weight set and combines min-max-normalized scores
7. **Hard filter** — `_hard_filter(ctx)` builds a boolean mask; zeroed rows excluded from top-k (safety guard: bypassed if <3 rows pass)
8. **BM25 blend** — `fused_filtered = 0.9 × fused_filtered + 0.1 × min_max(bm25.score(query_str))` (no-op if `rank_bm25` unavailable)
9. `top_k_indices(fused_filtered, 20)` → 20-candidate shortlist
10. **Reranker** (`Qwen3VLReranker`) — re-scores top-20, returns top-3 indices. Falls back to `mmr_select(λ=0.5)` on failure or unavailability
11. Map indices to DataFrame rows → build `FragranceRecommendation` objects → return `RecommendResponse`

### Graceful Degradation

The engine ships four fallback weight sets tuned for each combination of missing channels:

| Available channels | Weights |
|---|---|
| All four | text=0.30, multimodal=0.25, image=0.30, structured=0.15 |
| No image | text=0.40, multimodal=0.35, structured=0.25 |
| No multimodal | text=0.45, image=0.40, structured=0.15 |
| Text + structured only | text=0.80, structured=0.20 |
| Image + structured only | image=0.70, structured=0.30 |
| Structured only | uniform 1.0 |

The reranker and MMR diversification also degrade gracefully: if the reranker fails (GPU OOM, model not loaded), `mmr_select` runs automatically. If `q_emb` is unavailable (no embedder), the top-3 of the shortlist is returned directly by score.

---

## Frontend — Next.js Luxury Editorial UI

The frontend (`app/`, `components/`, `lib/`) is a Next.js 14 App Router application styled as a high-fashion editorial experience. It is the primary demo artifact.

### Pages

| Route | Description |
|---|---|
| `/` | Landing — cinematic hero with CTA |
| `/demo` | Two-column interactive demo — inputs left, results right |
| `/model` | How It Works — pipeline diagram and methodology prose |

### Demo Flow

1. User uploads an outfit image (`OutfitUploader` — drag-and-drop, base64 conversion)
2. User selects optional context: event type, time of day, mood (pill selectors), and sets a budget (range slider)
3. Submit triggers `POST /api/recommend` — the Next.js route handler orchestrates both backends
4. Results appear as animated `FragranceCard` components showing: name, house, key accords, curator's note, price, store, and "Acquire Scent" purchase link

### Design System

- **Color:** near-black (`#0a0a0a`) background, warm off-white text (`#f5f0e8`), gold accent (`#c9a96e`)
- **Type:** Cormorant Garamond (display), DM Sans (body), DM Mono (scores/prices)
- **Motion:** Framer Motion, expo-out easing `[0.16, 1, 0.3, 1]`, 80ms stagger on card reveals

### `/api/recommend` Orchestration (route.ts)

The Next.js route handler (`app/api/recommend/route.ts`) bridges the frontend and the two backends:

1. Receives `RecommendRequest` (base64 image, mimeType, context, budget)
2. Strips `data:image/...;base64,` prefix from image string
3. `POST http://localhost:8000/recommend` → `RecommendResponse` (name, house, score, notes, reasoning, occasion)
4. Extracts perfume names → `POST http://localhost:8001/search` with budget cap → pricing data per perfume
5. Merges pricing fields (`price`, `store`, `purchaseUrl`, `thumbnail`, `inBudget`) into each `FragranceRecommendation`
6. Returns merged response; if scraper fails (missing SERPAPI_KEY, timeout), returns ML results without pricing

---

## Scraper Pipeline

The scraper (`src/vibescents/perfume_scraper.py` + `scraper_app.py`) queries the SerpAPI Google Shopping endpoint to find real-time pricing for the recommended fragrances.

### How It Works

1. For each perfume name (e.g. `"Maison Francis Kurkdjian Baccarat Rouge 540"`), queries Google Shopping via SerpAPI with `q="{name} perfume fragrance"` and `num=20`
2. Filters results by budget: only items where `parsed_price ≤ budget` are kept
3. Sorts remaining results by store priority (Sephora > Nordstrom > Macy's > …) then by price ascending
4. Returns the top result per perfume (or `None` if no results within budget)

### Configuration

`SERPAPI_KEY` must be set in `.env`. Without it, `search_perfume()` raises `EnvironmentError` which the scraper API returns as a 500, which the Next.js route handles by falling back to ML-only results. Pricing fields display as "Price Unavailable" and the purchase button is disabled.

---

## Model Table

| Role | Model | Provider | Key Metric | Notes |
|---|---|---|---|---|
| Text embedding (query) | Qwen3-VL-Embedding-8B | Local GPU / HuggingFace | MMEB-V2 77.8 (#1) | ~16 GB VRAM bfloat16; 4096-d full; disabled on CPU |
| Multimodal embedding (query) | Qwen3-VL-Embedding-8B | Local GPU / HuggingFace | MMEB-V2 77.8 (#1) | Same model instance; image+text joint embedding |
| Corpus embedding (offline) | Qwen3-VL-Embedding-8B | Local GPU / HuggingFace | — | Generates `artifacts/qwen3vl_corpus/` (35,889 × 4096) |
| Image zero-shot classifier | CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) | Local (CPU+GPU) | — | 5 heads, 3 prompts per class, averaged before softmax |
| Image CNN (optional) | ResNet-50 + CLIP hybrid (`NeilCNNWrapper`) | Local GPU | — | Loads Neil's checkpoint if provided; same `ImageHeadProbabilities` interface |
| Enrichment LLM | Qwen3-8B | Local GPU (outlines) | — | Falls back to Gemini flash on T4 |
| Enrichment fallback | gemini-flash (latest) | Google API | — | Structured output via response_schema |
| Reranker | Qwen3-VL-Reranker-8B | Local GPU | — | In production path on A100; skipped gracefully on VRAM shortage → MMR fallback |
| Evaluation judge | gemini-1.5-pro | Google API | — | Different from label generator |
| Label generator | gemini-1.5-pro-preview | Google API | — | 3-run majority vote per benchmark case |

---

## Shared Schema — `EnrichmentSchemaV2`

Every fragrance must have all of these fields after enrichment:

| Field | Type | Range / Values | Generated by |
|---|---|---|---|
| `likely_season` | `Literal` | spring / summer / fall / winter / all-season | LLM enrichment |
| `likely_occasion` | `str` | Free text, e.g. "Black tie evening" | LLM enrichment |
| `formality` | `float` | 0.0 (casual) – 1.0 (black tie) | LLM enrichment |
| `fresh_warm` | `float` | 0.0 (crisp/fresh) – 1.0 (warm/cozy) | LLM enrichment |
| `day_night` | `float` | 0.0 (daytime) – 1.0 (evening/night) | LLM enrichment |
| `character_tags` | `list[str]` | 3–5 adjectives | LLM enrichment |
| `vibe_sentence` | `str` | One synthesized sentence | LLM enrichment |
| `longevity` | `str` | short / moderate / long | LLM enrichment |
| `projection` | `str` | intimate / moderate / strong | LLM enrichment |
| `mood_tags` | `list[str]` | ≥1 mood-oriented tag | LLM enrichment |
| `color_palette` | `list[str]` | ≥1 color descriptor | LLM enrichment |
| `gender` | `GenderLabel` | male / female / neutral | LLM enrichment (default: `"neutral"`) |
| `frequency` | `FrequencyLabel` | occasional / everyday | LLM enrichment (default: `"everyday"`) |

These fields feed three different downstream consumers:
1. **`retrieval_text`** construction (`enrich.py`) — all fields contribute to the embeddable string
2. **Structured branch scoring** — `formality`, `fresh_warm`, `day_night`, `likely_season` are used for attribute matching
3. **Image CLIP scoring** — `formality`, `day_night`, `likely_season`, plus `gender` and `frequency_of_use` where present, are the fragrance-side targets for 5-head NLL computation

---

## Known Gaps and Limitations

### Model Load Latency (Inference)

Loading two 8B models (Qwen3-VL-Embedding and Qwen3-VL-Reranker) plus CLIP ViT-L/14 into VRAM takes ~45–90 seconds. While once loaded inference is fast, the initial server boot time is a bottleneck in short-lived environments.

### CLIP Classifier Calibration

The CLIP 5-head softmax probabilities are likely overconfident (standard neural network behavior, confirmed in prior CLIP zero-shot work). Temperature scaling should be applied to calibrate probabilities before NLL computation. Current behavior: NLL scores may cluster near extremes, reducing discrimination. Multi-prompt averaging (3 prompts per class) partially mitigates overconfidence by averaging over phrasing variants, but does not substitute for proper calibration.

### Discretization Cliff Effects

The CNN attribute-to-fragrance matching discretizes continuous floats at fixed thresholds: formality 0.33/0.67, day_night 0.5. A fragrance with `formality=0.66` and one with `formality=0.68` map to different classes despite being nearly identical. This introduces cliff effects that can rank very similar fragrances very differently. Soft matching (interpolating between classes) would fix this.

### Multimodal Embedding Requires GPU at Inference

The Qwen3-VL-Embedding-8B model requires ~16 GB VRAM and is skipped on T4/CPU, producing a zero multimodal signal. The engine selects `_NO_MULTI_WEIGHTS` automatically in this case. T4 demos and A100 demos produce qualitatively different recommendations — the multimodal channel's absence removes the strongest cross-modal signal. This should be documented clearly in presentation materials.

---

## Benchmark Structure

20 end-to-end evaluation cases. Each case:

```python
class BenchmarkCaseLabel(BaseModel):
    case_id:                  str
    occasion_text:            str
    target_formality:         str      # "formal", "semi-formal", "casual"
    target_season:            str      # "spring", "summer", "fall", "winter"
    target_day_night:         str      # "day", "night"
    target_fresh_warm:        str      # "fresh", "warm"
    acceptable_accords:       list[str]
    acceptable_note_families: list[str]
    disallowed_traits:        list[str]
    example_good_fragrances:  list[str]
    confidence:               float    # 0.0–1.0, ≥0.6 required
```

**Label generation:** 3 independent runs of `gemini-1.5-flash` per case. Cases kept only if there is strong agreement across runs (majority vote on all categorical fields). Confidence <0.6 → case is discarded or reworked.

**Primary scoring (metadata-based):**
- `attribute_match@3` / `@5`: do top-3/5 results match target formality, season, day/night, fresh/warm?
- `neighborhood_hit@3` / `@5`: do top-3/5 results fall in an acceptable accord/note-family neighborhood?

**Secondary scoring (LLM judge):**
- Judge: `gemini-1.5-pro` — different from label generator to avoid self-confirmation bias
- Judge sees: outfit image, occasion text, fusion top candidates, reranker top candidates
- Judge outputs: preferred shortlist, rationale, confidence

**Presentation rules (from Gavin):**
- Do not present AI-generated benchmark labels as human ground truth
- Do not evaluate on hand-picked wins only — run all 20 cases
- Do not add architecture complexity before the baseline is benchmarked
- Reranker does not ship unless it wins on the benchmark

---

## Occasions Corpus

8 occasion descriptions covering the full formality spectrum. Used as test queries throughout the pipeline:

| ID | Description | Cluster |
|---|---|---|
| `casual_day` | Relaxed daytime — weekend errands, lunch with friends | Casual/social |
| `creative_office` | Creative industry office — expressive but professional | Mid-formal |
| `business_dinner` | Business dinner — polished, confident, evening | Formal |
| `wedding_guest` | Wedding guest — elegant, celebratory, occasion-appropriate | Formal |
| `black_tie` | Black tie evening — formal, refined, commanding | Formal |
| `streetwear_night` | Streetwear night out — bold, urban, youthful | Casual/social |
| `summer_party` | Summer outdoor party — fresh, playful, social | Casual/social |
| `editorial` | Fashion editorial — avant-garde, artistic, attention-commanding | Mid-formal |

From the Week 2 embedding heatmap (cosine similarity between occasion embeddings):
- Tightest pair: `business_dinner` ↔ `creative_office` at 0.722
- Most formal cluster: `black_tie`, `business_dinner`, `wedding_guest` all ≥ 0.683
- Widest gap (formality poles): `editorial` ↔ `casual_day` at 0.512

Expected retrieval behavior: `casual_day` and `black_tie` should return near-disjoint fragrance sets from the enriched index.

---

## Artifact Contract

All pipeline artifacts follow this structure:

| Type | Format | Path | Notes |
|---|---|---|---|
| Corpus embedding matrix | `.npy` float32 | `artifacts/qwen3vl_corpus/embeddings.npy` | Generated by offline pipeline |
| Corpus metadata | `.csv` | `artifacts/qwen3vl_corpus/metadata.csv` | Paired with embedding matrix |
| Full enriched dataset | `.csv` | `data/vibescent_enriched.csv` | 36K rows, all EnrichmentSchemaV2 fields |
| Occasions embeddings | `.npy` | `artifacts/occasions/embeddings.npy` | Week 2 CLIP-based; 8 occasion queries |
| Pipeline manifest | `.json` | `artifacts/*/manifest.json` | Stage gate: model, commit SHA, row count, dims, timestamp |
| Benchmark briefs | `.json` | `data/benchmark_briefs.json` | 20-case evaluation set |

Every `.npy` artifact has a companion `manifest.json` that stores: model name, commit SHA, row count, dimensions, creation timestamp, pipeline version. `stage_complete()` uses this manifest to determine whether to re-run a stage.

---

## Week Deliverable Status

### Week 2 (deadline: April 12, 2026)

| Deliverable | Owner | Status |
|---|---|---|
| Fragrance dataset selected and cleaned | Darren | ✓ `vibescent_500.csv` |
| Shared schema locked | Gavin | ✓ |
| Fragrance enrichment pipeline | Harsh | ✓ `enrich.py` |
| Occasions embeddings + heatmap | Harsh | ✓ `artifacts/occasions/` |
| Raw fragrance embeddings | Harsh | ✓ (notebook Stage 5) |
| Enriched fragrance embeddings | Harsh | ✓ (notebook Stages 10–14) |
| Multimodal doc embeddings (Qwen3-VL) | Harsh | ✓ (notebook Stage 17) |
| RAW vs ENRICHED retrieval comparison | Harsh | ✓ (notebook Stage 15) |
| Image scoring branch | Neil | Replaced by CLIP zero-shot in production (see `image_scoring.py`) |
| `results/week2_report.md` | Harsh | ✓ |

### Week 3–4 (completed)

| Deliverable | Owner | Status |
|---|---|---|
| 4-channel `VibeScoreEngine` | Harsh | ✓ `src/vibescents/engine.py` |
| `CLIPImageScorer` (zero-shot image branch) | Harsh | ✓ `src/vibescents/image_scoring.py` |
| `compute_structured_scores` (query-aware) | Harsh | ✓ `src/vibescents/structured_scorer.py` |
| FastAPI ML backend (`/healthz`, `/recommend`) | Harsh | ✓ `src/vibescents/backend_app.py` |
| SerpAPI pricing scraper + FastAPI wrapper | Harsh | ✓ `src/vibescents/perfume_scraper.py` + `scraper_app.py` |
| 20-case benchmark | Gavin | ✓ `data/benchmark_briefs.json` |
| Fusion weight grid search | Harsh | ✓ (heuristic defaults in `engine.py`) |
| Reranker (research) | Harsh | ✓ `src/vibescents/reranker.py` — not in production path |
| pytest suite (16 test files) | Harsh | ✓ `tests/` |

### Week 5 (current)

| Deliverable | Owner | Status |
|---|---|---|
| Qwen3-VL unified corpus re-embedding (all 35,889 rows, 4096-d) | Harsh | ✓ `artifacts/qwen3vl_corpus/embeddings.npy` |
| BM25 post-fusion blend (`BM25CorpusScorer`, 10% weight) | Harsh | ✓ `src/vibescents/bm25_scorer.py` |
| Hard filter `_hard_filter()` (Gala/Casual/Fresh/Warm gates) | Harsh | ✓ `src/vibescents/engine.py` |
| CLIP 5-head image scorer (added gender + frequency heads) | Harsh | ✓ `src/vibescents/image_scoring.py` |
| Qwen3-VL-Reranker-8B integration (local, MMR fallback) | Harsh | ✓ `src/vibescents/reranker.py` |
| EnrichmentSchemaV2: `gender` + `frequency` fields | Harsh | ✓ `src/vibescents/enrich.py` |
| Next.js luxury editorial frontend | Darren | ✓ `app/`, `components/`, `lib/` |
| `start.sh` multi-service orchestration | Harsh | ✓ |
| Demo validation and presentation | All | ⏳ In progress |

---

## Design Rationale — Anticipated Grilling Questions

Every question below is one a professor, recruiter, or teammate could reasonably ask about the system. Answers are written to be technically defensible, not just plausible.

---

### Fusion Architecture

**Q: Why a weighted sum instead of cross-attention or a learned fusion model?**

Cross-attention is architecturally superior — it would let each branch's contribution adapt per query based on what the other branches signal. But it requires training data to learn the attention weights: you need labeled `(outfit, occasion) → correct fragrances` pairs. We have 20 benchmark cases. Cross-attention has O(d²) parameters in the attention projection matrices. Training a 4-head attention layer on 20 examples produces a model that memorizes the training set and generalizes to nothing.

The weighted sum has zero trainable parameters. Its only degrees of freedom are the four scalar weights, which are optimized via grid search over 1,771 combinations on the 20-case benchmark. Grid search at step=0.05 finds the globally optimal weights within the quantization error — it doesn't get stuck in local minima. For four weights, this is feasible and correct. Gradient descent would find the same answer with more complexity and no added benefit.

The weighted sum is also the correct prior when you don't have evidence that the branch contributions should be query-adaptive. If you can't prove cross-attention helps on a benchmark, you shouldn't pay the training-data cost to use it.

**Q: Why are the initial heuristic weights 0.30 / 0.25 / 0.30 / 0.15 and not equal (0.25 each)?**

Equal weights are the maximally uninformed prior — they assume all signals are equally informative. The heuristic weights express a prior based on signal properties:

- Text (0.30): covers 35,889 fragrances — the only signal with full corpus coverage. High weight because it is the primary retrieval signal for most queries.
- Image CLIP (0.30): the only signal that reads the outfit's visual attributes explicitly. Equal to text because visual information and textual information are the two primary modalities.
- Multimodal (0.25): covers both modalities simultaneously across the **full 35,889-row corpus** but partially overlaps with the text branch signal. Slightly lower weight to avoid double-counting the shared text component.
- Structured (0.15): lower weight because it relies on deterministic attribute arithmetic rather than semantic embedding, though the signal is high-quality and **fully query-aware**. It now covers the **full 35,889-row enriched corpus**.

The four weight sets (`_FULL_WEIGHTS`, `_NO_IMAGE_WEIGHTS`, `_NO_MULTI_WEIGHTS`, `_TEXT_ONLY_WEIGHTS`) are hardcoded constants in `engine.py` — not loaded from any external file. Equal weights (0.25 each) are the fallback of last resort if an unknown channel combination is requested.

**Q: Why min-max normalization instead of z-score normalization or softmax?**

Three options:

1. **Min-max** (`(x - min) / (max - min)`): maps each signal to [0, 1] with the best and worst fragrances anchored at the extremes. Preserves the rank ordering within each branch exactly. The fusion then combines ranks-in-[0,1]-space. Simple, deterministic, no distributional assumption.

2. **Z-score** (`(x - μ) / σ`): centers at zero, unbounded range. The problem: a signal with very low variance (e.g., structured scores cluster around 0.50 ± 0.02) gets amplified dramatically — small differences in score become large differences in z-score. This would let a nearly flat signal dominate the fusion after normalization. Min-max is more conservative: a flat signal stays flat.

3. **Softmax** (`exp(x) / Σexp(x)`): converts scores to a probability distribution summing to 1. Softmax over 35,889 items is well-defined but has a temperature sensitivity problem — if scores are similar, softmax output is nearly uniform; if one score is much higher, it dominates. This non-linearity can cause a single outlier to collapse the distribution.

Min-max is chosen because it makes no distributional assumption, preserves within-branch ranking exactly, and handles the case of near-zero variance gracefully (degenerate: returns all zeros).

**Q: Why is branch independence assumed? Are the branches actually independent?**

The weighted sum implicitly assumes the four signals are independent — otherwise adding them causes double-counting. This assumption is approximately true but not perfectly true:

**Where independence holds:**
- Text encodes semantic content from the occasion description. Image CNN reads visual pixels. These information sources are genuinely different: text is what the user says, pixels are what the outfit looks like.
- Structured attributes were computed by an LLM from fragrance metadata — not from the query or the image. That's a different epistemic source from all three neural branches.

**Where independence partially breaks down:**
- Text and multimodal both use Qwen3 architecture family models and both process the occasion text as part of the query. The multimodal query embedding and the text query embedding will be correlated, particularly when the outfit image is low-information (plain background, minimal visual complexity). In the extreme case of a blank white pixel (the smoke test uses this), the multimodal branch degrades to a near-text-only signal.
- Text and image CNN both ultimately score against fragrance enrichment attributes — text via embedding similarity to retrieval_text (which contains enrichment fields), CNN via direct attribute comparison. For fragrances with high formality and high vibe scores, both branches will tend to rank them high.

**What this means in practice:** the fusion is not a statistically rigorous independence assumption — it's an engineering approximation. The correlations are partial and the branches still carry unique information. For a first system, this is appropriate. A production system would measure pairwise branch correlation on the benchmark and possibly apply decorrelation (e.g. PCA on the stacked signal matrix).

**Q: Why is the image branch scored via NLL rather than embedding the outfit image into the same 4096-d space and doing cosine similarity?**

Two different answers depending on which branch you're comparing to:

**vs. multimodal branch:** embedding the outfit image in the same 4096-d space IS what the multimodal branch does via Qwen3-VL. The multimodal branch produces a joint image+text query vector and does cosine similarity. The CLIP image branch is a deliberately different architectural choice — it classifies rather than embeds. The reason: Qwen3-VL embeddings are opaque. You can't inspect what a cosine similarity score of 0.73 means in terms of outfit attributes. `CLIPImageScorer` tells you `P_formal = 0.83` — you know exactly what the model predicted and can debug it. Both signals coexist because they capture different aspects: continuous semantic similarity (multimodal) vs. discrete attribute matching (CLIP).

**vs. pure cosine:** if you embedded the outfit image and compared it directly against fragrance embeddings, you'd be comparing outfit pixels to fragrance text — a cross-modal mismatch unless you use a joint embedding model (which is exactly what the multimodal branch does). The CLIP branch takes a different approach: classify the outfit into a discrete attribute space (formality, season, time, gender, frequency), and match those attributes against fragrance enrichment values. This is interpretable in a way that direct cross-modal cosine similarity is not.

**Q: Why joint probability (`P_formal × P_season × P_time × P_gender × P_frequency`) for image scoring instead of average?**

The joint probability is mathematically correct under the assumption that the five head predictions are conditionally independent given the outfit. It answers: "what is the probability that this fragrance is correct on ALL FIVE dimensions simultaneously?"

The sum/average asks: "on average, how well does this fragrance fit?" A fragrance with `P_formal=0.9, P_season=0.05, P_time=0.9, P_gender=0.8, P_frequency=0.7` would score `0.67` by average but `0.0227` by product. The product correctly penalizes a fragrance that fails on one dimension even if it succeeds on others. A winter fragrance recommended for a summer outfit should score near zero — the product enforces this; the average doesn't.

The tradeoff: the product is more sensitive to overconfident low probabilities. If CLIP assigns `P_season = 0.02` for the correct season (miscalibration), the product collapses. This is a real risk and is why temperature calibration matters. With 5 heads, the product can reach values as low as `0.05^5 ≈ 3×10⁻⁷` for a fragrance that matches poorly on all five dimensions — the dynamic range is wider than with 3 heads, which helps discrimination but amplifies calibration errors.

**Q: Why `argpartition` for top-K selection instead of full sort?**

`np.argpartition(-fused, k)[:k]` runs in O(N) average time (introselect algorithm). `np.argsort` runs in O(N log N). For N=35,889 and k=20, this is about a 11× speedup on the selection step. The result from `argpartition` is unordered — the top 20 are identified but not ranked among themselves. A secondary sort on just those 20 elements costs O(k log k) ≈ O(20 × 4) = negligible. Total: O(N) + O(k log k) vs O(N log N).

At N=35,889, the difference is small in absolute milliseconds. The choice reflects good algorithmic practice: don't sort what you don't need to sort.

---

### Model Selection

**Q: Why Qwen3-Embedding-8B for text instead of OpenAI `text-embedding-3-large` or `voyage-3-large` (the original plan)?**

The original plan specified `voyage-3-large` (MTEB 68.32, #1 English at time of planning). Qwen3-Embedding-8B was chosen instead for:

1. **No API dependency at inference time.** `voyage-3-large` requires a network call to Voyage AI per query. At demo time, API latency is unpredictable. Qwen3-Embedding runs locally on the same GPU as everything else — no external dependency.

2. **Cost.** Embedding 35,889 fragrance texts once via API costs real money. Pre-computing locally on a free Colab/Kaggle GPU is free. Inference embedding (one query per request) is also free locally.

3. **Comparable quality.** Qwen3-Embedding-8B's MTEB score is within the top tier. For the retrieval task of matching occasion descriptions to fragrance descriptions, the semantic understanding of a SOTA embedding model is more important than single-digit MTEB score differences.

4. **Matryoshka support.** Both models support Matryoshka truncation to 1024-d, enabling cross-modal compatibility with the multimodal model.

`text-embedding-3-large` (OpenAI) was not considered — same API dependency concern, and OpenAI embeddings are not open weights (can't inspect or fine-tune).

**Q: Why Qwen3-VL-Embedding-8B instead of CLIP directly for multimodal?**

CLIP (ViT-L/14) was designed for zero-shot image classification — aligning image embeddings with short noun-phrase text descriptions ("a photo of a cat"). It was not trained for document retrieval tasks where the text side is a 500-token structured description.

Qwen3-VL-Embedding-8B is trained specifically for cross-modal retrieval: it learns to embed long documents and multimodal queries into a shared space optimized for retrieval metrics. MMEB-V2 score (77.8 for Qwen3-VL vs ~64.x for CLIP ViT-L/14) directly measures cross-modal retrieval — this is not a proxy benchmark, it is the exact task we're solving.

Mechanistically: CLIP's text encoder is limited to 77 tokens (CLIP's training context length). Our `retrieval_text` strings exceed 200 tokens. CLIP would truncate them, losing the enrichment fields. Qwen3-VL handles full-length documents.

**Q: Why ResNet-50 as the CNN backbone instead of a more modern architecture (EfficientNet, ViT-B)?**

Neil's model uses ResNet-50 for the convolutional backbone. The reasons are pragmatic and defensible:

1. **Complementarity with CLIP.** CLIP ViT-L/14 is already handling the global semantic understanding. ResNet-50 adds local spatial features (texture, edge structure, garment detail) that ViT processes less precisely due to its patch-based attention. The two backbones complement each other rather than being redundant.

2. **Pretrained weights availability.** ResNet-50 has stable, well-validated pretrained weights on ImageNet. Swapping to EfficientNet-V2 would require re-validating the checkpoint format and fusion layer dimensions.

3. **Proven fine-tuning behavior.** ResNet-50's residual connections make it stable during fine-tuning on small datasets. More modern architectures often require careful hyperparameter tuning to avoid training instability.

4. **The bottleneck is CLIP, not the CNN.** The CLIP ViT-L/14 features (768-d) carry most of the semantic load. ResNet-50 (2048-d spatial features) adds texture and structure details. Even a weaker CNN backbone would contribute meaningfully to the fusion — the absolute quality of the ResNet features matters less than the qualitative difference they add on top of CLIP.

**Q: Why Qwen3-8B for enrichment instead of a much smaller model (Llama-3-8B, Mistral-7B)?**

Enrichment requires domain expertise: the model must correctly infer `formality=0.88` for BR540 vs `formality=0.22` for a beachy aquatic. This is not a task that can be solved by pattern matching on training data — it requires understanding of cultural fragrance norms, occasion dressing conventions, and sensory-to-conceptual translation.

Smaller models (7-8B) tend to produce fluent but inaccurate enrichment on this task — they hallucinate plausible-sounding but wrong formality scores, and their `vibe_sentence` outputs are generic ("a pleasant fragrance for all occasions"). The Qwen3-8B model has enough capacity to have absorbed fragrance-specific cultural knowledge from the training corpus and can make nuanced distinctions.

More critically: enrichment runs once offline. The GPU cost for 35,889 rows is acceptable for a one-time offline job. If this were running live per-request, the 8B model would be necessary.

**Q: Why switch the reranker from Gemini API (Week 4) to Qwen3-VL-Reranker-8B (Week 5)?**

**Week 4 used Gemini 3.1 Pro Preview** (API): top-tier multimodal reasoning, structured output. The problem: API dependency at demo time. A dropped connection, quota exhaustion, or latency spike fails the reranker visibly. The reranker was already research-only — adding a fragile API dependency on top made it worse.

**Week 5 uses Qwen3-VL-Reranker-8B (local GPU):**
1. **Zero API dependency.** No network call at inference. Demo is fully self-contained on A100.
2. **Sees the outfit image natively.** Purpose-built companion to Qwen3-VL-Embedding-8B — trained to score candidates given a query image. Scores on actual outfit pixels, not a text description of them.
3. **VRAM budget fits.** Qwen3-VL-Embedding-8B (~16 GB) + Qwen3-VL-Reranker-8B (~16 GB) ≈ 34 GB total. Fits on A100 80 GB. If VRAM is tight, the reranker is skipped and `mmr_select` runs instead — no hard failure.

The model pair (Embedding + Reranker from the same Qwen3-VL family) is also architecturally coherent: they share the same embedding space, so the reranker can directly compare query embeddings to candidate representations without cross-model mismatch.

**Q: Why use a different model (`gemini-1.5-pro`) as evaluation judge vs the label generator (`gemini-1.5-flash`)?**

Self-confirmation bias. If the same model generates the benchmark labels and then evaluates whether the system's output matches those labels, it will systematically prefer outputs that resemble its own generation style — not outputs that are actually correct.

This is a well-known problem in LLM evaluation: LLM-as-judge studies show that models prefer their own outputs 60–70% of the time even when the other output is objectively better by human standards. Using a different model family breaks this pattern and makes the evaluation more credible.

Additionally, `gemini-1.5-pro` is a stronger reasoning model — it may catch errors in the labels themselves that `gemini-1.5-flash` would not, because it's evaluating from a position of greater capability.

---

### Enrichment Design

**Q: Why use an LLM for enrichment instead of traditional NLP (keyword extraction, TF-IDF, rule-based classifiers)?**

Traditional NLP would give you:
- TF-IDF weights over note vocabulary → tells you which notes are distinctive, not what the fragrance means culturally
- Named entity recognition → extracts brand names, note names, not occasion suitability
- Rule-based classifiers → requires manually encoding "saffron + oud → formal" rules, which is incomplete and unmaintainable at 35,889 fragrances

The enrichment task requires **grounded inference from cultural knowledge**: "what occasion is this fragrance appropriate for?" is not answerable from the note list alone. It requires knowing that saffron + amber has historically appeared in formal Middle Eastern and European fine fragrance traditions, that the `amberwood` accord is associated with evening luxury consumption, that MFK BR540 specifically has become a cultural shorthand for high-status urban formal occasions. This is cultural knowledge that exists in the LLM's training data and not in any rule system or statistical model trained purely on fragrance metadata.

**Q: Why 0–1 floats for formality/fresh_warm/day_night instead of discrete categories (low/medium/high)?**

Two reasons:

1. **Gradient in the structured branch.** Attribute matching in the structured branch computes `|fragrance.formality - target|`. With discrete categories (0, 0.5, 1), a fragrance rated "medium" and one rated "high" have identical distance from any target in the "high" bucket. With floats, `formality=0.88` is measurably closer to a 0.90 target than `formality=0.67`. This makes the attribute scoring sensitive to fine distinctions.

2. **Flexibility in discretization.** The CNN scoring needs to bin fragrances into class indices (0, 1, 2 for formal). With discrete labels, that binning is fixed by whoever assigned the label. With floats, you can adjust the discretization thresholds (`<0.33 → casual`, `0.33–0.67 → semi-formal`, `>0.67 → formal`) without re-running enrichment — just change the threshold in `discretize_formality()`.

The cost: floats require the LLM to output precise decimal values, which LLMs do approximately. A model might output 0.7 vs 0.72 for two similar fragrances with no principled distinction. This is acceptable because the downstream use is relative ranking, not precise calibration.

**Q: Why does `outlines` (constrained decoding) matter vs just asking the LLM to output JSON and validating afterwards?**

Prompt-and-validate has a critical failure mode at scale: the LLM's output is stochastic. Even with a strong system prompt, it will occasionally:
- Omit a required field (`character_tags` silently absent → downstream code gets `KeyError`)
- Output `"formality": "high"` as a string instead of `0.88` as a float
- Produce `"likely_season": "summer/fall"` — not in the allowed Literal set
- Output syntactically invalid JSON (unclosed bracket on token 1800/4096)

With 35,889 enrichment calls, you expect some failure rate even if each individual call succeeds 99% of the time. Prompt-and-validate requires: try → catch exception → retry → catch again → log failure. This is three LLM calls and two exception catches per failed row.

With `outlines`, invalid output is structurally impossible. The token generation process is constrained so that the output is always valid JSON matching `EnrichmentSchemaV2`. No validation needed because correctness is enforced at generation time. For 35,889 rows, this eliminates an entire class of failures.

The cost: `outlines` requires loading the full model through the `outlines.models` API, which adds ~10s overhead per pipeline setup. Worth it for 35,889 rows.

**Q: Why is the 98% success rate threshold for enrichment validation the right value?**

2% failure rate on 35,889 rows = ~718 fragrances with null `vibe_sentence`. Each failed row gets embedded with the raw `retrieval_text` only (no vibe, no mood tags, no enrichment signal). For those 718 fragrances, the embedding quality degrades to near-raw level.

At 98% threshold: ~718 bad embeddings in a corpus of 35,889 is a signal-to-noise ratio that won't noticeably degrade retrieval metrics. Benchmark cases that happen to need one of those 718 fragrances will be affected, but it's unlikely to dominate the 20-case evaluation.

Why not 99% or 100%? Enrichment failures have two causes: (1) rate limiting from Gemini API, (2) truncation issues with verbose fragrance entries. Both are manageable but not eliminatable. A 99% threshold would fail the pipeline on 20 bad rows and force a full re-run. 98% is the practical ceiling given real-world API reliability.

**Q: Why re-run `enrich_dataframe()` with a 70% truncated prompt as the retry?**

LLM API failures on long prompts have two main causes: (1) the prompt exceeds a context limit (hard fail) or (2) the model gets confused by extremely verbose input and produces malformed output. Both are more likely as prompt length increases.

Truncating to 70% of the original prompt sacrifices: the least important fields (usually `category`, `concentration`, `gender` appear near the end of `_build_prompt()`). These are lower-signal for enrichment than `top_notes`, `middle_notes`, `base_notes`, `main_accords`. The retry with 70% loses some context but still contains the core fragrance information. This heuristic resolves ~80% of prompt-length-related failures without requiring a sophisticated field prioritization system.

---

### Corpus and Enrichment Design

**Q: Why enrich the full 35,889-row corpus instead of just a high-quality subset?**

In Week 4, the system only enriched the top 2,000 fragrances ("Tier B"). In Week 5, we scaled to the full 35,889-row dataset. This eliminates the **coverage gap** where niche or obscure fragrances were reachable only via raw text search. By enriching everything:
- **Consistent Signal:** Every fragrance now carries the same 12 enriched semantic attributes (formality, vibe, character tags, etc.).
- **Long-tail Discovery:** A user with a highly specific aesthetic can now find the "perfect" match among 35k fragrances with the same precision as they would for a global bestseller.
- **Fair Comparison:** All four scoring channels (Text, Multimodal, Image, Structured) now operate over the entire database, preventing "popularity bias" where the system would only recommend well-known scents because they were the only ones with rich features.

**Q: Why was `rating_count` originally used to define "Tier B"?**

`rating_count` is a proxy for **documentation quality**. A fragrance with 50,000 ratings has been extensively reviewed and is culturally well-known. Its notes list is likely complete, and there is substantial cultural writing about it that the enrichment LLM has likely seen during training. This ensured high-quality "seed" data during early development. While we now enrich all 35k rows, `rating_count` remains a useful metadata field for sorting and filtering.

**Q: Why require `top_notes`, `middle_notes`, `base_notes`, AND `main_accords` for the best retrieval quality?**

The `retrieval_text` construction pipeline uses all four:
```
Top: {top_notes} | Heart: {middle_notes} | Base: {base_notes}
Accords: {main_accords}
```
Missing `base_notes` means the fragrance's dry-down — often its most distinctive phase — is absent from the embedding. Missing `main_accords` removes the highest-level semantic clustering signal. Either omission degrades the retrieval text enough to affect embedding quality.

---

### Scoring Mathematics

**Q: Why is cosine similarity the right distance metric instead of L2 (Euclidean) or dot product?**

For embeddings, cosine similarity is the standard because it measures only the angle between vectors, ignoring magnitude. Semantically similar texts should point in the same direction in embedding space regardless of their "length" (which can be biased by word count).

The pipeline L2-normalizes all embeddings before storage (`normalize_rows()` in `similarity.py`). For unit vectors: `cos(θ) = a · b` exactly. This allows the matrix multiply `TEXT_EMBEDDINGS @ q.T` to compute cosine similarity instantly. For a 35,889 × 4096 matrix, this is ~147M floating-point multiply-accumulate operations — fast on any modern CPU or GPU.

**Q: Why did Week 4 truncate to 1024-d, and why did Week 5 switch to full 4096-d?**

**Week 4 rationale for truncation (Legacy):**
1. **Cross-model compatibility workaround.** Text branch used Qwen3-Embedding-8B; multimodal branch used Qwen3-VL-Embedding-8B. Truncating both to 1024-d via Matryoshka prefix was a workaround for their different dimensionality, though not a true semantic bridge.
2. Memory. 35,889 × 1024-d saved 4× memory, which mattered on low-VRAM T4 GPUs.

**Week 5 rationale for full 4096-d (Current):**
1. **Single model eliminates the mismatch.** Qwen3-VL-Embedding-8B is now the sole model for both text and multimodal queries. There is no cross-model mismatch to bridge.
2. **Quality at full dimension.** Full 4096-d captures the complete learned representation of the 8B model. For retrieval over 35K rows, additional precision helps distinguish fragrances with subtle differences in character.
3. **Hardware availability.** Memory is acceptable on A100/L4 GPUs. 560 MB for the corpus matrix is well within budget.

The L2 normalization note on the matrix multiply: for 35,889 × 4096 at float32, a single dot product query costs ~147M multiply-accumulates — still fast in a single BLAS call (~5 ms on A100).

**Q: Mathematically, what does L2 normalization of embeddings do and why is it essential before storing?**

L2 normalization maps every embedding vector to the unit hypersphere: `v_norm = v / ||v||₂` so that `||v_norm||₂ = 1`.

Why this matters:
- **Enables dot product as cosine similarity.** For unit vectors `a` and `b`: `a · b = ||a|| × ||b|| × cos(θ) = cos(θ)`. Without normalization, you'd need to divide by magnitudes at query time — a computation on every element of a 35,889-row matrix.
- **Prevents length bias.** Embedding models sometimes produce longer vectors for longer texts. Without normalization, a fragrance with a long `retrieval_text` would artificially score higher against any query, not because it's more semantically relevant but because its embedding has higher magnitude.
- **Pre-computation at storage time.** Normalizing once at embedding time (offline) means the per-query operation is pure matrix multiply. For 35,889 × 4096, this is ~147M floating-point multiply-accumulate operations — fast on any modern CPU or GPU.

**Q: Why use the negative log-likelihood for the image CLIP scoring instead of a simpler formulation?**

NLL has three mathematical properties that make it correct here:

1. **Log-space addition = probability product.** `-NLL = Σ log(P_head[target])`. This is numerically stable — multiplying five small probabilities directly can underflow to zero for dissimilar fragrances, while summing logs never underflows.

2. **Proper probabilistic scoring.** NLL is the standard loss for classification tasks precisely because it is calibrated to the softmax output. A model that outputs the correct class with probability 0.99 gets NLL ≈ 0.01. One that outputs 0.50 gets NLL ≈ 0.69. The penalty grows logarithmically, not linearly — this correctly penalizes uncertainty without catastrophically punishing near-correct predictions.

3. **`exp(-NLL)` is the joint likelihood.** Taking `exp(-NLL) = P_formal × P_season × P_time × P_gender × P_frequency` converts back to probability space, bounded [0, 1]. A score of 0.21 means CLIP assigns 21% joint probability to this fragrance matching the outfit across all five attributes — interpretable and comparable across the corpus.

Alternative: a simpler approach would be `argmax` — take the most likely class for each head and do a binary match against the fragrance attribute. This throws away all probability information. A fragrance in the right class at P=0.51 scores the same as one at P=0.99. NLL uses the full distribution.

---

### CLIP / CNN Architecture

**Q: Why five classification heads (formal, season, time, gender, frequency) for `CLIPImageScorer`?**

The three original heads — formality, season, time-of-day — are the dimensions where **visual appearance provides strong evidence**. You can infer outfit formality from collar, lapels, cut, and fabric. You can infer season from layering, fabric weight, and color palette. You can infer day vs. night from how dressed-up something is.

**Why gender and frequency were added in Week 5:**
- **Gender:** the 3-head system was returning masculine fragrances (Sauvage, Bleu de Chanel) for feminine editorial looks because they share high formality and evening scores. A dedicated gender head prevents this systematic failure.
- **Frequency:** a gala gown and a smart-casual office look can have similar formality scores. Frequency-of-wear separates them: gala → occasional (statement sillage), office → everyday (skin-close, approachable).

**What visual appearance cannot reliably predict:**
- Longevity: a property of the fragrance, not the outfit. The scorer cannot infer "this person wants a fragrance that lasts 12 hours" from pixels.
- Fresh/warm: weak visual correlation (bright → fresh, dark/rich → warm) but too noisy for a reliable head. The structured branch covers this via ground-truth enrichment attributes.

With `CLIPImageScorer` (zero-shot), adding heads costs only new text prompts — no retraining required. The tradeoff: each additional head multiplies the NLL product, amplifying calibration errors and making discrimination sharper simultaneously.

**Q: Why a shared trunk for all heads in Neil's CNN vs. independent classifiers?**

Multi-task learning via a shared trunk has well-established benefits:

1. **Implicit regularization.** The trunk must learn features that are jointly useful for formality AND season AND time prediction. This forces it to learn richer intermediate representations than any single-task classifier. A trunk that only predicts formality might memorize "black clothes → formal" without learning that the silhouette and structure are the real signal.

2. **Data efficiency.** Every labeled training image simultaneously trains all three heads through the shared trunk. With limited training data, sharing is more efficient than training three separate models.

3. **Correlated tasks.** Formality and time-of-day are correlated (formal outfits are more likely to be evening). A shared trunk can implicitly encode this correlation. Separate classifiers would learn it redundantly.

The tradeoff: if the tasks are truly independent and there's abundant training data, a shared trunk can be a bottleneck. For this application, the tasks are correlated enough that sharing is beneficial.

**Q: Why discretize `formality` into 3 classes (not 5 or 10)?**

More classes require exponentially more training data (each new class needs labeled examples). For outfit formality:
- 3 classes (casual / semi-formal / formal) cover the meaningful distinctions that matter for fragrance selection. The fragrance appropriate for "casual" vs "semi-formal" is meaningfully different; "semi-formal class 2 of 5" vs "semi-formal class 3 of 5" is not.
- The enrichment scalar `formality ∈ [0, 1]` already provides continuous precision for attribute matching in the structured branch. The CNN head's job is to provide a discrete visual signal — 3 classes is sufficient for that purpose.
- The cliff-effect limitation of discrete classes (a fragrance at 0.66 vs 0.68 maps to different classes at threshold 0.67) is real. But this is mitigated by the fact that the CNN is one of four signals — edge cases at class boundaries don't dominate the fusion.

---

### Evaluation Design

**Q: Why 20 benchmark cases? Why not 100 or 5?**

- **Why not 5:** Five cases is not statistically meaningful. A system that gets 4/5 right might be genuinely good or might have overfit to those 5 specific cases. Confidence intervals on a 5-case sample are too wide to draw conclusions.

- **Why not 100:** Generating 100 high-quality benchmark cases with 3-run majority-vote label generation requires ~300 LLM calls (at ~1-2s each = ~5-10 minutes just for labels), plus manual review of each case. The cases must cover the full formality spectrum, season space, and day/night dimension without redundancy. Designing 100 non-redundant, high-quality cases is a significant research task. 20 cases, each covering a distinct combination of formality × occasion × season, is achievable within Week 3 scope.

- **Why 20:** enough to compute meaningful `@3` and `@5` metrics with acceptable confidence intervals (±~22% at 95% CI for a binary metric). Enough to show that the system generalizes across the formality spectrum without cherry-picking.

**Q: Why metadata-based primary scoring instead of pure LLM judge?**

LLM judges have known biases: they prefer longer responses, responses that sound confident, and responses that match their training distribution. An LLM judge deciding whether a fragrance recommendation is "good" might prefer recommendations that use prestigious brand names, regardless of actual attribute fit.

Metadata-based scoring is deterministic and auditable: `attribute_match@3` asks "does the top result's formality attribute match the benchmark case's target formality?" This is a binary check computed from structured data. No LLM bias, no length preference, no prestige bias.

The LLM judge (`gemini-1.5-pro`) is used as secondary scoring specifically because it can catch holistic failures that metadata can't — a recommendation that is technically "formal" but completely wrong in mood or accord family. Both scoring methods together are more reliable than either alone.

**Q: Why require 3 independent label generation runs and take majority vote?**

LLM outputs are stochastic. The same prompt with temperature > 0 produces different outputs across runs. For benchmark labels, we want ground truth — not a single sample from a distribution. A label generated once might be:
- Correct and stable (the model consistently agrees)
- Unstable (high-variance across runs → the model is uncertain → the label is ambiguous)
- Consistently wrong (model bias)

Three runs and majority vote filters out high-variance cases (which are discarded or reworked) and increases confidence that stable labels reflect genuine model knowledge. Cases where the model disagrees across runs have confidence < 0.6 and are excluded — they're either ambiguous or poorly specified.

---

### Infrastructure

**Q: Why pre-generate 5 locked demo responses instead of running the full pipeline live for all demo cases?**

The full inference pipeline (text embed → multimodal embed → CNN forward pass → fusion) takes 3-8 seconds on A100. On T4 (multimodal skipped), it's 2-4 seconds. This is acceptable for a live demo.

However, demo situations have specific failure modes:
- The tunnel drops mid-demo
- The GPU runs out of memory after loading all models
- The Colab session disconnects
- The ngrok tunnel rate-limits and blocks the request

For the 5 rehearsed demo cases (wedding/evening/romantic, business/morning/confident, etc.), pre-generated responses are cached in the engine. A cache hit is instantaneous (<1ms) and bypasses all model inference. This makes those 5 specific cases bullet-proof: even if every model fails to load, the demo still works for rehearsed inputs.

Non-rehearsed inputs (a live audience member's novel query) still go through the full pipeline — this is the live demo moment. The locked cache is safety, not a replacement.

**Q: Why dual tunnel (Cloudflare primary + ngrok backup)?**

Single points of failure in live demos are unacceptable. Cloudflare Tunnel and ngrok fail in different ways:
- Cloudflare Tunnel is faster and more stable but requires `cloudflared` binary to be pre-installed and running. It can fail if the binary download is blocked or the tunnel registration fails.
- ngrok is more widely tested in Colab environments but has rate limits on free tier (40 connections/minute) and occasional authentication issues.

Running both provides immediate failover: if the Cloudflare URL stops responding, the presenter switches to the ngrok URL — a 5-second operation vs. restarting the tunnel from scratch (which takes 30-60 seconds). The active URL is printed clearly in Stage 6 output for the presenter.

**Q: Why checkpoint embeddings every 100 batches instead of at the end or every batch?**

Every-100-batch checkpointing is a balance between:

- **Crash recovery cost:** if the embedding job crashes at batch 850 of 1000, you lose at most 100 batches of work (the last checkpoint was at batch 800). Re-running 100 batches ≈ 10-20 minutes on A100.

- **Checkpoint overhead:** writing a numpy file to disk takes ~1-2 seconds for a 100-batch delta at batch_size=32. Every-batch checkpointing would mean ~3-5% of wall time spent on disk I/O rather than embedding. Every-100-batches: <0.1% overhead.

- **Colab disconnect pattern:** Colab typically disconnects after 12-hour idle or when the session is reclaimed. Jobs that run 4-8 hours are at meaningful risk. A 100-batch checkpoint interval ensures that most of the work is recoverable on reconnect.

Checkpoints store only the delta since the last checkpoint (not the cumulative result), so `embed_corpus_resume()` must glob all checkpoint files and concatenate them. This is the design choice in `week2_pipeline.py`.

**Q: Why a manifest.json per artifact instead of a single global state file?**

Each artifact directory (`fragrance_raw_full/`, `fragrance_enriched_2k/`, etc.) has its own `manifest.json`. This enables:

1. **Granular cache invalidation.** Changing the enrichment LLM (e.g., switching from Qwen to Gemini) should only invalidate the enrichment stage and downstream artifacts, not the raw embedding artifacts. With a single global state file, you'd need to manually track which stages depend on which others. With per-artifact manifests, each stage checks only its own manifest.

2. **Parallel development.** Neil's image artifacts and Harsh's text artifacts have independent manifests. Neil can rebuild his artifacts without touching Harsh's, and `stage_complete()` checks the right manifest for each stage.

3. **Provenance.** Each manifest records the exact model name, commit SHA, row count, and creation timestamp. If a retrieval result looks wrong, you can trace back exactly which model version and commit produced the embeddings.

---

### Known Failure Modes and Edge Cases

**Q: What happens when the outfit image is low quality, blurry, or shot against a cluttered background?**

The CNN preprocessing pipeline resizes to 224×224 bicubic. For a blurry image, the resize doesn't help — the CNN receives low-information pixels. The three classification heads will produce flatter probability distributions (higher entropy) because there are fewer discriminative visual features. Flat distributions mean the NLL scoring is near-uniform across all fragrances — the image branch contributes near-zero discriminative signal.

The fusion handles this gracefully: if `sig_img` is near-uniform, after min-max normalization it has near-zero variance. The 30% image weight effectively becomes zero-contributing. The remaining 70% (text + multimodal + structured) drives the recommendation.

The multimodal branch (Qwen3-VL) may be more robust to low-quality images because its training included noisy web images. But a truly low-information image will flatten both vision signals.

**Q: What if the user provides no context — no eventType, no timeOfDay, no mood?**

`context_to_query_string()` in `query.py`:
```python
def context_to_query_string(ctx: ContextInput) -> str:
    parts = [_EVENT_PHRASES.get(ctx.eventType, ctx.eventType),
             _TIME_PHRASES.get(ctx.timeOfDay, ctx.timeOfDay),
             _MOOD_PHRASES.get(ctx.mood, ctx.mood),
             ctx.customNotes]
    filtered = [p for p in parts if p]
    return " | ".join(filtered) if filtered else "elegant versatile fragrance suitable for any occasion"
```

With no context, `occ_text = "general occasion"`. This text is embedded and finds fragrances whose retrieval_text semantically overlaps with "general occasion" — which tends to return versatile, popular fragrances. The text signal degrades to a baseline embedding of a generic phrase.

The multimodal branch still uses the outfit image with "general occasion" as the text component. The visual signal from the outfit still influences which fragrances score highly. The image CNN is unaffected (it never reads the context text).

The structured branch is also unaffected (it currently ignores context anyway). In the fully implemented structured branch, "general occasion" would map to mid-range targets for formality, day_night, and fresh_warm.

**Q: What if the CNN misclassifies the outfit — says it's formal when it's casual?**

The incorrect branch score contributes 30% of the fusion signal in the wrong direction. If the outfit is a casual linen suit misclassified as formal:
- CNN scores high for formal fragrances (wrong)
- Text branch scores high for whatever the user typed as context (could correct this)
- Multimodal branch sees both the linen texture and the occasion text — may partially correct
- Structured branch (as currently implemented) is query-independent — doesn't help

Depending on the magnitude of the misclassification, the fusion may still produce a reasonable recommendation if the text and multimodal signals agree on something different. But a confident CNN misclassification (`P_formal = 0.95` for a casual outfit) will bias the top-50 shortlist. This is an argument for calibration (temperature scaling) and for careful weight selection — if the CNN is known to be less reliable on certain outfit types, its weight should be reduced.

**Q: What if two fragrances score identically after fusion (tie)?**

`np.argpartition` on tied scores is non-deterministic in order (behavior is implementation-defined for equal values). The secondary sort on the top-50 via `np.argsort` is stable for ties — it preserves the order from `argpartition`, which is effectively arbitrary.

For the demo, this is acceptable — ties between near-identical fragrances produce near-identical recommendations. For a production system, you'd want a deterministic tiebreaker (e.g., secondary sort by `rating_count` descending: prefer the more popular fragrance among equals).

**Q: What if the enrichment LLM produces systematically biased formality scores — e.g., always assigns high formality to everything?**

`validate_enrichment()` checks the success rate of `vibe_sentence` (non-null) but does NOT check for distributional correctness. A model that produces `formality=0.90` for every fragrance would pass the 98% success rate check.

Systematic bias in formality scores means:
1. The structured branch (currently broken) would produce near-uniform scores — which is the same as the current broken implementation
2. The CNN image branch compares outfit formality predictions against those biased scores — every fragrance appears "formal" → the CNN can never distinguish

Detection: compute the distribution of enriched `formality` scores across the full corpus and verify it roughly follows the expected distribution (not all-high, not all-low, covers the full range). This should be in `results/week2_report.md` Section 2 (fragrance embedding sanity check) but is currently pending.

**Q: What happens if the corpus embedding matrix and the corpus DataFrame have mismatched row counts?**

In Week 5, the corpus matrix (`artifacts/qwen3vl_corpus/embeddings.npy`) and the metadata CSV are produced from the same source file in the same offline run — row counts should always match. The manifest records `n_rows` for verification.

If they don't match (e.g. the CSV was filtered after embedding, or the matrix came from a different source version), `VibeScoreEngine.from_artifacts()` truncates to the minimum:
```python
_n = min(len(embeddings), len(df))
embeddings = embeddings[:_n]
df = df.iloc[:_n].reset_index(drop=True)
```
This is a defensive guard — you lose tail rows but avoid index-out-of-bounds during all four channel scoring operations. In Week 4, the multimodal branch required an explicit zero-padding step (`sig_mm = np.zeros(len(df)); sig_mm[:len(mm_raw)] = mm_raw`) because text and multimodal corpora had different sizes. Week 5's unified matrix eliminates that complexity.

**Q: Why is the embedding sanity check based on pairwise cosine similarity variance?**

`embedding_sanity_check()` samples 1,000 random pairs and computes their cosine similarities. If the variance of those similarities is < 0.001, it raises an error.

A variance near zero means all pairs have nearly identical cosine similarity — this happens when:
1. **Collapsed embeddings:** all vectors point in almost the same direction (anisotropy collapse). The embedding model failed to learn diverse representations. This is a known failure mode in contrastive learning when the negative sampling is too easy.
2. **All-zero vectors:** normalization of zero vectors returns zeros. If many texts produce zero-magnitude embeddings (e.g., empty string), all normalized versions are zero and all cosine similarities are zero → variance near zero.
3. **Constant vector:** a bug in preprocessing that emits the same vector for every input.

These failure modes produce retrieval that is effectively random (all fragrances look equally similar to any query). Catching this before shipping the artifact saves the entire downstream pipeline from producing garbage results silently.

The threshold of 0.001 is empirical — a healthy embedding distribution over diverse fragrance texts should have cosine similarity variance in the range 0.01–0.10.

---

### Retrieval Quality

**Q: How do you know the enriched retrieval is better than raw retrieval?**

Stage 15 of the pipeline runs a side-by-side comparison: the same 8 occasion queries are evaluated against both the raw embedding index and the enriched embedding index. Top-5 results are compared for each query.

Expected signal: for high-formality occasions (`black_tie`, `business_dinner`), the enriched index should return fragrances with high `formality` scores (0.75+) in the top-5; the raw index may return fragrances that merely have the words "dinner" or "evening" in their note lists. For `casual_day`, enriched should return fresh, low-formality fragrances; raw might return anything that mentions casual note families.

The comparison is logged in `results/week2_report.md` Section 3. If the enriched top-5 doesn't look meaningfully better on the formality extremes, something is wrong with either the enrichment quality or the `retrieval_text` construction.

**Q: Why does the multimodal model embed fragrance text as documents and not fragrance images?**

The fragrance database doesn't have images. Fragrances are not visually photographed products in this dataset — they're chemical compositions described in text. The multimodal branch uses Qwen3-VL as an asymmetric retrieval model: documents are text (fragrance `retrieval_text`), queries are multimodal (outfit image + occasion text). This is a valid use of VL embedding models — the "vision-language" refers to the query side, not the document side.

At inference time, the query `embed_multimodal_query(text=occasion, image_path=outfit_jpg)` produces a joint vector that captures: what the outfit looks like AND what the user said about the occasion. This joint vector is compared against fragrance text vectors. The model has learned to embed these different modalities into a shared space where semantic proximity is meaningful across the modality gap.

---

### Week 5 — New Mechanisms

**Q: Why add BM25 on top of dense retrieval instead of relying on embeddings alone?**

Dense retrieval (cosine similarity over 4096-d embeddings) captures semantic proximity — it finds fragrances whose meaning is similar to the query. It can miss exact lexical matches. A user who adds "oud" to their custom notes gets a query string containing "oud" — but whether the embedding model places this near the fragrance note vector for oud depends on training distribution and embedding geometry. It is not guaranteed.

BM25 (Best Match 25, Okapi variant) is a lexical term-frequency model. A `retrieval_text` containing "oud" scores high for any query containing "oud" — guaranteed, independent of embedding geometry.

The 10% BM25 blend (90% dense + 10% BM25) is a small correction signal, not a replacement. For most queries, the dense signal dominates. For queries with specific note terms ("oud", "iris", "vetiver", "petrichor"), BM25 boosts fragrances with exact lexical overlap that the dense model may have underranked. This is the classic hybrid search argument: dense retrieval for semantic understanding, sparse retrieval for lexical precision. The blend weight (0.10) is deliberately small — we don't want BM25 to override the semantic signal, just to correct obvious lexical misses.

**Q: Why does the hard filter run post-fusion instead of pre-retrieval?**

Pre-retrieval filtering (removing rows before scoring) has a critical failure mode: over-filtering. If the filter is too strict for a given query, you can reduce the candidate pool below 3 fragrances — top-k selection then fails or returns garbage.

Post-fusion filtering has a safety guard: if fewer than 3 rows survive the mask, the filter is bypassed entirely and the unfiltered top-k is returned. The hard filter is therefore a soft constraint — it improves results when conditions are clear (a gala query should not surface beach fragrances) and gracefully degrades when the filter would be too aggressive (a very niche context with few matching fragrances in the enriched corpus).

Pre-retrieval filtering also loses fusion score information. The filter makes binary decisions (include/exclude), but the fusion produces a continuous ranking. Running the filter after fusion means you preserve the ranking and only remove clear negatives — you do not blindly exclude fragrances that scored highly for good reasons.

**Q: Why MMR diversification instead of just taking top-3 by fusion score?**

Top-3 by fusion score has a clustering problem. In a 35,889-fragrance corpus, the highest-scoring candidates for a formal evening query tend to cluster in the same accord family. The top-20 for an oud-forward gala query might contain 12 oud-amber orientals that are all very similar to each other. Returning the top-3 of those 12 gives the user three near-duplicate recommendations in different bottles.

MMR (Maximal Marginal Relevance) balances relevance against diversity:
```
mmr_score(d_i) = λ × relevance(d_i, q) - (1-λ) × max_{d_j ∈ selected} similarity(d_i, d_j)
```
With λ=0.5, each pick maximizes 50% query relevance and 50% dissimilarity from already-selected candidates. The second pick is the most relevant fragrance that is also most different from the first. The third pick is most relevant and most different from both prior picks.

For a demo of three recommendations, showing three meaningfully different fragrance directions (heavy oriental, chypre, fresh floral) is more useful than showing the same top accord family three times. The tradeoff: MMR might pick the 4th-best relevance score if the 2nd and 3rd candidates are too similar. λ=0.5 is a conservative choice; λ→1 gives pure relevance, λ→0 gives pure diversity.

**Q: Why not use a single end-to-end multimodal LLM (e.g., Gemini Flash) to take the outfit image + context and directly output fragrance names?**

This is the obvious simple baseline. It was not chosen for four reasons:

1. **No grounding in the actual corpus.** A Gemini Flash call returns whatever fragrances the model has memorized from training data. It cannot search the 35,889-fragrance corpus — it has no access to it. The output is unconstrained by the catalog: the model might recommend discontinued fragrances, fragrances with wrong notes, or hallucinated entries.

2. **Not a retrieval system.** The task is corpus retrieval: find the best match from a specific catalog. LLM generation and vector retrieval solve different problems. The 4-channel pipeline is a retrieval system with a generative reranker at the end — not pure generation.

3. **No explainability.** The 4-channel pipeline produces: which signal drove the recommendation, what the fragrance's enrichment attributes are, and what its fusion score was. Pure LLM generation gives a fragrance name and a plausible-sounding rationale that may or may not reflect the actual fragrance's properties.

4. **No guarantees against the catalog.** With the pipeline, you can verify: does this fragrance exist in the corpus? Does its formality match the occasion? Is its embedding similar to the query? With LLM generation, you cannot verify any of these without a separate lookup step. For a demo where you want to show real, purchasable fragrances with live pricing, catalog grounding is essential.

The correct use of a large LLM in this system is as a reranker and rationale generator — given a constrained candidate set from corpus retrieval, it adds judgment on top, not generation from scratch.

**Q: What is the end-to-end latency budget for a single inference request?**

On an A100 80 GB with all models loaded:

| Step | Estimated latency |
|---|---|
| Query expansion | <1 ms |
| Text embedding (Qwen3-VL 4096-d, batch=1) | ~300–500 ms |
| Multimodal embedding (image + text, batch=1) | ~500–800 ms |
| CLIP image scoring (5 heads) | ~50–100 ms |
| Structured scoring (pure arithmetic) | <5 ms |
| Fusion + normalize (35,889 rows) | ~5–10 ms |
| Hard filter + BM25 blend | ~10 ms |
| Top-20 selection (`argpartition`) | <5 ms |
| Reranker (Qwen3-VL-Reranker-8B, 20 candidates) | ~1,000–2,000 ms |
| Response construction | <5 ms |
| **Total (reranker path)** | **~2–4 seconds** |
| **Total (MMR fallback path)** | **~1–1.5 seconds** |

Text + multimodal embedding dominate — both are single-batch forward passes through an 8B transformer. The reranker adds roughly as much again. Structured + fusion + BM25 are negligible. On T4 (multimodal skipped, no reranker): ~500–800 ms total.

**Q: Why expand the query string with rich descriptive phrases instead of embedding the raw user input?**

The user types: `Event = Gala`. Without expansion, the query embeds as the word "Gala." The embedding model places this near other text containing "Gala" — but fragrance `retrieval_text` is written in terms of accords, moods, and aesthetic descriptors, not event category names.

After expansion: `"black tie gala formal event elegant luxury oriental, chypre, oud, amber, incense, opulent, sillage"`. This string embeds directly into the fragrance chemistry-culture vocabulary space. Words like "oriental," "chypre," "sillage" appear in the fragrance `retrieval_text`. The query vector overlaps with these terms geometrically.

Without expansion, the query vector sits in the event-description region of the embedding space, which is semantically distant from the fragrance-notes region. The cosine similarities between a bare "Gala" vector and the fragrance corpus will be uniformly low — the text branch contributes near-zero discriminative signal.

The expansion lookup table is the only non-ML component in the pipeline. It encodes expert knowledge: a gala demands formal fragrance properties (opulent, sillage, black-tie) and specific accord families (oriental, chypre, oud). This is a handcrafted bridge between user intent vocabulary and fragrance vocabulary. The tradeoff: the expansion is fixed — if a user's context doesn't map cleanly to one of the lookup keys, the expansion falls back to the raw input. This is why `customNotes` (free text appended verbatim) exists.

---

### Overall System Design

**Q: Why four separate channels (text, multimodal, image, structured) instead of one end-to-end model trained jointly?**

A joint model would need labeled training data: `(outfit image, occasion, context) → correct ranked fragrance list`. We have 20 labeled benchmark cases. Training a joint model on 20 examples produces overfitting, not generalization.

The 4-channel architecture decomposes the problem into subproblems that can each be solved with existing pretrained models and zero task-specific training:
- Text: solved by retrieval-optimized embedding (Qwen3-VL-Embedding)
- Vision: solved by CLIP zero-shot classification
- Multimodal: solved by cross-modal retrieval (Qwen3-VL multimodal query)
- Structured: solved by arithmetic on LLM-generated enrichment attributes

Each of these is a well-posed retrieval or classification problem with strong pretrained solutions. Combining four high-quality signals with a weighted sum outperforms any single signal trained from scratch on 20 examples.

**Q: What is the weakest link in the pipeline, and how would you fix it in a production version?**

The weakest link is the **structured branch's attribute granularity**. It scores on `formality`, `fresh_warm`, `day_night`, and `likely_season` — five continuous dimensions. But fragrance character is high-dimensional: two fragrances can be identically formal, equally daytime, and equally spring-appropriate while being wildly different in character (one a light floral, one a heavy resinous chypre).

The fix: replace the 5-dimensional structured branch with a learned attribute embedding space trained on fragrance-specific dimensions (accord family, sillage preference, note family vector). With user interaction data (clicks, purchases), you could learn a fragrance preference embedding per user that functions as a personalized structured signal. The structured branch is the channel most amenable to improvement with data — the other three channels are already near-SOTA.

**Q: How does the system handle the cold-start problem when new fragrances are added to the catalog?**

New fragrances must go through the full offline pipeline: enrichment → embedding → manifest update → engine reload. This is a one-time offline cost per new fragrance, not a per-query cost.

In the current batch pipeline:
1. Enrichment: ~1-2 LLM calls per fragrance via `enrich_dataframe()` → adds `retrieval_text`, `formality`, `vibe_sentence`, etc.
2. Embedding: `Qwen3-VL-Embedding-8B` encodes `retrieval_text` → appended to `embeddings.npy`
3. Manifest: updated `n_rows` count
4. Engine reload: `VibeScoreEngine.from_artifacts()` reloads the full matrix

The cold-start problem is asymmetric: new fragrances cannot be retrieved until they are enriched and embedded. For a production system, this would require an incremental embedding pipeline (not a full re-run) and an in-memory hot-swap mechanism for the corpus matrix. The current design doesn't support this — it's batch-only.

**Q: How would you scale this system from 35,889 fragrances to 10 million products?**

At 10M products, three components break:

1. **Full matrix cosine similarity** — `(10M, 4096) @ (4096, 1)` is 41 billion floating-point operations per query. On A100, this takes ~400 ms per query — still acceptable. At 100 queries/second, the GPU is fully saturated on matrix multiplication alone. Solution: approximate nearest neighbor (ANN) search via FAISS with HNSW or IVF-PQ indices. HNSW retrieves top-100 from 10M vectors in ~1 ms with 97%+ recall.

2. **BM25 scoring** — `BM25Okapi.get_scores()` over 10M documents is linear in corpus size. Solution: Elasticsearch or OpenSearch with BM25 built-in, responding in ~10-20 ms for top-100 at 10M scale.

3. **Structured scoring** — pure arithmetic over a NumPy array scales to 100M rows trivially. This does not break.

4. **Engine startup time** — loading a `(10M, 4096)` float32 matrix is 160 GB. This does not fit in VRAM. Solution: FAISS index on disk + memory-mapped metadata, or split corpus across multiple GPUs.

The architecture's design with clean separation between channels makes each independently scalable.

**Q: How would you improve the system if you had 10,000 labeled `(outfit, occasion) → fragrance` interactions from real users?**

With real user interaction data, three major improvements become feasible:

1. **Fusion weight learning.** Replace the grid-searched heuristic weights with weights learned via gradient descent on the labeled set. Minimize negative log-likelihood of the correct fragrance in the ranked list. Expected improvement: the learned weights will reflect actual user preferences (which channel matters more for which query types) rather than engineering intuition.

2. **Reranker fine-tuning.** Fine-tune Qwen3-VL-Reranker-8B on contrastive pairs `(query, positive_fragrance, negative_fragrance)` drawn from the interaction data. A reranker trained on domain-specific preference data will significantly outperform the zero-shot baseline.

3. **Personalization.** With user identifiers, you can learn per-user preference vectors in the embedding space. This turns the structured branch from a query-only signal to a `query × user` signal — a user who consistently prefers woody fragrances gets an implicit thumb on the scale toward woody candidates.

The structured branch is likely the first to benefit: if users consistently prefer one accord family over another for the same occasion, the structured scoring function needs to model accord preference, not just formality/season attributes.

**Q: Why is this architecture appropriate for a portfolio project, and what would be different in a production system?**

**Portfolio strengths:**
- Demonstrates breadth: embedding, vision, multimodal, arithmetic, BM25, reranking, MMR — each is a real technique with principled motivation
- No shortcuts: the 35,889-row enrichment, unified corpus, and 5-head CLIP classifier are genuinely non-trivial engineering
- Defensible design choices: every component has an explicit reason (as documented in this section)
- Full pipeline: offline preprocessing, online inference, API, frontend, pricing scraper — not just a Jupyter notebook

**Production gaps (explicitly acknowledged):**
- Evaluation is 20 cases with LLM-generated labels — not human-labeled at scale
- No A/B testing infrastructure or online learning
- Single-node, no load balancing or fault tolerance
- Batch-only corpus update (no incremental)
- CLIP calibration not done (known gap)
- No content filtering for offensive outfit inputs

The portfolio framing is honest: this is a research prototype demonstrating the full stack of a multimodal retrieval system, not a deployed service. The production gaps are documented rather than hidden.

---

### Pipeline Robustness

**Q: What happens if the embedding model produces NaN or infinite values?**

The L2 normalization step (`normalize_rows()` in `similarity.py`) is the first line of defense: it divides by the L2 norm. If the norm is zero (which would happen with a zero vector), it divides by 1 instead (the `safe_norms = np.where(norms == 0, 1.0, norms)` guard). A NaN input would propagate through normalization as NaN, producing NaN cosine similarities, which would sort unpredictably via `argpartition`.

The embedding sanity check (`embedding_sanity_check()`) catches this before the artifact is stored: it verifies variance of pairwise similarities > 0.001. NaN values would produce NaN variances and fail the check.

In practice, NaN embeddings from transformer models indicate either (a) overflow in bfloat16 for very long inputs or (b) a corrupted model state. The offline pipeline would catch and flag these rows during the embedding stage.

**Q: Why does `_safe_probability()` clip to `[ε, 1.0]` instead of trusting the softmax output to be in range?**

Softmax over floating-point logits is mathematically guaranteed to produce values in (0, 1) exclusive. But:
- `float32` arithmetic can produce values like `7.5e-37` for strongly unfavored classes — numerically non-zero but so small that `log(p)` produces `-85` and a product of five such values underflows to zero
- `_EPSILON = 1e-8` clips these extreme values to a floor where `log(p) ≈ -18.4` — still a strong penalty, not an infinite one

More importantly: numpy's float32 doesn't guarantee IEEE-754 denormal handling consistently across hardware. On some CUDA devices, denormals (values < ~1e-38) are flushed to zero in hardware for performance. Clipping to `1e-8` before calling `log()` ensures no hardware-dependent zero-log crashes.

---

## References

- Qwen3-VL-Embedding-8B: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
- MMEB Leaderboard: https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard
- Gemini 1.5 Models: https://ai.google.dev/gemini-api/docs/models/gemini-1.5-pro
- Gemini structured outputs: https://ai.google.dev/gemini-api/docs/structured-output
- Gemini Batch API: https://ai.google.dev/gemini-api/docs/batch-api
- outlines (constrained generation): https://github.com/dottxt-ai/outlines
- Matryoshka Representation Learning: https://arxiv.org/abs/2205.13147
