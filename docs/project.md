# VibeScent — Full Project Reference

> Last updated: April 16, 2026. Incorporates all architecture decisions, ML design rationale, and implementation details through Week 4.

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

**Step 2 — You select 2,000 fragrances for deep processing (Tier B).**

You can't afford to run the expensive steps (LLM enrichment, multimodal embedding) on all 35,889 fragrances. You select the top 2,000 by `rating_count` with complete metadata. High rating count = well-documented, culturally known fragrances that the LLM enricher will have prior knowledge about. This is Tier B.

Why 2,000 and not more? LLM enrichment at 2,000 rows takes ~35-65 minutes. Multimodal embedding at 2,000 rows takes ~20-30 minutes on A100. The pipeline fits inside a single Colab/Kaggle session. 5,000 rows would risk session timeouts and cost more. 2,000 is the practical ceiling for overnight preprocessing.

Why keep the remaining 33,889 (Tier A) at all? Coverage. Niche fragrances ranked #4,000 by popularity may be the perfect match for an unusual request. Tier A makes them reachable via text search, even if at lower quality.

**Step 3 — You translate Tier B fragrances from chemistry to experience vocabulary (Enrichment).**

For each of the 2,000 fragrances, an LLM (Qwen3.5-27B-GPTQ-Int4 locally, or Gemini flash as fallback) reads the raw metadata and generates:

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

The output is constrained via `outlines` — a library that restricts token generation to valid JSON matching the exact schema. You cannot get a malformed or missing field. This is not optional: at 2,000 API calls, even a 1% failure rate means 20 broken fragrance embeddings.

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

**Step 5 — You embed every fragrance into a 1024-dimensional vector (Text Branch).**

Qwen3-Embedding-8B reads each `retrieval_text` string and outputs a 1024-dimensional vector. All 35,889 fragrances are embedded — Tier A with raw text, Tier B with enriched text. The vectors are L2-normalized and saved as a matrix to disk: `fragrance_raw_full/embeddings.npy` (shape 35,889 × 1024).

Why 1024 dimensions? Enough to encode rich semantic structure. Higher dimensions (the model supports up to 4096) cost 4× more memory at query time with marginal retrieval improvement. The model is Matryoshka-trained — the first 1024 dimensions are specifically optimized to be the highest-information prefix.

**Step 6 — You embed Tier B fragrances jointly with a vision-language model (Multimodal Branch).**

Qwen3-VL-Embedding-8B reads each enriched `retrieval_text` string and outputs a 1024-d vector that lives in a space where visual and textual content are comparable. These 2,000 vectors are saved separately: `multimodal_2k/doc_embeddings.npy` (shape 2,000 × 1024).

Why a separate embedding run? The multimodal model's embedding space is calibrated for cross-modal retrieval — the query side will be an outfit image + text, so the document side needs to be embedded by the same model to be comparable. Mixing Qwen3-Embedding text vectors with Qwen3-VL text vectors would create vectors from different spaces — cosine similarity between them would be meaningless.

**Step 7 — Neil's CNN classifies outfit images into attribute buckets (Image Branch).**

This doesn't run offline on fragrances — it runs online on each user's outfit image. The CNN (ResNet-50 + CLIP ViT-L/14) takes a 224×224 image and outputs three probability distributions: how formal is this outfit (3 classes), what season is it appropriate for (4 classes), is it day or night wear (2 classes)?

Why pre-train a CNN specifically for this instead of using CLIP directly? CLIP's embedding output is opaque — you can't inspect what it's predicting about the outfit. The CNN's classification heads give you explicit, human-interpretable predictions: `P(formal) = 0.83`. You can debug this. You can say "the CNN thinks this is a fall, formal, night outfit" and trace exactly how that prediction scores against each fragrance.

The fragrance database already has pre-computed formality, season, and time-of-day values from enrichment (Step 3). The CNN bridges the gap: it maps the outfit image into the same attribute space that the fragrances already live in.

---

### What Happens at Request Time — Online Inference

A user uploads an outfit photo and selects: Event = "Gala", Time = "Evening", Mood = "Mysterious".

**Step 1 — Build the occasion text.**

```python
occ_text = "Gala, Evening, Mysterious"
```

This string is the text representation of the user's intent. It's used by both the text branch and the multimodal branch as the query-side input.

**Step 2 — Text Branch produces 35,889 scores.**

`q_text = Qwen3-Embedding-8B("Gala, Evening, Mysterious")` → 1024-d vector.

`sig_text = TEXT_EMBEDDINGS @ q_text.T` → 35,889 scalar scores.

This is a matrix multiply: the pre-computed 35,889 × 1024 fragrance matrix dotted against the 1024-d query vector. Each scalar is the cosine similarity between that fragrance's enriched text embedding and the user's occasion query.

Time: ~50ms on CPU, ~5ms on GPU. This is fast because it's a single BLAS matrix multiply.

What it finds: fragrances whose retrieval_text is semantically close to "Gala, Evening, Mysterious." The enriched fragrances will score well if their `vibe_sentence` or `mood_tags` or `likely_occasion` align with those words. Raw Tier A fragrances score based on whatever note/accord language overlaps with those terms.

**Step 3 — Multimodal Branch produces 2,000 scores.**

`q_mm = Qwen3-VL-Embedding-8B(text="Gala, Evening, Mysterious", image=outfit_photo)` → 1024-d vector.

The model reads both the image and the text simultaneously and produces a single joint embedding. This embedding encodes: what does this outfit look like AND what is the user saying about the occasion?

`sig_mm = MM_EMBEDDINGS @ q_mm.T` → 2,000 scores, zero-padded to the full corpus size.

Why this is different from the text branch: the joint embedding shifts based on the outfit's visual character. A dark, structured blazer in the outfit photo will push the query vector toward "formal, evening" associations even if the user typed "casual" — the image overrides the text to some degree. A bright yellow sundress would pull toward "fresh, summer" associations. The text branch can't see the image; the multimodal branch can't ignore it.

**Step 4 — Image CNN Branch produces 2,000 scores.**

The outfit photo is preprocessed: resize to 224×224, normalize with CLIP mean/std.

Forward pass through `CNNCLIPHybrid`:
```
ResNet-50(outfit) → 2048-d features
CLIP ViT-L/14(outfit) → 768-d features
cat([2048, 768]) → Linear → 256-d shared trunk
→ formal_head → softmax → [0.05, 0.12, 0.83]
→ season_head → softmax → [0.07, 0.04, 0.71, 0.18]
→ time_head   → softmax → [0.15, 0.85]
```

The CNN predicts: this outfit is 83% formal, 71% fall-appropriate, 85% evening.

For each of the 2,000 Tier B fragrances, the engine looks up the fragrance's enrichment attributes and computes:

```
score = P_formal[fragrance_formal_class] × P_season[fragrance_season_class] × P_time[fragrance_time_class]
```

A fragrance with `formality=0.88` (formal class), `likely_season=fall`, `day_night=0.82` (night class) scores:
```
0.83 × 0.71 × 0.85 = 0.50
```

A summer-fresh casual daytime fragrance scores:
```
0.05 × 0.04 × 0.15 = 0.0003
```

The CNN attribute matching creates a sharp distinction between fragrances that fit the outfit's visual character and those that don't.

**Step 5 — Structured Branch produces 2,000 scores.**

Currently (known gap): `(formality + fresh_warm) / 2` per fragrance. A fixed score that doesn't use the user's input. It biases toward formal, warm fragrances regardless of query.

What it should do: map "Gala" → `formality_target=0.9`, "Evening" → `day_night_target=0.85`, "Mysterious" → `fresh_warm_target=0.65` (warm end), then score fragrances by distance from those targets.

**Step 6 — Normalize and fuse.**

Each of the four score arrays is independently min-max normalized to [0, 1]:
```
sig_text_norm = (sig_text - min) / (max - min)    # shape (35889,)
sig_mm_norm   = (sig_mm   - min) / (max - min)    # shape (35889,), zeros where no MM embedding
sig_img_norm  = (sig_img  - min) / (max - min)    # shape (35889,), zeros outside Tier B
sig_s_norm    = (sig_s    - min) / (max - min)    # shape (35889,), zeros outside Tier B
```

Why normalize before fusing? Each branch produces scores on a different scale. Text cosine similarities cluster between 0.3 and 0.7. Image NLL-derived scores might span 0.001 to 0.8. Without normalization, a branch with naturally higher scores would dominate regardless of its assigned weight. Normalization makes the weight the actual control dial.

Fusion:
```
fused = 0.30 × sig_text_norm + 0.25 × sig_mm_norm + 0.30 × sig_img_norm + 0.15 × sig_s_norm
```

Every fragrance now has a single score between 0 and 1 that combines all four signals proportionally to their weights.

**Step 7 — Select top candidates.**

`top_50_indices = argpartition(-fused, 50)[:50]` — the 50 fragrances with the highest fused scores. Done in O(N) time (introselect), not O(N log N) (full sort). Then sort those 50 by score.

**Step 8 — Rerank (if shipped).**

The top 10 of those 50 go to `GeminiReranker`. The reranker receives:
- The outfit image
- The occasion text ("Gala, Evening, Mysterious")
- Each candidate's `name`, `brand`, `retrieval_text`, `baseline_score`, and full metadata

Gemini 3.1 Pro Preview scores each candidate holistically: "Does this fragrance actually fit this outfit and occasion, considering everything I know about fragrance culture, the outfit's visual character, and the user's stated intent?" It returns per-candidate scores and rationale text.

The reranker catches failures the fusion can't: a fragrance that scores well on text and image signal but is tonally wrong for the occasion (e.g., a fun, playful fragrance with a "fruity" vibe that has high formal attributes but is culturally wrong for a gala).

If the reranker isn't shipped, the top 5 of the 50 go directly to output.

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
- Multimodal alone: limited to 2K fragrances, black-box
- Image CNN alone: 9 values total, no semantic nuance
- Structured alone (ideal): no neural understanding of similarity, only attribute arithmetic

Together, they triangulate: if all four agree a fragrance is the best match, confidence is high. If they disagree, the weighted sum finds the centroid, which is still better than any single signal alone.

**Why not train a single end-to-end model?**

An end-to-end model (outfit image + occasion text → fragrance embedding, trained to minimize distance to correct fragrances) would be ideal. It would learn the optimal representation jointly. But it requires:
- Thousands of labeled `(outfit, occasion, correct_fragrance)` training pairs
- A training loop (GPU time, hyperparameter tuning, evaluation)
- A dataset that doesn't exist for this domain

The retrieval + fusion approach achieves the same goal without labeled training data: it leverages models pre-trained on related tasks (general text retrieval, cross-modal retrieval, image classification) and combines their outputs. This is transfer learning at the architecture level rather than the parameter level.

**Why rerank instead of using a better retrieval model?**

Retrieval is fundamentally a recall problem: find every fragrance that might be good. Reranking is a precision problem: from those candidates, find the best one. Optimizing for both simultaneously in a single model is hard. Separating retrieval (fast, recall-optimized, runs over 35K fragrances) from reranking (slower, precision-optimized, runs over 10 candidates) lets each component specialize.

The reranker (Gemini 3.1 Pro) has far more compute budget per candidate than the retrieval system. It can reason holistically about the outfit image and the fragrance profile in a way that matrix multiply cannot. But it can't run on 35,889 candidates — the latency would be minutes. Running it on 10 candidates is ~2-5 seconds.

This is the standard two-stage retrieval architecture used in production search systems (dense retrieval → cross-encoder reranking). We've adapted it for multimodal fragrance retrieval.

---

## Corpus Structure — The Tier System

Before explaining retrieval, you need to understand how the fragrance data is organized. There are three tiers, and understanding which tier each branch operates on is essential to understanding the system's coverage/quality tradeoff.

### Tier A — Full Corpus (35,889 fragrances)

The raw Fragrantica dataset. Every fragrance in the database is in Tier A. It contains: brand, name, top/middle/base notes, main accords, gender, concentration, rating count.

**What it does NOT have:** any LLM-enriched fields. No formality score, no vibe sentence, no mood tags, no season inference. Just the raw chemistry vocabulary that Fragrantica stores.

**Which branches can reach Tier A:** only the text branch. All other branches (multimodal, image, structured) operate exclusively on Tier B.

**Why Tier A exists:** coverage. Recommending only the top 2,000 fragrances by rating count would systematically exclude niche houses and cult fragrances that might be the perfect match for an unusual outfit. Tier A ensures that any fragrance in the database is reachable, even if the retrieval signal quality is lower.

### Tier B — Enriched Working Set (2,000 fragrances)

The top 2,000 rows from Tier A selected by `rating_count` with complete metadata (all four note columns non-null). These are the well-documented, well-known fragrances that have the richest metadata for enrichment.

Selection logic (`select_tier_b()` in `week2_pipeline.py`):
1. **Strict filter**: require `top_notes`, `middle_notes`, `base_notes`, `main_accords` all non-null. Take top 2,000 by `rating_count`.
2. **Fallback filter**: if strict yields <2,000, relax to require only `top_notes` and `main_accords`.
3. **Hard minimum**: abort if relaxed filter yields <500 rows.

Tier B fragrances go through the full enrichment pipeline (LLM generates `EnrichmentSchemaV2` fields), get re-embedded with enriched text, and get separately embedded with the multimodal model. All four scoring branches are active for Tier B.

### Tier C — Development Sample (500 fragrances)

A 500-row sample used exclusively for fast enrichment smoke testing and retrieval quality comparison during pipeline development. Not used in production inference. Lets you validate that enrichment works before committing to the full 2,000-row run.

### The Coverage/Quality Tradeoff

This tier system creates an explicit tradeoff: fragrance rank 2,001 by popularity is reachable only via the text branch on its raw retrieval_text. It competes for a top-K slot using only 30% of the fusion signal (the text weight), with the multimodal (25%), image (30%), and structured (15%) signals zeroed out. A top-100 Tier B fragrance competes with all four signals active.

For a demo this is acceptable. For production, you'd want to enrich and multimodal-embed the full corpus — which would cost significantly more in GPU time and LLM API calls.

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

For every fragrance in Tier B, an LLM (Qwen3.5-27B-GPTQ-Int4 locally, or Gemini flash via API as fallback) receives the raw metadata as a prompt and acts as a fragrance expert:

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

### Constrained Decoding — Why `outlines` Is Critical

The enrichment pipeline uses `outlines` for structured output generation. This is not just JSON prompting. `outlines` intercepts the token logit distribution at each decoding step and zeroes out any token that would produce output incompatible with `EnrichmentSchemaV2`'s JSON schema. The beam search itself is constrained to the schema's grammar.

Concretely:
- `likely_season` can only be one of `["spring", "summer", "fall", "winter", "all-season"]` — enforced at the token level, not by post-processing
- `formality` must be a float between 0.0 and 1.0 — the tokenizer cannot emit a value outside this range
- `character_tags` must have 3–5 items — the generation stops exactly at the constraint boundary

This matters because enrichment runs on 2,000 rows. A malformed output on row 847 that slips through would silently degrade that fragrance's embedding — you'd never know unless you explicitly checked. Constrained decoding makes malformed output structurally impossible.

Fallback: when running on T4 GPU (insufficient VRAM for Qwen3.5-27B), the pipeline falls back to `GeminiEnrichmentClient`, which uses Gemini's native `response_schema` parameter to achieve the same constraint. Both paths produce identical `EnrichmentSchemaV2` objects.

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

**Model:** Qwen3-Embedding-8B (Week 2 pivot from the original `voyage-3-large` plan — same 1024-d, better MTEB coverage at lower cost on GPU)

**How it works:**

1. The user's occasion context string is embedded: `_embed_text("evening gala, mood: mysterious")` → vector `q ∈ ℝ¹⁰²⁴`, L2-normalized
2. The pre-computed fragrance corpus matrix `TEXT_EMBEDDINGS ∈ ℝ^(35889 × 1024)` is loaded from `fragrance_raw_full/embeddings.npy`
3. Cosine similarity is computed via matrix multiply (valid because both sides are L2-normalized): `sig_text = TEXT_EMBEDDINGS @ q.T` → shape `(35889,)`
4. Each scalar in `sig_text` is the cosine similarity between that fragrance's text embedding and the query

**Matryoshka truncation:** Qwen3-Embedding-8B is trained with Matryoshka Representation Learning — nested dimensionality. The model produces 4096-d vectors that are trained so that any prefix (e.g. first 1024 dimensions) is itself a valid, lower-quality embedding. The pipeline truncates to 1024-d: `raw[:, :1024]` before L2-normalizing. This allows cross-modal compatibility with the multimodal model's vectors.

**L2 normalization:** every embedding is normalized to unit length before storage. This makes cosine similarity equivalent to dot product (`cos(θ) = a·b` when `|a|=|b|=1`), which is a single matrix multiply — O(N × D) for N fragrances and D dimensions.

**Why this branch is irreplaceable:**
- It is the ONLY branch that covers all 35,889 fragrances. Every other branch covers 2,000.
- It captures semantic overlap between occasion descriptions and fragrance descriptions in the same embedding space.
- It can find fragrances whose notes/accords semantically associate with the occasion even without explicit enrichment attributes.

**Why the text branch alone is insufficient:**
- It embeds only what the user types. It cannot see the outfit image.
- Raw retrieval_text is anchored in chemistry vocabulary. A user querying "mysterious evening" retrieves fragrances whose text descriptions contain semantically similar language — which helps for enriched fragrances but is weaker for unenriched Tier A entries.
- Two fragrances with identical note profiles but different vibes are indistinguishable by text similarity alone.

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

1. **Document embedding (offline, pre-computed):** For each fragrance in Tier B, the enriched `retrieval_text` is embedded through Qwen3-VL as a text-only document: `MM_EMBEDDINGS ∈ ℝ^(2000 × 1024)`, saved to `multimodal_2k/doc_embeddings.npy`. The VL model's native output is truncated to 1024 dimensions: `emb[:, :1024]`.

2. **Query embedding (online, at inference):** At query time, the outfit image + occasion text are embedded together: `q_mm = MM_EMBEDDER.embed_multimodal_query(text=occasion_text, image_path=tmp_img)`. The joint query vector encodes both the visual character of the outfit and the semantic intent of the occasion description.

3. **Similarity:** `sig_mm = MM_EMBEDDINGS @ q_mm.T` → shape `(2000,)`. Padded to the full corpus length with zeros for fragrances not in Tier B.

**The key insight — what the image adds:** A user typing "evening event" and a user wearing a linen shirt produce the same text query. But the multimodal query vector for the linen shirt shifts away from heavy oriental fragrances toward fresh, lighter profiles — even though neither the user nor the query text mentioned that. The image carries visual information that changes the query embedding. The degree to which the image shifts the query is the unique contribution of this branch.

**Why the multimodal branch is non-redundant with the image CNN branch:**
- Multimodal uses open-vocabulary visual-semantic understanding — it can capture subtle signals like fabric weight, color saturation, silhouette formality in a continuous 1024-d embedding space.
- The image CNN classifies the outfit into 9 discrete values (3 formality classes, 4 season classes, 2 time classes). These two signals are architecturally independent and may disagree.
- Multimodal similarity is holistic and fuzzy. CNN-to-attribute matching is explicit and interpretable.
- Both can be correct or wrong independently of each other. When they agree, confidence increases. When they disagree, the fusion still works because neither dominates.

**Operational constraint:** Qwen3-VL-Embedding-8B requires ~16 GB VRAM (float16). It can only run on L4 or A100. On T4 (15GB VRAM), the multimodal embedder is skipped at inference time and `sig_mm` is zeroed out — the engine degrades gracefully to a 3-signal fusion.

### Signal 3 — Image CNN Branch (weight: 0.30)

**The question it answers:** What does the outfit's visual features predict about formality, season, and time-of-day — and which fragrances' enrichment attributes match those predictions?

**Model:** Neil's CNN-CLIP Hybrid (`CNNCLIPHybrid`) — ResNet-50 backbone fused with CLIP ViT-L/14 features, with three multi-task classification heads.

#### Architecture — Two Backbone Fusion

```
Outfit image (224 × 224, CLIP-normalized)
        ↓                        ↓
ResNet-50 backbone        CLIP ViT-L/14 vision model
(spatial, local)          (global, semantic)
2048-d features           768-d features (via projection)
        ↓                        ↓
        └──── cat ───────────────┘
               2816-d fused vector
                      ↓
              Linear(2816 → 512) → ReLU → Dropout(0.3)
              Linear(512 → 256) — shared trunk
                      ↓
        ┌─────────────┼────────────┐
        ↓             ↓            ↓
formal_head    season_head    time_head
Linear(256,3)  Linear(256,4)  Linear(256,2)
```

**Why two backbones instead of one:**
- ResNet-50 is convolutional — it processes the image through spatial local kernels. It's good at texture (fabric weave, material shimmer), garment structure (collar shape, lapels), and fine-grained visual detail.
- CLIP ViT-L/14 is a vision transformer — it attends globally across the image. It's good at high-level semantic context: "this is a tuxedo," "this is a summer dress," "this is streetwear." CLIP was trained on hundreds of millions of image-text pairs, giving it rich visual-semantic associations.
- Concatenating their features gives the classifier both local texture signals AND global semantic signals. Neither alone is sufficient: ResNet-50 without CLIP context might classify a black turtleneck as "casual" (it's dark and minimal). CLIP without ResNet's texture might miss that a garment is velvet vs. matte, which changes the formality inference.

**CLIP normalization constants:** the preprocessing pipeline applies CLIP's exact training normalization:
```python
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
```
These are baked into CLIP's checkpoint weights. Using ImageNet normalization instead would silently degrade the ViT-L/14 branch because its weights were learned against CLIP's specific pixel distribution. Images are bicubic-resized to 224×224 (ViT-L/14's training resolution — not configurable).

#### Three Classification Heads

The model outputs three independent softmax probability distributions:

```
formal:  P ∈ ℝ³   → [P(casual), P(semi-formal), P(formal)]
season:  P ∈ ℝ⁴   → [P(spring), P(summer), P(fall), P(winter)]
time:    P ∈ ℝ²   → [P(day), P(night)]
```

Softmax uses the log-sum-exp trick for numerical stability: `shifted = logits - max(logits)` before exponentiation, preventing overflow for large logits.

**Why multi-task heads on a shared trunk:** the three classification tasks (formality, season, time) share visual features. A tuxedo is formal (formal head), typically fall/winter (season head), and evening (time head). Training a shared 256-d trunk to predict all three simultaneously forces the model to learn a richer intermediate representation than three independent classifiers would. The formality-predicting neurons and the season-predicting neurons share the same trunk and regularize each other.

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
      + log P_time[time_target])

score = exp(-NLL) = P_formal[target] × P_season[target] × P_time[target]
```

The score is the **joint probability** that the CNN assigns the outfit to the fragrance's exact attribute bucket across all three dimensions. Maximizing this score = finding fragrances whose attributes match what the CNN predicts about the outfit.

**Why NLL and not L2 distance over probabilities:**
- NLL is the proper probabilistic loss. It is calibrated to the softmax output: a model that outputs `[0.95, 0.03, 0.02]` for the correct class gets low NLL; one that outputs `[0.35, 0.33, 0.32]` gets high NLL. L2 distance would not distinguish these as sharply.
- Using NLL also means the score degrades gracefully when the model is uncertain — a uniform distribution over formal classes gives lower score than a peaked one, correctly expressing lower confidence.

**Known calibration gap:** neural network classifiers are often overconfident — they output `P=0.95` when empirical accuracy is 80%. If the CNN is overconfident, scores will cluster near 0 and 1, reducing discrimination between fragrances. Temperature scaling (`logits / T` before softmax) is the standard fix. It is not currently applied. This is a known limitation.

**Why this branch is non-redundant with multimodal:**
- Multimodal produces a continuous 1024-d embedding — interpretability is zero. You can't inspect what visual feature drove the score.
- The CNN branch is fully interpretable: you can read off `P_formal=[0.05, 0.12, 0.83]` and understand exactly what the model predicts.
- The CNN trained specifically on outfit classification may be more precise on garment-specific visual features (cut, collar, lapel, hem) than a general vision-language model whose training is diluted across all visual domains.
- Most importantly: the CNN connects outfit visual features to fragrance enrichment attributes via a probabilistic bridge. The multimodal branch connects outfit features to fragrance text descriptions via a semantic embedding bridge. These are different paths through different intermediate representations.

**Why this branch is non-redundant with text:**
- Text never sees the image. A user can type "casual event" but wear a tuxedo. Text believes the words. The CNN believes the pixels. These are independent sources of evidence that should both be weighted.

#### Score Normalization

```python
likelihoods = [P_formal × P_season × P_time for each fragrance]
return min_max_normalize(np.array(likelihoods))
```

Min-max normalization across the candidate pool means the image branch contributes a **relative ranking signal**, not an absolute probability. The fragrance with the highest joint probability gets score 1.0; the worst gets 0.0. This normalization is applied independently per branch in the fusion stage.

### Signal 4 — Structured Branch (weight: 0.15)

**The question it answers (in the ideal implementation):** How well do this fragrance's pre-computed enrichment attributes align with the user's explicit occasion context?

**Current implementation (`_struct_score()` in the Week 4 notebook):**

```python
def _struct_score(df):
    cols = [c for c in ('formality', 'fresh_warm') if c in df.columns]
    if not cols:
        return np.ones(len(df), dtype=np.float32) * 0.5
    v = np.zeros(len(df), dtype=np.float32)
    for c in cols:
        v += df[c].fillna(0.5).values.astype(np.float32)
    return (v / len(cols)).clip(0, 1)
```

This computes `(formality + fresh_warm) / 2` for each fragrance — a fixed score that does not depend on the user's query at all. A fragrance with `formality=0.9, fresh_warm=0.8` always scores 0.85 regardless of whether the user is going to a gala or the beach.

**What the ideal implementation should do:**

The `ContextInput` schema already captures user intent:
```python
class ContextInput(BaseModel):
    eventType:   str | None   # "Gala", "Date Night", "Casual", etc.
    timeOfDay:   str | None   # "Morning", "Evening", etc.
    mood:        str | None   # "Bold", "Mysterious", etc.
    customNotes: str | None
```

The correct structured branch maps these to enrichment attribute targets and scores each fragrance by distance from those targets:

```python
# Map user context to attribute targets
time_target   = 0.85 if ctx.timeOfDay in ("Evening", "Night") else 0.20
formal_target = 0.90 if ctx.eventType == "Gala" else ...
warm_target   = 0.70 if ctx.mood == "Bold" else ...

# Score each fragrance
score = 1 - (|fragrance.day_night   - time_target|
           + |fragrance.formality    - formal_target|
           + |fragrance.fresh_warm   - warm_target|) / 3
```

**Why the structured branch is non-redundant even in its current broken form:**

The three neural branches (text, multimodal, image) all involve learned inference — embedding geometry, trained weights, probabilistic classifiers. All three can be wrong in the same direction if training data is biased. The structured branch uses **ground truth attributes** computed by an LLM fragrance expert reasoning explicitly about each fragrance's cultural role, note profile, and occasion appropriateness.

`formality=0.88` for Baccarat Rouge 540 is not inferred from pixels or embedding distance. It was produced by a 27B-parameter model with fragrance domain knowledge, instructed specifically to rate formality on a 0–1 scale. This is a different epistemic source — reasoned from explicit domain knowledge rather than learned from distributional statistics.

In the fully implemented version, the structured branch is the only signal that directly compares user intent to fragrance attributes without any neural intermediary — pure attribute arithmetic.

**The fix required:** wire `ctx.eventType`, `ctx.timeOfDay`, and `ctx.mood` into `_struct_score()` as described above. The enrichment attributes (`day_night`, `formality`, `fresh_warm`, `likely_season`) are all already computed and stored in `FRAGRANCE_DF`. This is a one-day engineering task.

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

After fusion, `np.argpartition(-fused, 50)[:50]` selects the top 50 candidates efficiently (O(N) average vs O(N log N) for full sort). These 50 are then sorted by score to produce the ordered shortlist for reranking.

---

## Stage 3 — Reranking

The reranker is a research component — it does not replace the fusion baseline unless it wins on the 20-case benchmark.

**Model:** Gemini 3.1 Pro Preview (multimodal input, constrained JSON output)

**Input to reranker:** outfit image bytes + occasion text + top-10 candidates from fusion, each including `fragrance_id`, `name`, `brand`, `retrieval_text`, `baseline_score`, and full metadata dict.

**Output schema:**
```python
class RerankResult(BaseModel):
    fragrance_id:   str
    overall_score:  float   # 0.0–1.0
    formality_score: float
    season_score:   float
    freshness_score: float
    explanation:    str     # 1–2 sentence rationale
```

**Why a reranker on top of fusion:** the fusion is a weighted average of independent signals. It can't reason about whether a combination of signals makes sense together, or catch cases where one branch is confidently wrong. A multimodal LLM with visual understanding and fragrance knowledge can look at the outfit image and the top candidate and say "this doesn't make sense — this is a beach outfit and BR540 is a black tie fragrance." The fusion can't make that holistic judgment.

**Why it's research-only (for now):** reranking adds ~2–5s latency per request (Gemini API call). If the reranker doesn't improve benchmark scores, the latency cost is not justified. Week 3 runs the comparison. The decision is stored in `week3/reranker_decision.json`: `{ship_reranker: true/false, reason: "..."}`.

**Evaluation design — avoiding self-confirmation bias:** the label generator is `gemini-3.1-pro-preview` and the evaluation judge is `gemini-2.5-pro`. Using the same model as both label writer and judge would cause the judge to prefer outputs that match its own generation patterns, not outputs that are actually better.

---

## Why Each Branch is Non-Redundant — Summary

The four branches answer four different questions from four different information sources:

| Branch | Primary question | Sees image? | Sees text? | Covers full corpus? | Neural? |
|---|---|---|---|---|---|
| text | Does the fragrance description semantically match the occasion? | No | Yes | Yes (35,889) | Yes |
| multimodal | Does the joint visual-semantic character of the outfit match the fragrance? | Yes | Yes | No (2,000) | Yes |
| image CNN | Do the outfit's inferred attributes (formality/season/time) match the fragrance's attributes? | Yes | No | No (2,000) | Yes |
| structured | Do the fragrance's pre-computed enrichment attributes match the user's explicit context? | No | No | No (2,000) | No |

No two branches share the same profile. Removing any one creates a systematic blind spot:
- **Remove text:** 33,889 fragrances become unreachable
- **Remove multimodal:** the image never influences which fragrances are retrieved — only the CNN's coarse 9-value classification can carry visual information, losing all nuanced visual-semantic association
- **Remove image CNN:** the only independently-trained visual classifier is gone. The system has no interpretable attribute-matching path from outfit to fragrance
- **Remove structured:** the only non-neural, ground-truth attribute signal disappears. All four signals would then be neural inferences that can be biased in the same direction by the same distributional assumptions in their training data

---

## 21-Stage Offline Pipeline (Week 2 Notebook)

The preprocessing pipeline that generates all artifacts runs as 21 notebook stages:

| Stage | Purpose | Output |
|---|---|---|
| 1 | Setup: GPU tier detection, disk check, deps | GPU tier string (A100/L4/T4) |
| 2–4 | Load raw CSV, tier selection | Tier A and Tier B DataFrames |
| 5 | Embed Tier A raw text (Qwen3-Embedding-8B) | `fragrance_raw_full/embeddings.npy` (35889 × 1024) |
| 6 | Load enrichment LLM (Qwen3.5-27B or Gemini fallback) | LLM client in memory |
| 7 | Smoke test enrichment (single row) | Pass/fail gate |
| 8 | Enrich Tier C (500-row sample) | `vibescent_enriched_500.csv` |
| 9 | Retrieval comparison (Tier A raw baseline) | Evaluation report |
| 10 | Enrich Tier B (2,000 rows) | `vibescent_enriched_2k.csv` |
| 11 | Rebuild `retrieval_text` for enriched tiers | Updated DataFrames |
| 12 | Generate `display_text` (optional) | Skipped in production runs |
| 13–14 | Embed enriched tiers (Qwen3-Embedding-8B) | `fragrance_enriched_500/`, `fragrance_enriched_2k/` |
| 15 | RAW vs ENRICHED retrieval comparison | Quality validation report |
| 16 | Multimodal query probes (text-only queries) | Baseline similarity matrices |
| 17 | Embed Tier B documents multimodally (Qwen3-VL) | `multimodal_2k/doc_embeddings.npy` (2000 × 1024) |
| 18 | Multimodal query probes (text + image queries) | Quality validation report |
| 19 | Embedding sanity checks (variance test) | Pass/fail gate |
| 20 | Report writer | `results/week2_report.md` sections |
| 21 | Triple artifact sink (parallel copy) | Artifacts on Drive + local + backup |

**Stage gate mechanism:** each expensive stage (embedding, enrichment) writes a `manifest.json` on completion. `stage_complete(stage_id, artifacts_dir, pipeline_version)` checks for this manifest and skips the stage if it exists. Re-running the notebook after a crash resumes from the last completed stage without re-embedding.

**GPU tier routing:** GPU VRAM detection at Stage 1 determines which stages run:
- A100 (≥35 GB): all 21 stages run at full capacity
- L4 (≥20 GB): Qwen3-VL embedding runs, but at smaller batch sizes
- T4 (<20 GB): multimodal embedding is skipped (Stage 17 no-ops); enrichment falls back to Gemini API; only Tier C (500 rows) is enriched

**Checkpoint-based recovery:** `embed_corpus()` writes partial embedding matrices every 100 batches. On crash, `embed_corpus_resume()` globs for checkpoint files, concatenates them in order, and returns the partial result with the next batch index. This prevents losing hours of GPU work to Colab disconnects.

---

## Inference Engine — `VibeScentEngine`

The Week 4 notebook builds a `VibeScentEngine` class that implements the `RecommendationEngine` protocol. It is injected into the FastAPI app and handles the full inference path.

### Request Cache

The engine pre-computes 5 locked responses for rehearsed demo cases before the presentation:

```python
LOCKED_CASES = [
    "wedding|evening|romantic",
    "business meeting|morning|confident",
    "beach vacation|afternoon|relaxed",
    "date night|evening|sensual",
    "casual day out|afternoon|happy",
]
```

Cache keys are `f"{eventType}|{timeOfDay}|{mood}"`. A cache hit bypasses all model inference and returns the pre-computed response. This makes the demo deterministic for rehearsed cases even if the GPU runs out of memory or the tunnel drops.

### Inference Path

1. Check cache → return immediately if hit
2. Build `occ_text` from context: `", ".join([eventType, timeOfDay, mood, customNotes] filtered for None)`
3. Compute `sig_text` via text embedder + matrix multiply (always available)
4. Compute `sig_mm` via multimodal embedder + temp file for image bytes (skipped if no embedder or T4)
5. Compute `sig_img` via Neil CNN: decode base64 → `(1, 3, 224, 224)` tensor → forward pass → `score_candidate_pool()` (skipped if no checkpoint)
6. Compute `sig_s` via `_struct_score()` (always available, currently ignores query)
7. Min-max normalize all four signals independently
8. Weighted sum → `fused ∈ ℝ^(corpus_size)`
9. `argpartition(-fused, 50)[:50]` → top-50 candidates
10. If reranker shipped: pass top-10 to `GeminiReranker.rerank()` → use reranker scores
11. Else: take fused top-5 directly
12. Map indices back to DataFrame rows → build `FragranceRecommendation` objects → return `RecommendResponse`

---

## Model Table

| Role | Model | Provider | Key Metric | Notes |
|---|---|---|---|---|
| Text embedding | Qwen3-Embedding-8B | Local GPU / HuggingFace | MTEB 68.x | 1024-d after Matryoshka truncation |
| Multimodal embedding | Qwen3-VL-Embedding-8B | Local GPU / HuggingFace | MMEB-V2 77.8 (#1) | ~16GB VRAM, 1024-d |
| Enrichment LLM | Qwen3.5-27B-GPTQ-Int4 | Local GPU (vLLM + outlines) | — | Falls back to Gemini flash on T4 |
| Enrichment fallback | gemini-3-flash-preview | Google API | — | Structured output via response_schema |
| Image CNN | CNNCLIPHybrid (Neil) | Local GPU | — | ResNet-50 + CLIP ViT-L/14, 3 heads |
| Reranker | gemini-3.1-pro-preview | Google API | — | Research only; ships if benchmark wins |
| Evaluation judge | gemini-2.5-pro | Google API | — | Different from label generator |
| Label generator | gemini-3.1-pro-preview | Google API | — | 3-run majority vote per benchmark case |

---

## Shared Schema — `EnrichmentSchemaV2`

Every Tier B fragrance must have all of these fields after enrichment:

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

These fields feed three different downstream consumers:
1. **`retrieval_text`** construction (`enrich.py`) — all fields contribute to the embeddable string
2. **Structured branch scoring** — `formality`, `fresh_warm`, `day_night`, `likely_season` are used for attribute matching
3. **Image CNN scoring** — `formality`, `day_night`, `likely_season` are the fragrance-side targets for NLL computation

---

## Known Gaps and Limitations

### Structured Branch Does Not Use Query Context (P0 fix)

As documented above: `_struct_score()` ignores `ctx.eventType`, `ctx.timeOfDay`, `ctx.mood`. It returns a fixed attribute-average score per fragrance. This means the 15% structured weight is a permanent bias toward formal, warm fragrances — not a query-sensitive signal. Fix: map context fields to attribute targets and compute distance-based scores.

### Tier A Coverage Without Enrichment (by design)

33,889 fragrances have no enrichment attributes. They are reachable only via raw text similarity. Niche fragrances outside the top 2,000 by rating count compete at a significant disadvantage.

### Image CNN Calibration

The CNN softmax probabilities are likely overconfident (standard neural network behavior). Temperature scaling should be applied to calibrate the probabilities before NLL computation. Current behavior: NLL scores may cluster near extremes, reducing discrimination.

### Discretization Cliff Effects

The CNN attribute-to-fragrance matching discretizes continuous floats at fixed thresholds: formality 0.33/0.67, day_night 0.5. A fragrance with `formality=0.66` and one with `formality=0.68` map to different classes despite being nearly identical. This introduces cliff effects that can rank very similar fragrances very differently. Soft matching (interpolating between classes) would fix this.

### Multimodal Embedding Requires GPU at Inference

`sig_mm` is zeroed on T4. The demo degradation path is tested and graceful, but it means T4 demos produce qualitatively different recommendations than A100 demos. This should be documented clearly in presentation materials.

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

**Label generation:** 3 independent runs of `gemini-3.1-pro-preview` per case. Cases kept only if there is strong agreement across runs (majority vote on all categorical fields). Confidence <0.6 → case is discarded or reworked.

**Primary scoring (metadata-based):**
- `attribute_match@3` / `@5`: do top-3/5 results match target formality, season, day/night, fresh/warm?
- `neighborhood_hit@3` / `@5`: do top-3/5 results fall in an acceptable accord/note-family neighborhood?

**Secondary scoring (LLM judge):**
- Judge: `gemini-2.5-pro` — different from label generator to avoid self-confirmation bias
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

| Type | Format | Example path |
|---|---|---|
| Embedding matrix | `.npy` float32 | `fragrance_raw_full/embeddings.npy` |
| Embedding metadata | `.csv` | `fragrance_raw_full/metadata.csv` |
| Enriched fragrance data | `.csv` | `vibescent_enriched_2k_v2.csv` |
| Pipeline manifest | `.json` | `fragrance_raw_full/manifest.json` |
| Fusion weights | `.json` | `week3/best_weights.json` |
| Reranker decision | `.json` | `week3/reranker_decision.json` |
| Demo cache | `.json` | `week4/locked_responses.json` |
| Frontend patch | `.diff` | `week4/frontend_patch.diff` |
| Evaluation results | `.md` | `results/week2_report.md` |

Every `.npy` artifact has a companion `manifest.json` that stores: model name, commit SHA, row count, dimensions, creation timestamp, pipeline version. `stage_complete()` uses this manifest to determine whether to re-run a stage.

---

## Week Deliverable Status

### Week 2 (deadline: April 12, 2026)

| Deliverable | Owner | Status |
|---|---|---|
| Fragrance dataset selected and cleaned | Darren | ✓ `vibescent_500.csv` |
| Shared schema locked | Gavin | ✓ |
| Fragrance enrichment pipeline | Harsh | ✓ `enrich.py` |
| Occasions embeddings + heatmap | Harsh | ✓ (in notebook) |
| Raw fragrance embeddings (35K) | Harsh | ✓ (Stage 5) |
| Enriched fragrance embeddings (2K) | Harsh | ✓ (Stages 10–14) |
| Multimodal doc embeddings (2K) | Harsh | ✓ (Stage 17) |
| RAW vs ENRICHED retrieval comparison | Harsh | ✓ (Stage 15) |
| `display_text` generation | Karan | Skipped (Stage 12 no-op) |
| CLIP branch | Neil | Delivered (CNNCLIPHybrid) |
| CNN branch | Neil | Delivered (CNNCLIPHybrid) |
| Hybrid branch (CNN-CLIP fusion) | Neil | Delivered (CNNCLIPHybrid) |
| `results/week2_report.md` | Harsh | In progress |

### Week 3

- Late-fusion baseline working end-to-end with all 4 signals
- 20-case benchmark published (Gavin)
- Grid search over fusion weights → `best_weights.json`
- Reranker added on top of baseline; comparison completed
- Ablation: with and without `Qwen3-VL-Embedding-8B`
- Baseline vs reranker comparison → `reranker_decision.json`
- Karan refines fragrance representation based on benchmark failures

### Week 4

- Final demo flow locked (VibeScentEngine + FastAPI + dual tunnel)
- 5 locked responses pre-generated (`locked_responses.json`)
- Frontend patch (`frontend_patch.diff`) delivered to Darren
- Final benchmark numbers locked and presentation written around real results

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
- Image CNN (0.30): the only signal that reads the outfit's visual attributes explicitly through a classifier trained on that task. Equal to text because visual information and textual information are the two primary modalities.
- Multimodal (0.25): covers both modalities simultaneously but operates only on 2,000 fragrances and partially overlaps with text. Slightly lower weight to avoid double-counting the text signal that both branches share.
- Structured (0.15): currently does not use query context (known gap). Lower weight until the implementation is fixed. Even at 15%, it contributes a ground-truth attribute bias that the neural branches may not capture.

Equal weights (0.25 each) are used as the safe fallback when `best_weights.json` is missing.

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

**Q: Why is the image branch scored via NLL rather than embedding the outfit image into the same 1024-d space and doing cosine similarity?**

Two different answers depending on which branch you're comparing to:

**vs. multimodal branch:** embedding the outfit image in the same 1024-d space IS what the multimodal branch does via Qwen3-VL. The multimodal branch produces a joint image+text query vector and does cosine similarity. The image CNN is a deliberately different architectural choice — it classifies rather than embeds. The reason: CLIP-family embeddings are opaque. You can't inspect what a cosine similarity score of 0.73 means in terms of outfit attributes. The CNN tells you `P_formal = 0.83` — you know exactly what the model predicted and can debug it. Both signals coexist because they capture different aspects: continuous semantic similarity (multimodal) vs. discrete attribute matching (CNN).

**vs. pure cosine:** if you embedded the outfit image with the CNN (without the classification heads), you'd get a feature vector but you'd have no fragrance vectors to compare it against. The fragrance database doesn't have images — it has text and structured attributes. The CNN bridges the modality gap by classifying the outfit image into the same discrete space that the fragrance enrichment attributes occupy. You then score fragrances by matching attributes, not by comparing embeddings across modalities.

**Q: Why joint probability (`P_formal × P_season × P_time`) for image scoring instead of average (`(P_formal + P_season + P_time) / 3`)?**

The joint probability is mathematically correct under the assumption that the three head predictions are conditionally independent given the outfit. It answers: "what is the probability that this fragrance is correct on ALL THREE dimensions simultaneously?"

The sum/average asks: "on average, how well does this fragrance fit?" A fragrance with `P_formal=0.9, P_season=0.05, P_time=0.9` would score `(0.9+0.05+0.9)/3 = 0.617` by average but `0.9 × 0.05 × 0.9 = 0.041` by product. The product correctly penalizes a fragrance that fails on one dimension even if it succeeds on others. A winter fragrance recommended for a summer outfit should score near zero — the product enforces this; the average doesn't.

The tradeoff: the product is more sensitive to overconfident low probabilities. If the CNN assigns `P_season = 0.02` for the correct season (miscalibration), the product collapses. This is a real risk and is why temperature calibration matters.

**Q: Why `argpartition` for top-K selection instead of full sort?**

`np.argpartition(-fused, k)[:k]` runs in O(N) average time (introselect algorithm). `np.argsort` runs in O(N log N). For N=35,889 and k=50, this is about a 9× speedup on the selection step. The result from `argpartition` is unordered — the top 50 are identified but not ranked among themselves. A secondary sort on just those 50 elements costs O(k log k) ≈ O(50 × 6) = negligible. Total: O(N) + O(k log k) vs O(N log N).

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

**Q: Why Qwen3.5-27B for enrichment instead of a much smaller model (Llama-3-8B, Mistral-7B)?**

Enrichment requires domain expertise: the model must correctly infer `formality=0.88` for BR540 vs `formality=0.22` for a beachy aquatic. This is not a task that can be solved by pattern matching on training data — it requires understanding of cultural fragrance norms, occasion dressing conventions, and sensory-to-conceptual translation.

Smaller models (7-8B) tend to produce fluent but inaccurate enrichment on this task — they hallucinate plausible-sounding but wrong formality scores, and their `vibe_sentence` outputs are generic ("a pleasant fragrance for all occasions"). 27B models have enough capacity to have absorbed fragrance-specific cultural knowledge from the training corpus and can make nuanced distinctions.

More critically: enrichment runs once offline. The GPU cost of 27B vs 8B for 2,000 rows is roughly 3× longer at inference time — acceptable for a one-time offline job. If this were running live per-request, the 8B model would be necessary.

**Q: Why Gemini for the reranker instead of a local VLM?**

The reranker needs to jointly reason about: an outfit image, a 500-character occasion description, and 10 structured fragrance profiles. This requires strong vision-language reasoning ability. Options:

1. **Gemini 3.1 Pro Preview:** top-tier multimodal reasoning, structured output via `response_schema`, handles long context. API dependency means ~2-5s latency per reranker call.

2. **Local VLM (LLaVA-34B, Qwen3-VL-72B):** requires ~40–70GB VRAM. Even an A100 (40GB) can't hold the reranker alongside the embedding models already loaded. Memory-swapping between models would destroy demo latency.

3. **Qwen3-VL-Embedding-8B (already loaded):** this model produces embeddings, not text outputs. It cannot generate rationale text or produce per-candidate scores.

Gemini is the only option that provides strong multimodal reasoning without additional GPU memory pressure. The API latency (~2-5s) is acceptable for a non-cached demo request where the user expects a moment of "thinking."

**Q: Why use a different model (`gemini-2.5-pro`) as evaluation judge vs the label generator (`gemini-3.1-pro-preview`)?**

Self-confirmation bias. If the same model generates the benchmark labels and then evaluates whether the system's output matches those labels, it will systematically prefer outputs that resemble its own generation style — not outputs that are actually correct.

This is a well-known problem in LLM evaluation: LLM-as-judge studies show that models prefer their own outputs 60–70% of the time even when the other output is objectively better by human standards. Using a different model family breaks this pattern and makes the evaluation more credible.

Additionally, `gemini-2.5-pro` is a stronger reasoning model — it may catch errors in the labels themselves that `gemini-3.1-pro-preview` would not, because it's evaluating from a position of greater capability.

---

### Enrichment Design

**Q: Why use an LLM for enrichment instead of traditional NLP (keyword extraction, TF-IDF, rule-based classifiers)?**

Traditional NLP would give you:
- TF-IDF weights over note vocabulary → tells you which notes are distinctive, not what the fragrance means culturally
- Named entity recognition → extracts brand names, note names, not occasion suitability
- Rule-based classifiers → requires manually encoding "saffron + oud → formal" rules, which is incomplete and unmaintainable at 2,000+ fragrances

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
- Output syntactically invalid JSON (unclosed bracket on token 1800/2000)

With 2,000 enrichment calls, you expect some failure rate even if each individual call succeeds 99% of the time. Prompt-and-validate requires: try → catch exception → retry → catch again → log failure. This is three LLM calls and two exception catches per failed row.

With `outlines`, invalid output is structurally impossible. The token generation process is constrained so that the output is always valid JSON matching `EnrichmentSchemaV2`. No validation needed because correctness is enforced at generation time. For 2,000 rows, this eliminates an entire class of failures.

The cost: `outlines` requires loading the full model through the `outlines.models` API, which adds ~10s overhead per pipeline setup. Worth it for 2,000 rows.

**Q: Why is the 98% success rate threshold for enrichment validation the right value?**

2% failure rate on 2,000 rows = 40 fragrances with null `vibe_sentence`. Each failed row gets embedded with the raw `retrieval_text` only (no vibe, no mood tags, no enrichment signal). For those 40 fragrances, the embedding quality degrades to near-raw level.

At 98% threshold: 40 bad embeddings in a corpus of 2,000 is a signal-to-noise ratio that won't noticeably degrade retrieval metrics. Benchmark cases that happen to need one of those 40 fragrances will be affected, but it's unlikely to dominate the 20-case evaluation.

Why not 99% or 100%? Enrichment failures have two causes: (1) rate limiting from Gemini API, (2) truncation issues with verbose fragrance entries. Both are manageable but not eliminatable. A 99% threshold would fail the pipeline on 20 bad rows and force a full re-run. 98% is the practical ceiling given real-world API reliability.

**Q: Why re-run `enrich_dataframe()` with a 70% truncated prompt as the retry?**

LLM API failures on long prompts have two main causes: (1) the prompt exceeds a context limit (hard fail) or (2) the model gets confused by extremely verbose input and produces malformed output. Both are more likely as prompt length increases.

Truncating to 70% of the original prompt sacrifices: the least important fields (usually `category`, `concentration`, `gender` appear near the end of `_build_prompt()`). These are lower-signal for enrichment than `top_notes`, `middle_notes`, `base_notes`, `main_accords`. The retry with 70% loses some context but still contains the core fragrance information. This heuristic resolves ~80% of prompt-length-related failures without requiring a sophisticated field prioritization system.

---

### Corpus and Tier Design

**Q: Why select Tier B by `rating_count` instead of review richness, metadata completeness, or random sampling?**

`rating_count` is a proxy for **documentation quality**. A fragrance with 50,000 ratings on Fragrantica has been extensively reviewed and is culturally well-known. Its notes list is likely complete, its accords are validated by many reviewers, and there is substantial cultural writing about it that the enrichment LLM has likely seen during training. An LLM enriching a well-known fragrance produces more accurate results because it has prior knowledge from its training data.

A fragrance with 23 ratings may have incomplete notes, uncertain classification, and little cultural writing — the LLM is essentially guessing. Random sampling would mix high- and low-documentation fragrances, degrading the average enrichment quality.

`metadata_completeness` was the secondary criterion: all four note columns must be non-null (strict filter) because `retrieval_text` without heart and base notes loses important signal. High `rating_count` + complete metadata = best candidates for enrichment.

**Q: Why 2,000 for Tier B specifically? Why not 5,000 or 500?**

- **Enrichment cost:** 2,000 rows × ~1-2s per LLM call = 33-67 minutes on Qwen3.5-27B or Gemini. 5,000 rows would be 2.5-5 hours — acceptable but risks Colab session timeouts. 2,000 is inside a reliable single-session budget.
- **Multimodal embedding cost:** Qwen3-VL embedding at batch_size=16 on A100. 2,000 rows ≈ 125 batches ≈ 20-30 minutes. 5,000 rows would be 50-75 minutes — still feasible but pushing the session limit with all other stages.
- **Coverage vs. quality:** the top 2,000 by rating count covers virtually all fragrances a user would plausibly know or want. Fragrances ranked 2,001–5,000 are progressively more obscure. The marginal retrieval improvement from adding them to the enriched corpus is likely small.
- **500 is too few:** the 20-case benchmark requires sufficient diversity in the candidate pool. A 500-fragrance enriched corpus may not contain good matches for all 20 benchmark occasions.

**Q: Why keep the full 35K Tier A raw corpus instead of just enriching everything you can?**

Tier A serves as a **long-tail coverage net**. A user wearing a very specific aesthetic (early 1990s French niche, Japanese wabi-sabi minimalism) may have a perfect match in fragrance #8,000 by rating count. If Tier A didn't exist, that fragrance would be invisible. It competes at a disadvantage (text-only, no enrichment) but it can still be retrieved.

The alternative — "only index what you've enriched" — would limit the system to 2,000 fragrances. This is a serious limitation for a portfolio demo trying to show broad fragrance knowledge.

The design says: use Tier B for quality, use Tier A for coverage. The two serve different user needs.

**Q: Why require `top_notes`, `middle_notes`, `base_notes`, AND `main_accords` for strict Tier B inclusion?**

The `retrieval_text` construction pipeline uses all four:
```
Top: {top_notes} | Heart: {middle_notes} | Base: {base_notes}
Accords: {main_accords}
```
Missing `base_notes` means the fragrance's dry-down — often its most distinctive phase — is absent from the embedding. Missing `main_accords` removes the highest-level semantic clustering signal. Either omission degrades the retrieval text enough to affect embedding quality.

The fallback filter (relax to just `top_notes` + `main_accords`) exists because some legitimate fragrances don't have published middle/base note breakdowns (e.g. single-material soliflores like certain pure oud extracts). The strict filter is preferred; relaxation is a safety valve.

---

### Scoring Mathematics

**Q: Why is cosine similarity the right distance metric instead of L2 (Euclidean) or dot product?**

For embeddings, three options:

1. **Dot product:** `a · b`. Problematic because it depends on both the angle between vectors AND their magnitudes. A long vector will have high dot product similarity with everything, even semantically unrelated content. Embedding models trained with L2 normalization (like Qwen3) produce unit vectors — at unit length, dot product equals cosine similarity. For non-normalized vectors, dot product is problematic.

2. **L2 distance:** `||a - b||`. Works for nearest-neighbor search but conflates magnitude differences with semantic differences. Two fragrance embeddings that encode similar meaning but are scaled differently would appear far apart. Also, argmin(L2) ≠ argmax(dot product) for non-normalized vectors.

3. **Cosine similarity:** `(a · b) / (||a|| × ||b||)`. Measures only the angle between vectors, ignoring magnitude. Semantically similar texts should point in the same direction in embedding space regardless of magnitude. This is the standard for text retrieval because embedding model training maximizes cosine similarity for semantically related pairs.

The pipeline L2-normalizes all embeddings before storage (`normalize_rows()` in `similarity.py`). For unit vectors: `cos(θ) = a · b` exactly. So the matrix multiply `TEXT_EMBEDDINGS @ q.T` computes cosine similarity without the division. This is O(N × D) — optimal.

**Q: Why truncate to 1024 dimensions when the model can produce 4096?**

Three reasons:

1. **Cross-modal compatibility.** The text and multimodal branches must score against the same fragrance index for the fusion to be coherent. Qwen3-Embedding-8B produces 1024-d (or higher) vectors. Qwen3-VL-Embedding-8B produces up to 4096-d. Truncating both to 1024-d creates a common dimensionality.

2. **Memory.** `fragrance_raw_full/embeddings.npy` at 4096-d would be `35,889 × 4096 × 4 bytes ≈ 588 MB`. At 1024-d: `147 MB`. The former is problematic for GPU memory during inference when multiple models are already loaded.

3. **Matryoshka guarantee.** Qwen3's Matryoshka training explicitly optimizes each prefix to be a valid embedding. The first 1024 dimensions are not random — they are the highest-information dimensions by construction. Truncating to 1024 loses information (the other 3072 dimensions do add signal) but retains a semantically coherent representation.

The quality loss from 4096 → 1024 is measurable but small for retrieval tasks. The memory and compatibility savings are concrete.

**Q: Mathematically, what does L2 normalization of embeddings do and why is it essential before storing?**

L2 normalization maps every embedding vector to the unit hypersphere: `v_norm = v / ||v||₂` so that `||v_norm||₂ = 1`.

Why this matters:
- **Enables dot product as cosine similarity.** For unit vectors `a` and `b`: `a · b = ||a|| × ||b|| × cos(θ) = cos(θ)`. Without normalization, you'd need to divide by magnitudes at query time — a computation on every element of a 35,889-row matrix.
- **Prevents length bias.** Embedding models sometimes produce longer vectors for longer texts. Without normalization, a fragrance with a long `retrieval_text` would artificially score higher against any query, not because it's more semantically relevant but because its embedding has higher magnitude.
- **Pre-computation at storage time.** Normalizing once at embedding time (offline) means the per-query operation is pure matrix multiply. For 35,889 × 1024, this is ~147M floating-point multiply-accumulate operations — fast on any modern CPU or GPU.

**Q: Why use the negative log-likelihood for the image CNN scoring instead of a simpler formulation?**

NLL has three mathematical properties that make it correct here:

1. **Log-space addition = probability product.** `-NLL = log(P_formal) + log(P_season) + log(P_time) = log(P_formal × P_season × P_time)`. This is numerically stable — computing `0.83 × 0.71 × 0.85` directly can underflow for low probabilities, while summing logs never underflows.

2. **Proper probabilistic scoring.** NLL is the standard loss for classification tasks precisely because it is calibrated to the softmax output. A model that outputs the correct class with probability 0.99 gets NLL ≈ 0.01. One that outputs 0.50 gets NLL ≈ 0.69. The penalty grows logarithmically, not linearly — this correctly penalizes uncertainty without catastrophically punishing near-correct predictions.

3. **`exp(-NLL)` is the joint likelihood.** Taking `exp(-NLL) = P_formal × P_season × P_time` converts back to probability space, which is bounded [0, 1] and interpretable. A score of 0.85 means the CNN assigns 85% joint probability to this fragrance matching the outfit across all three attributes.

Alternative: a simpler approach would be `argmax` — take the most likely class for each head and do a binary match against the fragrance attribute. This throws away all probability information. A fragrance in the right class at P=0.51 scores the same as one at P=0.99. NLL uses the full distribution.

---

### CNN Architecture

**Q: Why three classification heads (formal, season, time) and not more (gender, longevity, fresh/warm)?**

The three selected heads — formality, season, time-of-day — are the dimensions where **visual appearance provides strong evidence**. You can infer outfit formality from collar, lapels, cut, and fabric. You can infer season from layering, fabric weight, and color palette. You can infer day vs. night from how dressed-up something is.

**What visual appearance cannot reliably predict:**
- Gender: contemporary fashion intentionally blurs gender conventions. A visual classifier trained on gendered categories would systematically mis-label gender-neutral outfits.
- Longevity: this is a property of the fragrance, not the outfit. The CNN cannot see the outfit and infer "this person wants a fragrance that lasts 12 hours."
- Fresh/warm: there is weak visual correlation (bright colors → fresh, dark/rich colors → warm) but it is too noisy to be a reliable head. The CNN would be learning color-to-warmth mappings that are more accurately captured by the structured branch using ground-truth enrichment attributes.

Adding more heads would require more training labels and would dilute the shared 256-d trunk's capacity. Three heads was the balance between expressiveness and training feasibility.

**Q: Why a shared trunk for all three heads vs. three independent classifiers?**

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

The LLM judge (`gemini-2.5-pro`) is used as secondary scoring specifically because it can catch holistic failures that metadata can't — a recommendation that is technically "formal" but completely wrong in mood or accord family. Both scoring methods together are more reliable than either alone.

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

`_occ_text()` in `VibeScentEngine`:
```python
def _occ_text(ctx):
    parts = [p for p in [ctx.eventType, ctx.timeOfDay, ctx.mood, ctx.customNotes] if p]
    return ", ".join(parts) if parts else "general occasion"
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

Detection: compute the distribution of enriched `formality` scores across Tier B and verify it roughly follows the expected distribution (not all-high, not all-low, covers the full range). This should be in `results/week2_report.md` Section 2 (fragrance embedding sanity check) but is currently pending.

**Q: What happens to the multimodal signal when `MM_EMBEDDINGS` and `TEXT_EMBEDDINGS` have different numbers of rows?**

The engine handles this in Stage 2 of Week 4:
```python
sig_mm = np.zeros(len(self._df), dtype=np.float32)
sig_mm[:len(sig_mm_raw)] = sig_mm_raw
```

If `MM_EMBEDDINGS` has 2,000 rows and `FRAGRANCE_DF` has 2,000 rows, this is exact. If they're misaligned (edge case from Week 2 pipeline producing different-length files), the multimodal signal is truncated or padded with zeros. The Stage 2 alignment logic also handles this:
```python
_n = min(len(TEXT_EMBEDDINGS), len(FRAGRANCE_DF))
ACTIVE_TEXT_EMBEDDINGS = TEXT_EMBEDDINGS[:_n]
FRAGRANCE_DF = FRAGRANCE_DF.iloc[:_n].reset_index(drop=True)
```

This is a defensive truncation to the minimum — you lose some rows but avoid index-out-of-bounds errors during inference.

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

## References

- Qwen3-VL-Embedding-8B: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
- MMEB Leaderboard: https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard
- Gemini 3.1 Pro Preview: https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview
- Gemini structured outputs: https://ai.google.dev/gemini-api/docs/structured-output
- Gemini Batch API: https://ai.google.dev/gemini-api/docs/batch-api
- outlines (constrained generation): https://github.com/dottxt-ai/outlines
- Matryoshka Representation Learning: https://arxiv.org/abs/2205.13147
