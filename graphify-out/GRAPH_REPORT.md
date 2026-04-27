# Graph Report - vibescent  (2026-04-26)

## Corpus Check
- 77 files · ~118,379 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 785 nodes · 1313 edges · 52 communities detected
- Extraction: 62% EXTRACTED · 38% INFERRED · 0% AMBIGUOUS · INFERRED: 497 edges (avg confidence: 0.7)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]

## God Nodes (most connected - your core abstractions)
1. `EnrichmentSchemaV2` - 38 edges
2. `Settings` - 32 edges
3. `ContextInput` - 31 edges
4. `VibeScoreEngine` - 23 edges
5. `ImageHeadProbabilities` - 19 edges
6. `RecommendRequest` - 17 edges
7. `RecommendResponse` - 17 edges
8. `Qwen3VLMultimodalEmbedder` - 17 edges
9. `main()` - 15 edges
10. `_parse_enrichment()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Tuxedo Outfit` --semantically_similar_to--> `Gala or Black Tie Event`  [INFERRED] [semantically similar]
  artifacts/colab_upload_bundle/assets/outfits/tuxedo.jpg → assets/outfits/tuxedo.jpg
- `Tuxedo Outfit` --semantically_similar_to--> `Wedding Formal Occasion`  [INFERRED] [semantically similar]
  artifacts/colab_upload_bundle/assets/outfits/tuxedo.jpg → assets/outfits/tuxedo.jpg
- `Settings` --uses--> `Generate one enrichment object from a prompt.`  [INFERRED]
  src/vibescents/settings.py → artifacts/colab_upload_bundle/src/vibescents/enrich.py
- `Settings` --uses--> `Enrich dataframe rows with EnrichmentSchemaV2 fields.`  [INFERRED]
  src/vibescents/settings.py → artifacts/colab_upload_bundle/src/vibescents/enrich.py
- `Settings` --uses--> `Build retrieval_text from raw and enriched fields.`  [INFERRED]
  src/vibescents/settings.py → artifacts/colab_upload_bundle/src/vibescents/enrich.py

## Hyperedges (group relationships)
- **Four-Signal Retrieval and Fusion Pipeline** — presentation_text_retrieval_branch, presentation_multimodal_branch, presentation_image_cnn_branch, presentation_structured_branch, readme_score_fusion_formula [EXTRACTED 0.95]
- **Enrichment as Vocabulary Translation (core system insight)** — presentation_core_insight, presentation_enrichment_pipeline, karan_retrieval_text_enriched, karan_embedding_text_raw [EXTRACTED 0.90]
- **Team Ownership and Module Responsibility** — readme_contributor_harsh, readme_contributor_neil, readme_contributor_karan, readme_contributor_darren, readme_src_vibescents_pkg, neil_cnn_clip_hybrid_model, karan_cluster_vibe_pipeline, darren_nextjs_frontend [EXTRACTED 0.90]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.04
Nodes (71): EnrichmentClient, Enrich dataframe rows with EnrichmentSchemaV2 fields., Enrich dataframe rows with EnrichmentSchemaV2 fields., Build retrieval_text from raw and enriched fields., Enrich dataframe rows with EnrichmentSchemaV2 fields., Build retrieval_text from raw and enriched fields., Generate one enrichment object from a prompt., Generate one enrichment object from a prompt. (+63 more)

### Community 1 - "Community 1"
Cohesion: 0.03
Nodes (68): Backend Python Requirements (requirements.txt), Colab Upload Bundle Requirements, A100 40GB GPU Requirement, GPU Resource Proposal (Harsh → Vayun Malik), Embedding Space Mismatch Problem, Qwen3-VL-Reranker-8B, Qwen3-VL Unified Stack Proposal, SigLIP 2 SO400M Image Classifier (+60 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (37): create_app(), create_configured_app(), decode_request_image(), Generate recommendation response for the frontend contract., Generate recommendation response for the frontend contract., Production entry point — injects VibeScoreEngine from disk artifacts.      Launc, RecommendationEngine, UnconfiguredRecommendationEngine (+29 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (43): consolidate_case_drafts(), main(), _check_artifacts(), Step 4 — Text-to-fragrance retrieval comparison: RAW vs ENRICHED.  Loads pre-com, run_comparison(), dump_json(), ensure_dir(), guess_mime_type() (+35 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (33): CLIPStandalone, Frozen CLIP ViT-L/14 image encoder with a projection MLP and 5 task heads., Return 512-d projected CLIP features. (B, 512), Args:             x: (B, 3, 224, 224) — pixel values pre-normalised with CLIP me, Compute per-task losses and weighted total.          Args:             output: d, CNNBaseline, Return 512-d projected backbone features. (B, 512), Args:             x: (B, 3, 224, 224)         Returns:             dict with key (+25 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (27): BaseModel, BenchmarkGenerator, BenchmarkLabelClient, _build_outlines_generator(), _parse_benchmark_label(), QwenOutlinesBenchmarkLabelClient, Generate a benchmark label for one case., Generate a benchmark label for one case. (+19 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (28): _append_failure_record(), _build_guided_decoding_params(), _build_outlines_generator(), _build_prompt(), _build_retrieval_text(), _build_vllm_sampling_params(), _call_with_schema(), enrich_dataframe() (+20 more)

### Community 7 - "Community 7"
Cohesion: 0.1
Nodes (31): _coerce_float(), _coerce_head_outputs(), _coerce_str(), discretize_day_night(), discretize_formality(), discretize_frequency(), discretize_gender(), from_checkpoint() (+23 more)

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (19): build_prompt_embeddings(), encode_images(), encode_texts(), get_device(), main(), class_embs: { class_id_str: Tensor(num_prompts, D) }     Returns argmax class id, Return L2-normalised text embeddings, shape (N, D)., Return L2-normalised image embeddings, shape (N, D). (+11 more)

### Community 9 - "Community 9"
Cohesion: 0.18
Nodes (24): context_to_query_string(), Assemble a natural-language query string from a ContextInput., ContextInput, compute_structured_scores(), Score each row in df against the user context.     Returns float32 array of shap, test_all_known_fields_join_with_pipe(), test_custom_notes_appended(), test_custom_notes_stripped_of_whitespace() (+16 more)

### Community 10 - "Community 10"
Cohesion: 0.14
Nodes (21): _parse_price(), search_perfume(), search_perfumes(), _store_priority(), search(), SearchRequest, test_parse_price_empty_string(), test_parse_price_no_digits() (+13 more)

### Community 11 - "Community 11"
Cohesion: 0.11
Nodes (19): decode_b64_image_bytes(), decode_b64_to_cnn_tensor(), Image preprocessing for Neil's CNN-CLIP hybrid model.  Decodes base64-encoded im, Decode a base64 image string to raw bytes.      Parameters     ----------     im, Decode a base64 image and return a CLIP-normalized CNN input tensor.      Perfor, _build_pil_stub(), _build_tv_functional_stub(), Return torchvision.transforms.functional stub wired to return fake_tensor. (+11 more)

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (22): _parse_notes(), _str_or_none(), _make_engine(), test_build_response_fallback_occasion(), test_build_response_top_3(), test_engine_init_normalises_embeddings(), test_fuse_all_channels(), test_fuse_image_and_structured_fallback() (+14 more)

### Community 13 - "Community 13"
Cohesion: 0.12
Nodes (21): API Recommend Route (/api/recommend), ContextForm Component, ContextInput Interface, Design Tokens (color palette, typography), FragranceRecommendation Interface, Framer Motion, lib/types.ts (shared TypeScript types), Next.js App Router (14+) (+13 more)

### Community 14 - "Community 14"
Cohesion: 0.15
Nodes (16): build_cluster_prompt(), get_cluster_representatives(), get_fragrance_vibe(), label_clusters_heuristic(), label_clusters_with_llm(), load_cluster_vibes(), VibeScent — LLM-Based Cluster Vibe Labeling Replaces the manual NOTES_VIBE_DICT, Rule-based fallback for testing without GPU access.     Uses keyword matching on (+8 more)

### Community 15 - "Community 15"
Cohesion: 0.21
Nodes (14): build_weight_grid(), _fuse_normalized(), fuse_scores(), grid_search_weights(), GridSearchResult, min_max_normalize(), normalize_signal_map(), Weighted sum of already-normalized signal vectors (no re-normalization). (+6 more)

### Community 16 - "Community 16"
Cohesion: 0.18
Nodes (13): Black Tie Occasion, Monochrome Black and White Palette, Formal Wear Category, Black Tie Formality, Tuxedo Jacket, Dress Trousers, Gala or Black Tie Event, Wedding Formal Occasion (+5 more)

### Community 17 - "Community 17"
Cohesion: 0.29
Nodes (10): build_parser(), test_parser_embed_csv_optional_model(), test_parser_embed_csv_required_args(), test_parser_embed_occasions(), test_parser_generate_benchmark_defaults(), test_parser_missing_required_arg_exits(), test_parser_multimodal_retrieve_custom_top_k(), test_parser_multimodal_retrieve_defaults() (+2 more)

### Community 18 - "Community 18"
Cohesion: 0.27
Nodes (9): POST(), test_search_empty_perfumes_returns_400(), test_search_multiple_fragrances(), test_search_negative_budget_returns_400(), test_search_none_result_uses_fallback(), test_search_returns_200_for_valid_request(), test_search_serpapi_key_missing_returns_500(), test_search_unexpected_exception_returns_500() (+1 more)

### Community 19 - "Community 19"
Cohesion: 0.52
Nodes (12): Casual/Relaxed Cluster (casual_day, streetwear_night, summer_party), Formal Daytime Cluster (creative_office, business_dinner, wedding_guest, black_tie), Occasion Similarity Heatmap, Black Tie Occasion, Business Dinner Occasion, Casual Day Occasion, Creative Office Occasion, Editorial Occasion (+4 more)

### Community 20 - "Community 20"
Cohesion: 0.23
Nodes (12): Beige / Tan Color Palette, Casual Outfit Placeholder Image, Casual Style Category, Editorial Outfit Image (Placeholder), Editorial Style Category, Everyday / Casual Occasion, Low Formality Level, Editorial Fashion Style (+4 more)

### Community 21 - "Community 21"
Cohesion: 0.23
Nodes (12): Business Casual Formality, White Dress Shirt, Dark Trousers, Quilted Vest, Confident Mood, Polished Mood, Casual Friday Occasion, Office/Business Occasion (+4 more)

### Community 22 - "Community 22"
Cohesion: 0.24
Nodes (12): Casual Formality Level, Fresh Aquatic Tropical Fragrance Profile, Hawaiian Aloha Shirt, Hawaiian Shirt Outfit Photo, Golden Hour Sunset Lighting, Resort / Outdoor Casual Occasion, Vacation / Beach Occasion, Blue and White Color Palette (+4 more)

### Community 23 - "Community 23"
Cohesion: 0.2
Nodes (7): FashionpediaDataset, load_fashionpedia_data(), PyTorch Dataset for Fashionpedia outfit images with multi-task attributes., Args:             image_paths: List of paths to image files             labels_d, Returns (image_tensor, label_dict)., Load Fashionpedia dataset with train/val split.      Args:         data_dir: Roo, Dataset

### Community 24 - "Community 24"
Cohesion: 0.38
Nodes (3): handleChange(), handleDrop(), processFile()

### Community 25 - "Community 25"
Cohesion: 0.33
Nodes (5): inspect_cluster(), match_outfit_to_clusters(), VibeScent — Pipeline Sanity Tests Run this after label_cluster_vibes.py to check, Print a cluster's vibe vector and sample fragrances., Simulate the inference pipeline:     outfit → LLM → vibe vector → find nearest f

### Community 26 - "Community 26"
Cohesion: 0.4
Nodes (2): handleSubmit(), getRecommendations()

### Community 27 - "Community 27"
Cohesion: 0.83
Nodes (4): Red Background with YouTube Logos, Portrait Photo - Young Man with YouTube Background, Young Adult Male Subject, YouTube Platform Branding

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (2): Colab T4 Lightweight Dependencies (requirements.colab.t4.txt), Gemma 4 E4B int4 (T4 lightweight enrichment)

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): tailwind.config.ts

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): postcss.config.mjs

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): next-env.d.ts

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): eslint.config.mjs

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Wrap Neil's CNN-CLIP hybrid model with probability extraction helpers.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): VibeScent — Merge & Deduplicate Fragrance Datasets Combines Parfumo, rdemarqui,

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Lowercase, strip whitespace, collapse spaces.

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Dedup key from name + brand.

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Parfumo CSV from TidyTuesday.

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): rdemarqui perfume_database CSV.

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Rawanalqarni Perfume_Dataaset CSV.

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Fragrantica cleaned dataset from Kaggle (olgagmiufana1).

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Ayush perfume dataset from Kaggle.

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Given multiple rows for the same fragrance (from different sources),     produce

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): dataset_paths: dict mapping loader name -> local file path, e.g.         {"parfu

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Patch harsh_week5_qwen3vl.ipynb: Stage 3 (TF32/FA2/compile) + Stage 4 (batch rer

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Takes a list of string notes, looks them up in the dictionary,      and averages

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Wrap Neil's CNN-CLIP hybrid model with probability extraction helpers.

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Compute per-task losses and weighted total.          Args:             output: d

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Loads the model, processor, and prompt embeddings into memory.

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Decodes the Next.js image, runs it through the pre-loaded CLIP model,      and f

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Partitions the dataframe using categorical keys, then uses an AI language      m

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): PipelineVisual Component

## Knowledge Gaps
- **130 isolated node(s):** `tailwind.config.ts`, `postcss.config.mjs`, `next-env.d.ts`, `eslint.config.mjs`, `Weighted sum of already-normalized signal vectors (no re-normalization).` (+125 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 26`** (5 nodes): `page.tsx`, `recommend.ts`, `handleImageReady()`, `handleSubmit()`, `getRecommendations()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (2 nodes): `Colab T4 Lightweight Dependencies (requirements.colab.t4.txt)`, `Gemma 4 E4B int4 (T4 lightweight enrichment)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `tailwind.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `postcss.config.mjs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `next-env.d.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `eslint.config.mjs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Wrap Neil's CNN-CLIP hybrid model with probability extraction helpers.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `VibeScent — Merge & Deduplicate Fragrance Datasets Combines Parfumo, rdemarqui,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Lowercase, strip whitespace, collapse spaces.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Dedup key from name + brand.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Parfumo CSV from TidyTuesday.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `rdemarqui perfume_database CSV.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Rawanalqarni Perfume_Dataaset CSV.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Fragrantica cleaned dataset from Kaggle (olgagmiufana1).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Ayush perfume dataset from Kaggle.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Given multiple rows for the same fragrance (from different sources),     produce`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `dataset_paths: dict mapping loader name -> local file path, e.g.         {"parfu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Patch harsh_week5_qwen3vl.ipynb: Stage 3 (TF32/FA2/compile) + Stage 4 (batch rer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Takes a list of string notes, looks them up in the dictionary,      and averages`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Wrap Neil's CNN-CLIP hybrid model with probability extraction helpers.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Compute per-task losses and weighted total.          Args:             output: d`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `Loads the model, processor, and prompt embeddings into memory.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Decodes the Next.js image, runs it through the pre-loaded CLIP model,      and f`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `Partitions the dataframe using categorical keys, then uses an AI language      m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `PipelineVisual Component`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Settings` connect `Community 2` to `Community 0`, `Community 3`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Why does `EnrichmentSchemaV2` connect `Community 0` to `Community 3`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Why does `VibeScoreEngine` connect `Community 2` to `Community 9`, `Community 12`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 36 inferred relationships involving `EnrichmentSchemaV2` (e.g. with `EnrichmentClient` and `QwenOutlinesEnrichmentClient`) actually correct?**
  _`EnrichmentSchemaV2` has 36 INFERRED edges - model-reasoned connections that need verification._
- **Are the 31 inferred relationships involving `Settings` (e.g. with `Qwen3VLReranker` and `Qwen3-VL-Reranker-8B pointwise reranker — local GPU, zero API keys.`) actually correct?**
  _`Settings` has 31 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `ContextInput` (e.g. with `Score each row in df against the user context.     Returns float32 array of shap` and `VibeScoreEngine`) actually correct?**
  _`ContextInput` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `VibeScoreEngine` (e.g. with `Qwen3VLMultimodalEmbedder` and `CLIPImageScorer`) actually correct?**
  _`VibeScoreEngine` has 14 INFERRED edges - model-reasoned connections that need verification._