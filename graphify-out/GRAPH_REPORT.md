# Graph Report - .  (2026-04-25)

## Corpus Check
- 114 files · ~122,431 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 766 nodes · 1489 edges · 46 communities detected
- Extraction: 77% EXTRACTED · 23% INFERRED · 0% AMBIGUOUS · INFERRED: 342 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Recommendation Engine & Schemas|Recommendation Engine & Schemas]]
- [[_COMMUNITY_Infrastructure & Dependencies|Infrastructure & Dependencies]]
- [[_COMMUNITY_Enrichment Pipeline (Week 2)|Enrichment Pipeline (Week 2)]]
- [[_COMMUNITY_CLIP & CNN Vision Encoders|CLIP & CNN Vision Encoders]]
- [[_COMMUNITY_CLI & Evaluation Pipeline|CLI & Evaluation Pipeline]]
- [[_COMMUNITY_Image Scoring Module|Image Scoring Module]]
- [[_COMMUNITY_API Layer (Next.js + FastAPI)|API Layer (Next.js + FastAPI)]]
- [[_COMMUNITY_Fragrance Data Enrichment|Fragrance Data Enrichment]]
- [[_COMMUNITY_Image Preprocessing|Image Preprocessing]]
- [[_COMMUNITY_Qwen3 VL Embedding Model|Qwen3 VL Embedding Model]]
- [[_COMMUNITY_Benchmarking & Evaluation|Benchmarking & Evaluation]]
- [[_COMMUNITY_Dataset Merging & Cleaning|Dataset Merging & Cleaning]]
- [[_COMMUNITY_Frontend Demo Interface|Frontend Demo Interface]]
- [[_COMMUNITY_Score Fusion|Score Fusion]]
- [[_COMMUNITY_Zero-Shot CLIP Retrieval|Zero-Shot CLIP Retrieval]]
- [[_COMMUNITY_Fragrance Cluster Labeling|Fragrance Cluster Labeling]]
- [[_COMMUNITY_Formal Occasion Imagery|Formal Occasion Imagery]]
- [[_COMMUNITY_Fashionpedia Dataset Loader|Fashionpedia Dataset Loader]]
- [[_COMMUNITY_Occasion Similarity Clusters|Occasion Similarity Clusters]]
- [[_COMMUNITY_Business Casual Outfit|Business Casual Outfit]]
- [[_COMMUNITY_Hawaiian Casual Outfit|Hawaiian Casual Outfit]]
- [[_COMMUNITY_Placeholder Outfit Assets|Placeholder Outfit Assets]]
- [[_COMMUNITY_Perfume Search & Scraping|Perfume Search & Scraping]]
- [[_COMMUNITY_Outfit Image Uploader|Outfit Image Uploader]]
- [[_COMMUNITY_Vibe Pipeline Debugging|Vibe Pipeline Debugging]]
- [[_COMMUNITY_Demo Page & API Client|Demo Page & API Client]]
- [[_COMMUNITY_Clustered DB Builder|Clustered DB Builder]]
- [[_COMMUNITY_Notes-to-Vibe Mapping|Notes-to-Vibe Mapping]]
- [[_COMMUNITY_Context Form Component|Context Form Component]]
- [[_COMMUNITY_Test Image (YouTube)|Test Image (YouTube)]]
- [[_COMMUNITY_Week 5 Notebook Patches|Week 5 Notebook Patches]]
- [[_COMMUNITY_App Root Layout|App Root Layout]]
- [[_COMMUNITY_Landing Page|Landing Page]]
- [[_COMMUNITY_Hero Component|Hero Component]]
- [[_COMMUNITY_Navigation Bar|Navigation Bar]]
- [[_COMMUNITY_Footer Component|Footer Component]]
- [[_COMMUNITY_TagPill Component|Tag/Pill Component]]
- [[_COMMUNITY_Button Component|Button Component]]
- [[_COMMUNITY_Fragrance Card Component|Fragrance Card Component]]
- [[_COMMUNITY_Submit Button Component|Submit Button Component]]
- [[_COMMUNITY_T4 Lightweight Config|T4 Lightweight Config]]
- [[_COMMUNITY_Tailwind Config|Tailwind Config]]
- [[_COMMUNITY_PostCSS Config|PostCSS Config]]
- [[_COMMUNITY_Next.js Type Defs|Next.js Type Defs]]
- [[_COMMUNITY_ESLint Config|ESLint Config]]
- [[_COMMUNITY_Pipeline Visual Doc|Pipeline Visual Doc]]

## God Nodes (most connected - your core abstractions)
1. `EnrichmentSchemaV2` - 29 edges
2. `Settings` - 28 edges
3. `VibeScoreEngine` - 22 edges
4. `ImageHeadProbabilities` - 17 edges
5. `main()` - 16 edges
6. `Qwen3VLMultimodalEmbedder` - 16 edges
7. `RecommendResponse` - 15 edges
8. `CNNCLIPHybrid` - 15 edges
9. `retrieve_with_multimodal_query()` - 14 edges
10. `enrich_dataframe()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Tuxedo Outfit` --semantically_similar_to--> `Gala or Black Tie Event`  [INFERRED] [semantically similar]
  artifacts/colab_upload_bundle/assets/outfits/tuxedo.jpg → assets/outfits/tuxedo.jpg
- `Tuxedo Outfit` --semantically_similar_to--> `Wedding Formal Occasion`  [INFERRED] [semantically similar]
  artifacts/colab_upload_bundle/assets/outfits/tuxedo.jpg → assets/outfits/tuxedo.jpg
- `Next.js Frontend (ScentAI)` --semantically_similar_to--> `ScentAI Project`  [INFERRED] [semantically similar]
  docs/darren.md → CLAUDE.md
- `backend_app.py (FastAPI /healthz + /recommend)` --semantically_similar_to--> `FastAPI Backend`  [INFERRED] [semantically similar]
  docs/harsh.md → README.md
- `Colab Upload Bundle Requirements` --semantically_similar_to--> `Colab GPU Dependencies (requirements.colab.txt)`  [INFERRED] [semantically similar]
  artifacts/colab_upload_bundle/notebooks/requirements.colab.txt → notebooks/requirements.colab.txt

## Hyperedges (group relationships)
- **Four-Signal Retrieval and Fusion Pipeline** — presentation_text_retrieval_branch, presentation_multimodal_branch, presentation_image_cnn_branch, presentation_structured_branch, readme_score_fusion_formula [EXTRACTED 0.95]
- **Enrichment as Vocabulary Translation (core system insight)** — presentation_core_insight, presentation_enrichment_pipeline, karan_retrieval_text_enriched, karan_embedding_text_raw [EXTRACTED 0.90]
- **Team Ownership and Module Responsibility** — readme_contributor_harsh, readme_contributor_neil, readme_contributor_karan, readme_contributor_darren, readme_src_vibescents_pkg, neil_cnn_clip_hybrid_model, karan_cluster_vibe_pipeline, darren_nextjs_frontend [EXTRACTED 0.90]

## Communities

### Community 0 - "Recommendation Engine & Schemas"
Cohesion: 0.06
Nodes (42): Generate recommendation response for the frontend contract., Generate recommendation response for the frontend contract., Production entry point — injects VibeScoreEngine from disk artifacts.      Launc, BaseModel, Generate a benchmark label for one case., Generate a benchmark label for one case., Qwen3VLMultimodalEmbedder, Multimodal embedder backed by Qwen3-VL-Embedding-8B (local GPU, no API key). (+34 more)

### Community 1 - "Infrastructure & Dependencies"
Cohesion: 0.03
Nodes (68): Backend Python Requirements (requirements.txt), Colab Upload Bundle Requirements, A100 40GB GPU Requirement, GPU Resource Proposal (Harsh → Vayun Malik), Embedding Space Mismatch Problem, Qwen3-VL-Reranker-8B, Qwen3-VL Unified Stack Proposal, SigLIP 2 SO400M Image Classifier (+60 more)

### Community 2 - "Enrichment Pipeline (Week 2)"
Cohesion: 0.08
Nodes (57): EnrichmentSchemaV2, _enrichment_df(), _fake_disk_usage(), _make_df(), _make_torch_stub(), Ensure global model state is clean before and after every test., Build a synthetic fragrance DataFrame.      Parameters     ----------     n:, Write a valid manifest.json into *artifacts_dir*, matching stage_complete's expe (+49 more)

### Community 3 - "CLIP & CNN Vision Encoders"
Cohesion: 0.06
Nodes (34): CLIPStandalone, Frozen CLIP ViT-L/14 image encoder with a projection MLP and 5 task heads., Return 512-d projected CLIP features. (B, 512), Args:             x: (B, 3, 224, 224) — pixel values pre-normalised with CLIP me, Compute per-task losses and weighted total.          Args:             output: d, CNNBaseline, Return 512-d projected backbone features. (B, 512), Args:             x: (B, 3, 224, 224)         Returns:             dict with key (+26 more)

### Community 4 - "CLI & Evaluation Pipeline"
Cohesion: 0.09
Nodes (33): build_parser(), main(), _check_artifacts(), Step 4 — Text-to-fragrance retrieval comparison: RAW vs ENRICHED.  Loads pre-com, run_comparison(), dump_json(), ensure_dir(), guess_mime_type() (+25 more)

### Community 5 - "Image Scoring Module"
Cohesion: 0.12
Nodes (31): CPU-viable SigLIP 2 zero-shot classifier → ImageHeadProbabilities.      SigLIP 2, Run SigLIP 2 zero-shot on raw image bytes → ImageHeadProbabilities., SigLIP2ImageScorer, _coerce_float(), _coerce_head_outputs(), _coerce_str(), discretize_day_night(), discretize_formality() (+23 more)

### Community 6 - "API Layer (Next.js + FastAPI)"
Cohesion: 0.13
Nodes (16): create_app(), create_configured_app(), decode_request_image(), RecommendationEngine, set_recommendation_engine(), UnconfiguredRecommendationEngine, Protocol, Partitions the dataframe using categorical keys, then uses an AI language      m (+8 more)

### Community 7 - "Fragrance Data Enrichment"
Cohesion: 0.21
Nodes (20): _append_failure_record(), _build_outlines_generator(), _build_prompt(), _build_retrieval_text(), enrich_dataframe(), EnrichmentClient, GeminiEnrichmentClient, main() (+12 more)

### Community 8 - "Image Preprocessing"
Cohesion: 0.15
Nodes (23): decode_b64_image_bytes(), decode_b64_to_cnn_tensor(), Image preprocessing for Neil's CNN-CLIP hybrid model.  Decodes base64-encoded im, Decode a base64 image string to raw bytes.      Parameters     ----------     im, Decode a base64 image and return a CLIP-normalized CNN input tensor.      Perfor, _build_pil_stub(), _build_tv_functional_stub(), Return torchvision.transforms.functional stub wired to return fake_tensor. (+15 more)

### Community 9 - "Qwen3 VL Embedding Model"
Cohesion: 0.15
Nodes (11): ModelOutput, forward(), language_model(), _pooling_last(), Qwen3VLEmbedder, Qwen3VLForEmbedding, Qwen3VLForEmbeddingOutput, Vendored from Qwen/Qwen3-VL-Embedding-8B HuggingFace model repo.  Source: https: (+3 more)

### Community 10 - "Benchmarking & Evaluation"
Cohesion: 0.22
Nodes (10): BenchmarkGenerator, BenchmarkLabelClient, _build_outlines_generator(), consolidate_case_drafts(), GeminiBenchmarkLabelClient, _parse_benchmark_label(), QwenOutlinesBenchmarkLabelClient, _repair_payload() (+2 more)

### Community 11 - "Dataset Merging & Cleaning"
Cohesion: 0.17
Nodes (20): build_unified_dataset(), clean(), empty_unified(), load_ayush(), load_fragrantica_clean(), load_parfumo(), load_rawanalqarni(), load_rdemarqui() (+12 more)

### Community 12 - "Frontend Demo Interface"
Cohesion: 0.12
Nodes (21): API Recommend Route (/api/recommend), ContextForm Component, ContextInput Interface, Design Tokens (color palette, typography), FragranceRecommendation Interface, Framer Motion, lib/types.ts (shared TypeScript types), Next.js App Router (14+) (+13 more)

### Community 13 - "Score Fusion"
Cohesion: 0.26
Nodes (14): build_weight_grid(), _fuse_normalized(), fuse_scores(), grid_search_weights(), GridSearchResult, min_max_normalize(), normalize_signal_map(), Weighted sum of already-normalized signal vectors (no re-normalization). (+6 more)

### Community 14 - "Zero-Shot CLIP Retrieval"
Cohesion: 0.27
Nodes (14): build_prompt_embeddings(), encode_images(), encode_texts(), extract_vibe_dictionary(), get_device(), initialize_model(), main(), class_embs: { class_id_str: Tensor(num_prompts, D) }     Returns argmax class id (+6 more)

### Community 15 - "Fragrance Cluster Labeling"
Cohesion: 0.2
Nodes (16): build_cluster_prompt(), get_cluster_representatives(), get_fragrance_vibe(), label_clusters_heuristic(), label_clusters_with_llm(), load_cluster_vibes(), VibeScent — LLM-Based Cluster Vibe Labeling Replaces the manual NOTES_VIBE_DICT, Rule-based fallback for testing without GPU access.     Uses keyword matching on (+8 more)

### Community 16 - "Formal Occasion Imagery"
Cohesion: 0.18
Nodes (13): Black Tie Occasion, Monochrome Black and White Palette, Formal Wear Category, Black Tie Formality, Tuxedo Jacket, Dress Trousers, Gala or Black Tie Event, Wedding Formal Occasion (+5 more)

### Community 17 - "Fashionpedia Dataset Loader"
Cohesion: 0.2
Nodes (7): FashionpediaDataset, load_fashionpedia_data(), PyTorch Dataset for Fashionpedia outfit images with multi-task attributes., Args:             image_paths: List of paths to image files             labels_d, Returns (image_tensor, label_dict)., Load Fashionpedia dataset with train/val split.      Args:         data_dir: Roo, Dataset

### Community 18 - "Occasion Similarity Clusters"
Cohesion: 0.52
Nodes (12): Casual/Relaxed Cluster (casual_day, streetwear_night, summer_party), Formal Daytime Cluster (creative_office, business_dinner, wedding_guest, black_tie), Occasion Similarity Heatmap, Black Tie Occasion, Business Dinner Occasion, Casual Day Occasion, Creative Office Occasion, Editorial Occasion (+4 more)

### Community 19 - "Business Casual Outfit"
Cohesion: 0.23
Nodes (12): Business Casual Formality, White Dress Shirt, Dark Trousers, Quilted Vest, Confident Mood, Polished Mood, Casual Friday Occasion, Office/Business Occasion (+4 more)

### Community 20 - "Hawaiian Casual Outfit"
Cohesion: 0.24
Nodes (12): Casual Formality Level, Fresh Aquatic Tropical Fragrance Profile, Hawaiian Aloha Shirt, Hawaiian Shirt Outfit Photo, Golden Hour Sunset Lighting, Resort / Outdoor Casual Occasion, Vacation / Beach Occasion, Blue and White Color Palette (+4 more)

### Community 21 - "Placeholder Outfit Assets"
Cohesion: 0.23
Nodes (12): Beige / Tan Color Palette, Casual Outfit Placeholder Image, Casual Style Category, Editorial Outfit Image (Placeholder), Editorial Style Category, Everyday / Casual Occasion, Low Formality Level, Editorial Fashion Style (+4 more)

### Community 22 - "Perfume Search & Scraping"
Cohesion: 0.38
Nodes (6): search(), SearchRequest, _parse_price(), search_perfume(), search_perfumes(), _store_priority()

### Community 23 - "Outfit Image Uploader"
Cohesion: 0.5
Nodes (6): clearImage(), handleChange(), handleDrag(), handleDrop(), processFile(), triggerSelect()

### Community 24 - "Vibe Pipeline Debugging"
Cohesion: 0.38
Nodes (5): inspect_cluster(), match_outfit_to_clusters(), VibeScent — Pipeline Sanity Tests Run this after label_cluster_vibes.py to check, Print a cluster's vibe vector and sample fragrances., Simulate the inference pipeline:     outfit → LLM → vibe vector → find nearest f

### Community 25 - "Demo Page & API Client"
Cohesion: 0.33
Nodes (3): handleImageReady(), handleSubmit(), getRecommendations()

### Community 26 - "Clustered DB Builder"
Cohesion: 0.83
Nodes (2): build_clustered_database(), FragranceVibe

### Community 27 - "Notes-to-Vibe Mapping"
Cohesion: 0.5
Nodes (2): calculate_vibe_vector(), Takes a list of string notes, looks them up in the dictionary,      and averages

### Community 28 - "Context Form Component"
Cohesion: 0.67
Nodes (2): ContextForm(), PillGroup()

### Community 29 - "Test Image (YouTube)"
Cohesion: 0.83
Nodes (4): Red Background with YouTube Logos, Portrait Photo - Young Man with YouTube Background, Young Adult Male Subject, YouTube Platform Branding

### Community 30 - "Week 5 Notebook Patches"
Cohesion: 0.67
Nodes (1): Patch harsh_week5_qwen3vl.ipynb: Stage 3 (TF32/FA2/compile) + Stage 4 (batch rer

### Community 31 - "App Root Layout"
Cohesion: 0.67
Nodes (1): RootLayout()

### Community 32 - "Landing Page"
Cohesion: 0.67
Nodes (1): Home()

### Community 33 - "Hero Component"
Cohesion: 0.67
Nodes (1): Hero()

### Community 34 - "Navigation Bar"
Cohesion: 0.67
Nodes (1): Navbar()

### Community 35 - "Footer Component"
Cohesion: 0.67
Nodes (1): Footer()

### Community 36 - "Tag/Pill Component"
Cohesion: 0.67
Nodes (1): Tag()

### Community 37 - "Button Component"
Cohesion: 0.67
Nodes (1): Button()

### Community 38 - "Fragrance Card Component"
Cohesion: 0.67
Nodes (1): FragranceCard()

### Community 39 - "Submit Button Component"
Cohesion: 0.67
Nodes (1): SubmitButton()

### Community 40 - "T4 Lightweight Config"
Cohesion: 1.0
Nodes (2): Colab T4 Lightweight Dependencies (requirements.colab.t4.txt), Gemma 4 E4B int4 (T4 lightweight enrichment)

### Community 41 - "Tailwind Config"
Cohesion: 1.0
Nodes (1): tailwind.config.ts

### Community 42 - "PostCSS Config"
Cohesion: 1.0
Nodes (1): postcss.config.mjs

### Community 43 - "Next.js Type Defs"
Cohesion: 1.0
Nodes (1): next-env.d.ts

### Community 44 - "ESLint Config"
Cohesion: 1.0
Nodes (1): eslint.config.mjs

### Community 67 - "Pipeline Visual Doc"
Cohesion: 1.0
Nodes (1): PipelineVisual Component

## Knowledge Gaps
- **112 isolated node(s):** `tailwind.config.ts`, `Demo server — serves pre-cached locked responses, no GPU required. Run:  uv run`, `Returns locked responses for every request — no model loading.`, `postcss.config.mjs`, `next-env.d.ts` (+107 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Clustered DB Builder`** (4 nodes): `build_clustered_database()`, `FragranceVibe`, `enrich_dataset.py`, `enrich_dataset.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Notes-to-Vibe Mapping`** (4 nodes): `notes_to_vibe.py`, `calculate_vibe_vector()`, `Takes a list of string notes, looks them up in the dictionary,      and averages`, `notes_to_vibe.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Context Form Component`** (4 nodes): `ContextForm.tsx`, `ContextForm()`, `PillGroup()`, `ContextForm.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Week 5 Notebook Patches`** (3 nodes): `update_week5_nb.py`, `update_week5_nb.py`, `Patch harsh_week5_qwen3vl.ipynb: Stage 3 (TF32/FA2/compile) + Stage 4 (batch rer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `App Root Layout`** (3 nodes): `layout.tsx`, `layout.tsx`, `RootLayout()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Landing Page`** (3 nodes): `page.tsx`, `page.tsx`, `Home()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Hero Component`** (3 nodes): `Hero.tsx`, `Hero()`, `Hero.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Navigation Bar`** (3 nodes): `Navbar.tsx`, `Navbar.tsx`, `Navbar()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Footer Component`** (3 nodes): `Footer.tsx`, `Footer()`, `Footer.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tag/Pill Component`** (3 nodes): `Tag.tsx`, `Tag.tsx`, `Tag()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Button Component`** (3 nodes): `Button()`, `Button.tsx`, `Button.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Fragrance Card Component`** (3 nodes): `FragranceCard.tsx`, `FragranceCard()`, `FragranceCard.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Submit Button Component`** (3 nodes): `SubmitButton.tsx`, `SubmitButton.tsx`, `SubmitButton()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `T4 Lightweight Config`** (2 nodes): `Colab T4 Lightweight Dependencies (requirements.colab.t4.txt)`, `Gemma 4 E4B int4 (T4 lightweight enrichment)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tailwind Config`** (1 nodes): `tailwind.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `PostCSS Config`** (1 nodes): `postcss.config.mjs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Next.js Type Defs`** (1 nodes): `next-env.d.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ESLint Config`** (1 nodes): `eslint.config.mjs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pipeline Visual Doc`** (1 nodes): `PipelineVisual Component`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Settings` connect `Recommendation Engine & Schemas` to `Benchmarking & Evaluation`, `CLI & Evaluation Pipeline`, `API Layer (Next.js + FastAPI)`, `Fragrance Data Enrichment`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Why does `VibeScoreEngine` connect `Recommendation Engine & Schemas` to `Image Scoring Module`, `API Layer (Next.js + FastAPI)`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `EnrichmentSchemaV2` connect `Enrichment Pipeline (Week 2)` to `Recommendation Engine & Schemas`, `CLI & Evaluation Pipeline`, `Fragrance Data Enrichment`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Are the 24 inferred relationships involving `EnrichmentSchemaV2` (e.g. with `EnrichmentClient` and `QwenOutlinesEnrichmentClient`) actually correct?**
  _`EnrichmentSchemaV2` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 26 inferred relationships involving `Settings` (e.g. with `Qwen3VLReranker` and `Qwen3-VL-Reranker-8B pointwise reranker — local GPU, zero API keys.`) actually correct?**
  _`Settings` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `VibeScoreEngine` (e.g. with `Qwen3VLMultimodalEmbedder` and `SigLIP2ImageScorer`) actually correct?**
  _`VibeScoreEngine` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `ImageHeadProbabilities` (e.g. with `VibeScoreEngine` and `4-channel fusion + listwise LLM reranker recommendation engine — zero API keys r`) actually correct?**
  _`ImageHeadProbabilities` has 11 INFERRED edges - model-reasoned connections that need verification._