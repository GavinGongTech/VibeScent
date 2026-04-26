import json
import re

nb_path = 'notebooks/harsh_offline_pipeline.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

cell_4_src = """# Stage 4: LLM Enrichment (Qwen3-8B — local GPU, zero API keys)
# Loads Qwen3-8B in bfloat16 (~16 GB), uses outlines to constrain-decode
# into EnrichmentSchemaV2 JSON — guaranteed valid output.
# Alternatives: model_name = "google/gemma-3-12b-it" or "google/gemma-3-27b-it"
import sys, os, json, traceback
sys.path.insert(0, REPO_DIR)
import torch, gc
import pandas as pd
from tqdm.auto import tqdm

from vibescents.enrich import (
    QwenOutlinesEnrichmentClient, build_retrieval_text,
    ENRICHMENT_COLUMNS, _build_prompt, _serialize_value
)

ENRICHMENT_MODEL = "Qwen/Qwen3-8B"  # change to any HF instruct model
CHECKPOINT_EVERY = 100
BATCH_SIZE = 16  # Batch size for vLLM generation

gc.collect()
torch.cuda.empty_cache()

print(f"Loading enrichment model: {ENRICHMENT_MODEL}")
print(f"VRAM before: {torch.cuda.memory_allocated()/1e9:.1f} GB")

client = QwenOutlinesEnrichmentClient(model_name=ENRICHMENT_MODEL)

print(f"VRAM after model load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# Identify rows needing enrichment (skip already-filled rows on resume)
_need_mask = (
    df_work["vibe_sentence"].isna()
    if "vibe_sentence" in df_work.columns
    else pd.Series([True] * len(df_work), index=df_work.index)
)
_need_idx = df_work[_need_mask].index.tolist()
print(f"Rows needing enrichment: {len(_need_idx)} / {len(df_work)}")

for col in ENRICHMENT_COLUMNS:
    if col not in df_work.columns:
        df_work[col] = None

_completed, _failed = 0, 0
_failures = []

with tqdm(total=len(_need_idx), desc="Enriching (Batched)") as pbar:
    for i in range(0, len(_need_idx), BATCH_SIZE):
        batch_idx = _need_idx[i:i+BATCH_SIZE]
        batch_prompts = [_build_prompt(df_work.loc[idx]) for idx in batch_idx]
        
        try:
            records = client.generate_batch(batch_prompts)
            for idx, record in zip(batch_idx, records):
                if record is None:
                    _failed += 1
                    _failures.append({"idx": idx, "error": "Failed to parse generation output into EnrichmentSchemaV2."})
                else:
                    for col in ENRICHMENT_COLUMNS:
                        df_work.at[idx, col] = _serialize_value(getattr(record, col))
                    _completed += 1
        except Exception as e:
            error_msg = f"Batch failed: {str(e)}\\n{traceback.format_exc()}"
            print(f"\\n[ERROR] {error_msg}")
            for idx in batch_idx:
                _failed += 1
                _failures.append({"idx": idx, "error": error_msg})
                
        pbar.update(len(batch_idx))
        pbar.set_postfix(ok=_completed, fail=_failed)

        if (_completed + _failed) >= CHECKPOINT_EVERY and (_completed + _failed) % CHECKPOINT_EVERY < BATCH_SIZE:
            df_work.to_csv(CHECKPOINT_CSV, index=False)

# Final checkpoint + retrieval_text
df_work.to_csv(CHECKPOINT_CSV, index=False)
df_work = build_retrieval_text(df_work)
df_work.to_csv(ENRICHED_CSV, index=False)

if _failures:
    with open(FAILURES_LOG, "w") as fh:
        for r in _failures:
            fh.write(json.dumps(r) + "\\n")

print(f"\\nEnrichment done: {_completed} ok, {_failed} failed")
print(f"Saved: {ENRICHED_CSV}")
print("\\nSample vibe_sentences:")
for _, row in df_work[df_work["vibe_sentence"].notna()].head(3).iterrows():
    print(f"  {row['name']}: {str(row['vibe_sentence'])[:100]}")
"""

nb['cells'][4]['source'] = [line + '\n' for line in cell_4_src.split('\n')][:-1]

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
