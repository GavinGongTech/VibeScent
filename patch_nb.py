import json
import re

nb_path = 'notebooks/harsh_offline_pipeline.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

# Cell 1: Environment Setup + Enhanced Debugging + Git Clone Fix
cell_1_src = """# Stage 1: Environment Setup
# !! Run this first, then restart runtime, then run all remaining cells !!
import subprocess, sys, traceback

# --- Enhanced Debugging ---
def custom_excepthook(exc_type, exc_value, exc_traceback):
    print("\\n" + "="*60)
    print("!!! AN ERROR OCCURRED !!!")
    print("="*60)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("="*60)
    print("!!! CHECK THE TRACEBACK ABOVE TO FIND THE EXACT LINE OF CODE !!!\\n")
sys.excepthook = custom_excepthook

_pkgs = [
    'google-genai',
    'pandas',
    'numpy',
    'transformers>=4.57.0',
    'torch',
    'accelerate',
    'qwen-vl-utils>=0.0.14',
    'outlines',
    'json-repair',
    'tqdm',
]
print("Installing dependencies...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + _pkgs)

# Clone or update repo
import os
from google.colab import userdata

print("Checking for GitHub Token in Colab Secrets...")
try:
    github_token = userdata.get('GITHUB_TOKEN')
    REPO_URL = f'https://{github_token}@github.com/GavinReid82/vibescent.git'
    print("Using provided GITHUB_TOKEN for clone.")
except userdata.SecretNotFoundError:
    print("GITHUB_TOKEN not found in Colab Secrets. Attempting public clone...")
    REPO_URL = 'https://github.com/GavinReid82/vibescent.git'

REPO_DIR = '/content/vibescent'
try:
    if not os.path.exists(REPO_DIR):
        print(f"Cloning repo into {REPO_DIR}...")
        subprocess.check_call(['git', 'clone', REPO_URL, REPO_DIR])
    else:
        print("Pulling latest changes...")
        subprocess.check_call(['git', '-C', REPO_DIR, 'pull'])
except subprocess.CalledProcessError as e:
    print(f"\\n[CRITICAL ERROR] Git command failed with exit code {e.returncode}")
    print("If the repo is private, add your 'GITHUB_TOKEN' to Google Colab Secrets (the key icon on the left sidebar).")
    raise e

print("Installing repo in editable mode...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-e', REPO_DIR])

print('\\nEnvironment ready. Restart runtime now, then continue from Stage 2.')
"""
nb['cells'][1]['source'] = [line + '\n' for line in cell_1_src.split('\n')][:-1]

# Cell 4: Parallelism and Resumability
cell_4_src = """# Stage 4: LLM Enrichment (Qwen3-8B — local GPU, zero API keys)
# Loads Qwen3-8B in bfloat16 (~16 GB), uses outlines to constrain-decode
# into EnrichmentSchemaV2 JSON — guaranteed valid output.
# Alternatives: model_name = "google/gemma-3-12b-it" or "google/gemma-3-27b-it"
import sys, os, json, threading, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MAX_WORKERS = 12  # parallel workers for vllm batching

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

def process_row(idx):
    try:
        row = df_work.loc[idx]
        prompt = _build_prompt(row)
        record = client.generate(prompt)
        return idx, {col: _serialize_value(getattr(record, col)) for col in ENRICHMENT_COLUMNS}, None
    except Exception as e:
        error_msg = f"Error on row {idx}: {str(e)}\\n{traceback.format_exc()}"
        return idx, None, error_msg

# Using ThreadPoolExecutor for concurrent batching via vLLM
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_row, idx): idx for idx in _need_idx}
    
    with tqdm(total=len(futures), desc="Enriching (Parallel)") as pbar:
        for future in as_completed(futures):
            idx, data, error = future.result()
            if error:
                _failed += 1
                _failures.append({"idx": idx, "error": error})
                print(f"\\n[ERROR] {error}")
            else:
                for col, val in data.items():
                    df_work.at[idx, col] = val
                _completed += 1
            
            pbar.update(1)
            pbar.set_postfix(ok=_completed, fail=_failed)

            if (_completed + _failed) % CHECKPOINT_EVERY == 0:
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
    
print("Notebook patched successfully!")
