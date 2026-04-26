import json

nb_path = 'notebooks/harsh_offline_pipeline.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

cell_5_src = """# Stage 5: Corpus Re-embedding (Qwen3-VL-Embedding-8B)
# ONE-TIME: produces embeddings in the same vector space as query-time embedding
# Run AFTER Stage 4 so retrieval_text includes vibe_sentence + enriched fields
import time, torch, gc, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, REPO_DIR)

gc.collect(); torch.cuda.empty_cache()

# TF32 for faster matmuls on Blackwell/Ampere
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load enriched CSV (from Stage 4 output or ENRICHED_CSV on Drive)
_csv = ENRICHED_CSV if os.path.exists(ENRICHED_CSV) else INPUT_CSV
df_embed = pd.read_csv(_csv)
print(f'Embedding {len(df_embed)} rows from: {_csv}')

# Build retrieval_text if missing
if 'retrieval_text' not in df_embed.columns or df_embed['retrieval_text'].isna().all():
    from vibescents.enrich import build_retrieval_text
    df_embed = build_retrieval_text(df_embed)

texts = df_embed['retrieval_text'].fillna(df_embed.get('name', '')).tolist()
print(f'Sample retrieval_text: {texts[0][:120]}')

# Load embedder — bump batch size for corpus throughput
from vibescents.embeddings import Qwen3VLMultimodalEmbedder
from vibescents.settings import Settings

Qwen3VLMultimodalEmbedder._BATCH_SIZE = 64  # throughput-optimised for corpus
_s = Settings(api_key=None)
embedder = Qwen3VLMultimodalEmbedder(settings=_s, load_in_8bit=False)

print(f'VRAM after embedder load: {torch.cuda.memory_allocated()/1e9:.1f} GB')

out_emb = os.path.join(EMBEDDINGS_DIR, 'embeddings.npy')
out_manifest = os.path.join(EMBEDDINGS_DIR, 'manifest.json')
CKPT_DIR = os.path.join(EMBEDDINGS_DIR, 'ckpts')
os.makedirs(CKPT_DIR, exist_ok=True)

# Resume logic
def get_checkpoints():
    files = sorted(glob.glob(os.path.join(CKPT_DIR, "embeddings_ckpt_*.npy")), 
                   key=lambda p: int(Path(p).stem.split("_")[-1]))
    return files

ckpt_files = get_checkpoints()
already_embedded = 0
for f in ckpt_files:
    already_embedded += np.load(f).shape[0]

print(f'Found {len(ckpt_files)} checkpoints. Already embedded: {already_embedded} / {len(texts)}')

texts_to_embed = texts[already_embedded:]

if texts_to_embed:
    print(f'Resuming embedding for remaining {len(texts_to_embed)} rows...')
    t0 = time.perf_counter()
    
    # We chunk the remaining texts so we can save periodic checkpoints
    CHUNK_SIZE = Qwen3VLMultimodalEmbedder._BATCH_SIZE * 50 # Checkpoint every 3200 rows
    
    for i in range(0, len(texts_to_embed), CHUNK_SIZE):
        chunk = texts_to_embed[i:i+CHUNK_SIZE]
        print(f"  -> Processing chunk {i//CHUNK_SIZE + 1} ({len(chunk)} rows)...")
        
        chunk_emb = embedder.embed_multimodal_documents(chunk)
        
        # Save checkpoint
        ckpt_idx = len(get_checkpoints())
        ckpt_path = os.path.join(CKPT_DIR, f"embeddings_ckpt_{ckpt_idx}.npy")
        np.save(ckpt_path, chunk_emb)
        print(f"  -> Saved checkpoint: {ckpt_path}")
        
    elapsed = time.perf_counter() - t0
    print(f'Embedded {len(texts_to_embed)} remaining rows in {elapsed:.0f}s  ({len(texts_to_embed)/elapsed:.0f} rows/s)')

# Load all checkpoints to form the final matrix
all_ckpts = get_checkpoints()
final_parts = [np.load(f) for f in all_ckpts]

if final_parts:
    embeddings = np.vstack(final_parts)
else:
    embeddings = np.empty((0, 0), dtype=np.float32)

print(f'Embeddings shape: {embeddings.shape}  dtype: {embeddings.dtype}')
assert embeddings.shape[0] == len(texts), f"Expected {len(texts)} embeddings, got {embeddings.shape[0]}"

# Verify L2 normalization
norms = np.linalg.norm(embeddings, axis=1)
print(f'Norms — mean: {norms.mean():.4f}  std: {norms.std():.5f}  min: {norms.min():.4f}')
print(f'NaN count: {np.isnan(embeddings).sum()}')
assert np.isnan(embeddings).sum() == 0, 'NaN values in embeddings!'

# Save
np.save(out_emb, embeddings)
manifest = {
    'model': 'Qwen/Qwen3-VL-Embedding-8B',
    'created': __import__('datetime').datetime.utcnow().isoformat(),
    'n_rows': int(embeddings.shape[0]),
    'dim': int(embeddings.shape[1]),
    'dtype': str(embeddings.dtype),
    'source_csv': _csv,
}
with open(out_manifest, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f'\\nSaved embeddings: {out_emb}')
print(f'Saved manifest:   {out_manifest}')
print(json.dumps(manifest, indent=2))
"""

nb['cells'][5]['source'] = [line + '\n' for line in cell_5_src.split('\n')][:-1]

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
