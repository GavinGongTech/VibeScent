import json

nb_path = 'notebooks/harsh_week5_qwen3vl.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

# Cell 1 (Stage 0) patch
new_cell_1 = """# ============================================================
# STAGE 0 — ONE-TIME corpus re-embedding with Qwen3-VL-Embedding-8B
# Run this cell ONCE. It saves qwen3vl_corpus/embeddings.npy to Drive.
# Skip on subsequent runs — Stage 2 loads the saved file.
# Runtime: ~8 min on A100 80GB (35k rows, batch 64)
# ============================================================
import os, sys, subprocess, time, traceback, glob

# --- Enhanced Debugging ---
def custom_excepthook(exc_type, exc_value, exc_traceback):
    print("\\n" + "="*60)
    print("!!! AN ERROR OCCURRED !!!")
    print("="*60)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("="*60)
    print("!!! CHECK THE TRACEBACK ABOVE TO FIND THE EXACT LINE OF CODE !!!\\n")
if not hasattr(sys, 'excepthook_set'):
    sys.excepthook = custom_excepthook
    sys.excepthook_set = True

REPO_DIR = '/content/vibescent'
ZIP_PATH = '/content/drive/MyDrive/vibescent.zip'

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

if not os.path.exists(REPO_DIR):
    if os.path.exists(ZIP_PATH):
        print(f"Extracting {ZIP_PATH} to /content/...")
        subprocess.check_call(['unzip', '-q', ZIP_PATH, '-d', '/content/'])
        if os.path.exists('/content/vibescent-main') and not os.path.exists(REPO_DIR):
            os.rename('/content/vibescent-main', REPO_DIR)
        print(f"Successfully extracted to {REPO_DIR}")
    else:
        print(f"\\n[CRITICAL ERROR] Could not find {ZIP_PATH}.")
        print("Please upload 'vibescent.zip' to the root of your Google Drive.")
        raise FileNotFoundError(ZIP_PATH)
else:
    print(f"Project folder already exists at {REPO_DIR}. Skipping extraction.")

os.chdir(REPO_DIR)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'uv'],
               check=True, capture_output=True)

_deps = ['torch>=2.4,<3.0', 'transformers>=4.57,<5.0', 'accelerate>=1.3,<2.0',
         'qwen-vl-utils>=0.0.14', 'Pillow>=10.0', 'numpy', 'pandas', 'tqdm']
subprocess.run(['uv', 'pip', 'install', '--system', '-q', '-e', REPO_DIR] + _deps,
               check=True, capture_output=True)

DRIVE_BASE = '/content/drive/MyDrive/vibescent'
OUT_DIR    = f'{DRIVE_BASE}/qwen3vl_corpus'
os.makedirs(OUT_DIR, exist_ok=True)

OUT_EMB  = f'{OUT_DIR}/embeddings.npy'
OUT_META = f'{OUT_DIR}/metadata_path.txt'

if os.path.exists(OUT_EMB):
    import numpy as np
    _existing = np.load(OUT_EMB)
    print(f'[SKIP] Corpus already embedded: {OUT_EMB}  shape={_existing.shape}')
    print('Stage 0 complete — proceed to Stage 1.')
else:
    import torch, numpy as np, pandas as pd
    from tqdm.auto import tqdm
    from pathlib import Path

    # ── Load fragrance DataFrame ──────────────────────────────────────────────
    _csv_cands = [
        f'{DRIVE_BASE}/vibescent_enriched_2k_v2.csv',
        f'{DRIVE_BASE}/vibescent_enriched_500_v2.csv',
        f'{REPO_DIR}/data/processed/vibescent_unified.csv',
        f'{REPO_DIR}/data/vibescent_enriched.csv',
    ]
    df = None
    for _p in _csv_cands:
        if os.path.exists(_p):
            df = pd.read_csv(_p)
            print(f'DataFrame: {_p}  shape={df.shape}')
            break
    if df is None:
        raise FileNotFoundError('No fragrance DataFrame found. Run Week 2 first.')

    # Build embedding_text for each row: top_notes | middle_notes | base_notes | vibe_sentence
    def _build_text(row):
        parts = []
        name  = str(row.get('name', '') or '')
        brand = str(row.get('brand', '') or '')
        if name:  parts.append(name)
        if brand: parts.append(brand)
        for col in ('top_notes', 'middle_notes', 'base_notes', 'main_accords', 'vibe_sentence'):
            v = str(row.get(col, '') or '')
            if v and v.lower() not in ('nan', 'none', ''):
                parts.append(v)
        return ' | '.join(parts)[:512]

    texts = [_build_text(r) for _, r in df.iterrows()]
    print(f'Built {len(texts)} embedding texts. Sample: {texts[0][:100]}')

    # ── Load Qwen3-VL-Embedding-8B ────────────────────────────────────────────
    MODEL_NAME = 'Qwen/Qwen3-VL-Embedding-8B'
    print(f'Loading {MODEL_NAME} in bfloat16 ...')

    # Use the vibescents embedder directly
    sys.path.insert(0, REPO_DIR)
    from vibescents.embeddings import Qwen3VLMultimodalEmbedder
    from vibescents.settings import Settings

    s = Settings()  # no API key needed for local model
    # Bump batch size for large GPU
    Qwen3VLMultimodalEmbedder._BATCH_SIZE = 64
    embedder = Qwen3VLMultimodalEmbedder(settings=s, load_in_8bit=False)
    print('Model loaded. Starting corpus embedding...')

    # RESUMABLE EMBEDDING LOGIC
    CKPT_DIR = os.path.join(OUT_DIR, 'ckpts')
    os.makedirs(CKPT_DIR, exist_ok=True)

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
        
        CHUNK_SIZE = Qwen3VLMultimodalEmbedder._BATCH_SIZE * 50
        with tqdm(total=len(texts_to_embed), desc="Embedding (Outer Chunks)") as outer_pbar:
            for i in range(0, len(texts_to_embed), CHUNK_SIZE):
                chunk = texts_to_embed[i:i+CHUNK_SIZE]
                chunk_emb = embedder.embed_multimodal_documents(chunk)
                
                ckpt_idx = len(get_checkpoints())
                ckpt_path = os.path.join(CKPT_DIR, f"embeddings_ckpt_{ckpt_idx}.npy")
                np.save(ckpt_path, chunk_emb)
                
                outer_pbar.update(len(chunk))
            
        elapsed = time.perf_counter() - t0
        print(f'Embedded {len(texts_to_embed)} remaining rows in {elapsed:.0f}s')

    all_ckpts = get_checkpoints()
    final_parts = [np.load(f) for f in all_ckpts]

    if final_parts:
        corpus_emb = np.vstack(final_parts)
    else:
        corpus_emb = np.empty((0, 0), dtype=np.float32)

    np.save(OUT_EMB, corpus_emb.astype(np.float32))
    with open(OUT_META, 'w') as _mf:
        _mf.write(str(list(_csv_cands)[0]))

    # Also save to local repo artifacts for reference
    _local_out = f'{REPO_DIR}/artifacts/qwen3vl_corpus'
    os.makedirs(_local_out, exist_ok=True)
    np.save(f'{_local_out}/embeddings.npy', corpus_emb.astype(np.float32))

    print(f'Saved to: {OUT_EMB}')
    print(f'Shape: {corpus_emb.shape} | dtype: {corpus_emb.dtype}')
    print('Stage 0 COMPLETE. Proceed to Stage 1.')
"""
nb['cells'][1]['source'] = [line + '\n' for line in new_cell_1.split('\n')][:-1]


# Cell 2 (Stage 1) patch
new_cell_2 = """# Stage 1: Setup
import os, sys, subprocess, threading, time, traceback
_t0 = time.perf_counter()
PIPELINE_VERSION = 'w5v1'
_IS_KAGGLE = os.path.exists('/kaggle/working') or os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
REPO_DIR  = '/kaggle/working/vibescent' if _IS_KAGGLE else '/content/vibescent'
FASTAPI_PORT = 8000; FASTAPI_HOST = '127.0.0.1'
_BASE_URL = f'http://{FASTAPI_HOST}:{FASTAPI_PORT}'

# --- Enhanced Debugging ---
def custom_excepthook(exc_type, exc_value, exc_traceback):
    print("\\n" + "="*60)
    print("!!! AN ERROR OCCURRED !!!")
    print("="*60)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("="*60)
    print("!!! CHECK THE TRACEBACK ABOVE TO FIND THE EXACT LINE OF CODE !!!\\n")
if not hasattr(sys, 'excepthook_set'):
    sys.excepthook = custom_excepthook
    sys.excepthook_set = True

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'uv'],
               check=True, capture_output=True)

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except Exception as e:
    print(f"Drive mount warning: {e}")

# ZIP extraction logic
ZIP_PATH = '/content/drive/MyDrive/vibescent.zip'
if not os.path.isdir(REPO_DIR):
    if os.path.exists(ZIP_PATH):
        print(f"Extracting {ZIP_PATH} to /content/...")
        subprocess.check_call(['unzip', '-q', ZIP_PATH, '-d', '/content/'])
        if os.path.exists('/content/vibescent-main') and not os.path.exists(REPO_DIR):
            os.rename('/content/vibescent-main', REPO_DIR)
    else:
        print(f"\\n[CRITICAL ERROR] Could not find {ZIP_PATH}.")
        print("Please run the 'zip_project.sh' script locally to create 'vibescent.zip',")
        print("then upload it to the root of your Google Drive before running this cell.")
        raise FileNotFoundError(ZIP_PATH)
else:
    print('Project folder already exists. Skipping extraction.')

os.chdir(REPO_DIR)

from tqdm.auto import tqdm
bar = tqdm(['Drive', 'Secrets', 'Pkg', 'Deps', 'GPU'], desc='Stage 1', ncols=100, unit='step')

# Thread 1: Drive mount + paths
_drive_exc = None
def _mount():
    global _drive_exc, DRIVE_BASE, W4_ARTIFACTS, W5_ARTIFACTS, HF_CACHE, CLOUDFLARED_BIN
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        DRIVE_BASE = '/content/drive/MyDrive/vibescent'
        HF_CACHE   = '/content/drive/MyDrive/hf_cache'
    except Exception as e:
        _drive_exc = e
        DRIVE_BASE = f'{REPO_DIR}/artifacts'
        HF_CACHE   = '/tmp/hf_cache'
    W4_ARTIFACTS = f'{DRIVE_BASE}/week4'
    W5_ARTIFACTS = f'{DRIVE_BASE}/week5'
    CLOUDFLARED_BIN = '/usr/local/bin/cloudflared'
    os.makedirs(W4_ARTIFACTS, exist_ok=True)
    os.makedirs(W5_ARTIFACTS, exist_ok=True)
    os.environ['HF_HOME'] = HF_CACHE
    os.makedirs(HF_CACHE, exist_ok=True)
_dt = threading.Thread(target=_mount, daemon=False); _dt.start()
bar.update(1)

# Thread 2: Secrets
_hf_token = None
def _secrets():
    global _hf_token
    def _g(n):
        try:
            from google.colab import userdata; return userdata.get(n)
        except: return os.environ.get(n)
    _hf_token = _g('HF_TOKEN') or _g('HUGGINGFACE_TOKEN')
_st = threading.Thread(target=_secrets, daemon=False); _st.start()
_dt.join(); _st.join()
bar.update(1)

if _hf_token:
    os.environ['HF_TOKEN'] = _hf_token
    os.environ['HUGGING_FACE_HUB_TOKEN'] = _hf_token
    print('HF_TOKEN loaded')
else:
    print('WARNING: HF_TOKEN not set — model download may fail for gated repos')

# Package install
subprocess.run(['uv', 'pip', 'install', '--system', '-q', '-e', REPO_DIR],
               check=True, capture_output=True)
bar.update(1)

_DEPS = [
    'torch>=2.4,<3.0', 'transformers>=4.57,<5.0', 'accelerate>=1.3,<2.0',
    'qwen-vl-utils>=0.0.14', 'fastapi>=0.115,<1.0', 'uvicorn[standard]>=0.34,<1.0',
    'httpx>=0.27,<1.0', 'pyngrok>=7.2,<8.0', 'nest-asyncio>=1.6,<2.0',
    'Pillow>=10.0,<12.0', 'tqdm', 'pandas', 'numpy',
]
subprocess.run(['uv', 'pip', 'install', '--system', '-q'] + _DEPS,
               check=True, capture_output=True)
bar.update(1)

# Cloudflared
if not os.path.isfile(CLOUDFLARED_BIN):
    subprocess.run(['wget', '-q',
        'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64',
        '-O', CLOUDFLARED_BIN], capture_output=True)
    subprocess.run(['chmod', '+x', CLOUDFLARED_BIN], capture_output=True)

import torch
GPU_TIER = 'CPU'; DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    _v = torch.cuda.get_device_properties(0).total_memory / 1e9
    GPU_TIER = 'BLACKWELL' if _v >= 75 else ('A100' if _v >= 35 else ('L4' if _v >= 20 else 'T4'))
    print(f'GPU: {torch.cuda.get_device_name(0)}  VRAM={_v:.1f}GB  tier={GPU_TIER}')
else:
    print('WARNING: No CUDA GPU detected')

bar.update(1); bar.close()
if _drive_exc:
    print(f'WARNING: Drive mount failed ({_drive_exc}). Using local fallback.')
print(f'Setup {time.perf_counter()-_t0:.1f}s | DRIVE={DRIVE_BASE} | GPU={GPU_TIER}')
"""
nb['cells'][2]['source'] = [line + '\n' for line in new_cell_2.split('\n')][:-1]

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
