import json

# --- 1. Patch harsh_offline_pipeline.ipynb ---
with open('notebooks/harsh_offline_pipeline.ipynb', 'r') as f:
    nb1 = json.load(f)

cell_1_src = """# Stage 1: Environment Setup
# !! Run this first, then restart runtime, then run all remaining cells !!
import subprocess, sys, traceback, os

# --- Enhanced Debugging (IPython / Colab compatible) ---
import traceback
import IPython

def custom_exc(shell, etype, evalue, tb, tb_offset=None):
    print("\\n" + "="*60)
    print("!!! AN ERROR OCCURRED !!!")
    print("="*60)
    traceback.print_exception(etype, evalue, tb)
    print("="*60)
    print("!!! CHECK THE TRACEBACK ABOVE TO FIND THE EXACT LINE OF CODE !!!\\n")

_ipython = IPython.get_ipython()
if _ipython:
    _ipython.set_custom_exc((Exception,), custom_exc)
    _ipython.magic("xmode Verbose")  # Further enhance built-in tracebacks

print("Step 1: Mounting Google Drive...")
try:
    from google.colab import drive
    # force_remount=True helps if a previous mount crashed
    drive.mount('/content/drive', force_remount=True)
except Exception as e:
    print("\\n[ERROR] Google Drive mount failed.")
    print("Cause:", str(e))
    print("\\nACTION REQUIRED:")
    print("1. Ensure you are logged into the same Google account as this Colab.")
    print("2. Click the 'Connect to Google Drive' button in the popup.")
    print("3. If no popup appears, refresh the page or restart the runtime.")
    raise RuntimeError("Google Drive must be mounted to load the project ZIP and Cache.")

REPO_DIR = '/content/vibescent'
# Check both possible mount paths (sometimes Colab uses 'My Drive' vs 'MyDrive')
ZIP_CANDS = [
    '/content/drive/MyDrive/vibescent.zip',
    '/content/drive/My Drive/vibescent.zip',
    '/content/vibescent.zip' # Fallback for direct manual upload to file browser
]

ZIP_PATH = None
for cand in ZIP_CANDS:
    if os.path.exists(cand):
        ZIP_PATH = cand
        break

if not os.path.exists(REPO_DIR):
    if ZIP_PATH:
        print(f"Step 2: Extracting {ZIP_PATH} to /content/...")
        subprocess.check_call(['unzip', '-q', ZIP_PATH, '-d', '/content/'])
        
        # Auto-renaming logic
        extracted_dirs = [d for d in os.listdir('/content') if os.path.isdir(os.path.join('/content', d)) and 'vibescent' in d.lower()]
        if extracted_dirs and not os.path.exists(REPO_DIR):
            found_dir = os.path.join('/content', extracted_dirs[0])
            if found_dir != REPO_DIR:
                os.rename(found_dir, REPO_DIR)
        
        if not os.path.exists(REPO_DIR):
            print(f"\\n[CRITICAL ERROR] Extraction completed but {REPO_DIR} was not created.")
            raise FileNotFoundError(f"Expected project at {REPO_DIR}")
        print(f"Successfully extracted to {REPO_DIR}")
    else:
        print("\\n[CRITICAL ERROR] vibescent.zip NOT FOUND.")
        print("\\nACTION REQUIRED:")
        print("1. Run './zip_project.sh' on your local computer.")
        print("2. Upload the resulting 'vibescent.zip' to the root of your Google Drive.")
        print("3. Alternatively, upload 'vibescent.zip' directly to the Colab file sidebar (left).")
        raise FileNotFoundError("Could not find vibescent.zip in Drive or local storage.")
else:
    print(f"Project folder already exists at {REPO_DIR}. Skipping extraction.")

os.chdir(REPO_DIR)

# Enable persistent package caching on Google Drive
UV_CACHE = '/content/drive/MyDrive/uv_cache'
os.makedirs(UV_CACHE, exist_ok=True)
os.environ['UV_CACHE_DIR'] = UV_CACHE

print("Step 3: Installing 'uv' for lightning-fast dependency resolution...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'uv'], check=True)

_pkgs = [
    'outlines==1.2.12',
    'outlines-core==0.2.14',
    'jedi',
    'google-genai',
    'pandas',
    'numpy',
    'transformers>=4.57.0',
    'torch',
    'accelerate',
    'vllm',
    'bitsandbytes',
    'qwen-vl-utils>=0.0.14',
    'json-repair',
    'tqdm',
    'hf_transfer',
]
print("Step 4: Installing project and dependencies using 'uv'...")
subprocess.run(['uv', 'pip', 'install', '--system', '-q', '-e', REPO_DIR] + _pkgs, check=True)

print('\\nEnvironment ready. Restart runtime now, then continue from Stage 2.')
"""
nb1['cells'][1]['source'] = [line + '\n' for line in cell_1_src.split('\n')][:-1]

with open('notebooks/harsh_offline_pipeline.ipynb', 'w') as f:
    json.dump(nb1, f, indent=1)


# --- 2. Patch harsh_week5_qwen3vl.ipynb ---
with open('notebooks/harsh_week5_qwen3vl.ipynb', 'r') as f:
    nb2 = json.load(f)

# The week 5 setup is in cell index 2
cell_2_src_w5 = """# Stage 1: Setup
import os, sys, subprocess, threading, time, traceback
_t0 = time.perf_counter()
PIPELINE_VERSION = 'w5v1'
_IS_KAGGLE = os.path.exists('/kaggle/working') or os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
REPO_DIR  = '/kaggle/working/vibescent' if _IS_KAGGLE else '/content/vibescent'
FASTAPI_PORT = 8000; FASTAPI_HOST = '127.0.0.1'
_BASE_URL = f'http://{FASTAPI_HOST}:{FASTAPI_PORT}'

# --- Enhanced Debugging (IPython / Colab compatible) ---
import traceback
import IPython

def custom_exc(shell, etype, evalue, tb, tb_offset=None):
    print("\\n" + "="*60)
    print("!!! AN ERROR OCCURRED !!!")
    print("="*60)
    traceback.print_exception(etype, evalue, tb)
    print("="*60)
    print("!!! CHECK THE TRACEBACK ABOVE TO FIND THE EXACT LINE OF CODE !!!\\n")

_ipython = IPython.get_ipython()
if _ipython:
    _ipython.set_custom_exc((Exception,), custom_exc)
    _ipython.magic("xmode Verbose")

print("Step 1: Mounting Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
except Exception as e:
    print(f"Drive mount failure: {e}")
    raise RuntimeError("Google Drive must be mounted.")

# ZIP extraction logic
ZIP_CANDS = [
    '/content/drive/MyDrive/vibescent.zip',
    '/content/drive/My Drive/vibescent.zip',
    '/content/vibescent.zip'
]
ZIP_PATH = None
for cand in ZIP_CANDS:
    if os.path.exists(cand):
        ZIP_PATH = cand
        break

if not os.path.isdir(REPO_DIR):
    if ZIP_PATH:
        print(f"Step 2: Extracting {ZIP_PATH} to /content/...")
        subprocess.check_call(['unzip', '-q', ZIP_PATH, '-d', '/content/'])
        extracted_dirs = [d for d in os.listdir('/content') if os.path.isdir(os.path.join('/content', d)) and 'vibescent' in d.lower()]
        if extracted_dirs and not os.path.exists(REPO_DIR):
            found_dir = os.path.join('/content', extracted_dirs[0])
            if found_dir != REPO_DIR:
                os.rename(found_dir, REPO_DIR)
    else:
        print("\\n[CRITICAL ERROR] vibescent.zip NOT FOUND.")
        raise FileNotFoundError("Could not find vibescent.zip in Drive or local storage.")
else:
    print('Project folder already exists. Skipping extraction.')

os.chdir(REPO_DIR)

# UV Cache setup
UV_CACHE = '/content/drive/MyDrive/uv_cache'
import os; os.makedirs(UV_CACHE, exist_ok=True)
os.environ['UV_CACHE_DIR'] = UV_CACHE

print("Step 3: Installing 'uv'...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'uv'],
               check=True, capture_output=True)

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
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    os.makedirs(HF_CACHE, exist_ok=True)
_dt = threading.Thread(target=_mount, daemon=False); _dt.start()
bar.update(1)

# Thread 2: Secrets
_hf_token = None
def _secrets():
    global _hf_token
    try:
        from google.colab import userdata as _ud
        _hf_token = _ud.get('HF_TOKEN') or ''
    except Exception:
        import os as _os
        _hf_token = _os.environ.get('HF_TOKEN', '')
_st = threading.Thread(target=_secrets, daemon=False); _st.start()
_dt.join(); _st.join()
bar.update(1)

if _hf_token:
    os.environ['HF_TOKEN'] = _hf_token
    os.environ['HUGGING_FACE_HUB_TOKEN'] = _hf_token
    print('HF_TOKEN loaded')

# Package install
subprocess.run(['uv', 'pip', 'install', '--system', '-q', '-e', REPO_DIR],
               check=True, capture_output=True)
bar.update(1)

_DEPS = [
    'outlines==1.2.12', 'outlines-core==0.2.14', 'jedi',
    'torch>=2.4,<3.0', 'transformers>=4.57,<5.0', 'accelerate>=1.3,<2.0',
    'vllm', 'bitsandbytes',
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

bar.update(1); bar.close()
print(f'Setup {time.perf_counter()-_t0:.1f}s | DRIVE={DRIVE_BASE} | GPU={GPU_TIER}')
"""
nb2['cells'][2]['source'] = [line + '\n' for line in cell_2_src_w5.split('\n')][:-1]

with open('notebooks/harsh_week5_qwen3vl.ipynb', 'w') as f:
    json.dump(nb2, f, indent=1)

print("Robust Drive mount and ZIP candidates patched into both notebooks.")
