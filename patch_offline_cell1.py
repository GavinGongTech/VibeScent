import json

nb_path = 'notebooks/harsh_offline_pipeline.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

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

print("Mounting Google Drive to load the project zip...")
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as e:
    print(f"Drive mount warning: {e}")

REPO_DIR = '/content/vibescent'
ZIP_PATH = '/content/drive/MyDrive/vibescent.zip'

if not os.path.exists(REPO_DIR):
    if os.path.exists(ZIP_PATH):
        print(f"Extracting {ZIP_PATH} to /content/...")
        subprocess.check_call(['unzip', '-q', ZIP_PATH, '-d', '/content/'])
        
        # If it extracted as vibescent-main or similar, rename it
        if os.path.exists('/content/vibescent-main') and not os.path.exists(REPO_DIR):
            os.rename('/content/vibescent-main', REPO_DIR)
        elif not os.path.exists(REPO_DIR):
            print(f"\\n[CRITICAL ERROR] Extraction completed but {REPO_DIR} was not created.")
            print("Make sure the zip file contains a 'vibescent' folder at its root.")
            raise FileNotFoundError(REPO_DIR)
            
        print(f"Successfully extracted to {REPO_DIR}")
    else:
        print(f"\\n[CRITICAL ERROR] Could not find {ZIP_PATH}.")
        print("Please run the 'zip_project.sh' script locally to create 'vibescent.zip',")
        print("then upload it to the root of your Google Drive before running this cell.")
        raise FileNotFoundError(ZIP_PATH)
else:
    print(f"Project folder already exists at {REPO_DIR}. Skipping extraction.")

os.chdir(REPO_DIR)

print("Installing 'uv' for lightning-fast dependency resolution...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'uv'], check=True)

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
print("Installing project and dependencies using 'uv'...")
subprocess.run(['uv', 'pip', 'install', '--system', '-q', '-e', REPO_DIR] + _pkgs, check=True)

print('\\nEnvironment ready. Restart runtime now, then continue from Stage 2.')
"""

nb['cells'][1]['source'] = [line + '\n' for line in cell_1_src.split('\n')][:-1]

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
print("Patched Cell 1 with uv installation.")
