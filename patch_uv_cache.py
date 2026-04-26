import json

# --- 1. Patch harsh_offline_pipeline.ipynb ---
with open('notebooks/harsh_offline_pipeline.ipynb', 'r') as f:
    nb1 = json.load(f)

source1 = nb1['cells'][1]['source']
new_source1 = []
for line in source1:
    if "print(\"Installing 'uv' for lightning-fast dependency resolution...\")" in line:
        new_source1.append("\n# Enable persistent package caching on Google Drive\n")
        new_source1.append("UV_CACHE = '/content/drive/MyDrive/uv_cache'\n")
        new_source1.append("os.makedirs(UV_CACHE, exist_ok=True)\n")
        new_source1.append("os.environ['UV_CACHE_DIR'] = UV_CACHE\n\n")
    new_source1.append(line)

nb1['cells'][1]['source'] = new_source1
with open('notebooks/harsh_offline_pipeline.ipynb', 'w') as f:
    json.dump(nb1, f, indent=1)


# --- 2. Patch harsh_week5_qwen3vl.ipynb ---
with open('notebooks/harsh_week5_qwen3vl.ipynb', 'r') as f:
    nb2 = json.load(f)

source2 = nb2['cells'][2]['source']
new_source2 = []
for line in source2:
    # Add vllm and bitsandbytes to _DEPS
    if "'tqdm', 'pandas', 'numpy'," in line:
        line = line.replace("'tqdm'", "'tqdm', 'vllm', 'bitsandbytes'")
    
    # Add UV_CACHE logic before uv pip install
    if "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'uv']" in line:
        new_source2.append("\n# Enable persistent package caching on Google Drive\n")
        new_source2.append("UV_CACHE = '/content/drive/MyDrive/uv_cache'\n")
        new_source2.append("import os; os.makedirs(UV_CACHE, exist_ok=True)\n")
        new_source2.append("os.environ['UV_CACHE_DIR'] = UV_CACHE\n\n")
        
    new_source2.append(line)

nb2['cells'][2]['source'] = new_source2
with open('notebooks/harsh_week5_qwen3vl.ipynb', 'w') as f:
    json.dump(nb2, f, indent=1)

print("Persistent UV caching and vLLM dependencies added to both notebooks.")
