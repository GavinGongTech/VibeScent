import json

def add_hf_transfer(nb_path, cell_idx):
    with open(nb_path, 'r') as f:
        nb = json.load(f)
        
    source = nb['cells'][cell_idx]['source']
    
    # 1. Add hf_transfer to the pkgs list
    for i, line in enumerate(source):
        if "'tqdm'," in line:
            source.insert(i+1, "    'hf_transfer',\n")
            break
            
    # 2. Add os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1' after HF_HOME
    for i, line in enumerate(source):
        if "os.environ['HF_HOME'] =" in line:
            source.insert(i+1, "os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'\n")
            break
            
    nb['cells'][cell_idx]['source'] = source
    
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)

add_hf_transfer('notebooks/harsh_offline_pipeline.ipynb', 1)
# For week5, hf_transfer needs to go in _DEPS in cell 2 (index 1)
with open('notebooks/harsh_week5_qwen3vl.ipynb', 'r') as f:
    nb = json.load(f)
source = nb['cells'][1]['source']
for i, line in enumerate(source):
    if "'tqdm', 'pandas', 'numpy'," in line:
        source[i] = line.replace("'tqdm'", "'tqdm', 'hf_transfer'")
    if "os.environ['HF_HOME'] = HF_CACHE" in line:
        source.insert(i+1, "    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'\n")
nb['cells'][1]['source'] = source
with open('notebooks/harsh_week5_qwen3vl.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("HF Transfer enabled in both notebooks.")
