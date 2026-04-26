import json

def insert_hf_home(source_list, after_line):
    new_src = []
    inserted = False
    for line in source_list:
        new_src.append(line)
        if after_line in line and not inserted:
            new_src.append("\n# Set Hugging Face cache to Google Drive to prevent redownloading models\n")
            new_src.append("HF_CACHE = '/content/drive/MyDrive/hf_cache'\n")
            new_src.append("import os\n")
            new_src.append("os.makedirs(HF_CACHE, exist_ok=True)\n")
            new_src.append("os.environ['HF_HOME'] = HF_CACHE\n")
            inserted = True
    return new_src

# 1. Patch harsh_offline_pipeline.ipynb Stage 2 (Cell 2)
with open('notebooks/harsh_offline_pipeline.ipynb', 'r') as f:
    nb1 = json.load(f)
nb1['cells'][2]['source'] = insert_hf_home(nb1['cells'][2]['source'], "drive.mount('/content/drive')")
with open('notebooks/harsh_offline_pipeline.ipynb', 'w') as f:
    json.dump(nb1, f, indent=1)

# 2. Patch harsh_week5_qwen3vl.ipynb Stage 0 (Cell 1)
with open('notebooks/harsh_week5_qwen3vl.ipynb', 'r') as f:
    nb2 = json.load(f)
nb2['cells'][1]['source'] = insert_hf_home(nb2['cells'][1]['source'], "drive.mount('/content/drive', force_remount=False)")
with open('notebooks/harsh_week5_qwen3vl.ipynb', 'w') as f:
    json.dump(nb2, f, indent=1)

print("HF_HOME caching patched successfully into both notebooks.")
