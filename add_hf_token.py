import json

with open('notebooks/harsh_offline_pipeline.ipynb', 'r') as f:
    nb = json.load(f)

source = nb['cells'][2]['source']
insertion_idx = -1
for i, line in enumerate(source):
    if "os.environ['HF_HOME'] = HF_CACHE" in line:
        insertion_idx = i + 1
        break

if insertion_idx != -1:
    source.insert(insertion_idx, "os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'\n")
    source.insert(insertion_idx + 1, "try:\n")
    source.insert(insertion_idx + 2, "    from google.colab import userdata\n")
    source.insert(insertion_idx + 3, "    hf_token = userdata.get('HF_TOKEN')\n")
    source.insert(insertion_idx + 4, "    if hf_token:\n")
    source.insert(insertion_idx + 5, "        os.environ['HF_TOKEN'] = hf_token\n")
    source.insert(insertion_idx + 6, "        print('Loaded HF_TOKEN from Colab Secrets.')\n")
    source.insert(insertion_idx + 7, "except Exception:\n")
    source.insert(insertion_idx + 8, "    pass\n")

nb['cells'][2]['source'] = source

with open('notebooks/harsh_offline_pipeline.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Added HF_TOKEN loading to offline notebook.")
