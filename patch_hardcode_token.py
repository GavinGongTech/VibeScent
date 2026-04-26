import json

import os
TOKEN = os.environ.get('HF_TOKEN', '')

# Patch harsh_offline_pipeline.ipynb
with open('notebooks/harsh_offline_pipeline.ipynb', 'r') as f:
    nb1 = json.load(f)

source1 = nb1['cells'][2]['source']
for i, line in enumerate(source1):
    if "from google.colab import userdata" in line:
        source1[i] = f"    hf_token = '{TOKEN}'\n"
    elif "hf_token = userdata.get('HF_TOKEN')" in line:
        source1[i] = "    os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token\n"
    elif "Loaded HF_TOKEN from Colab Secrets" in line:
        source1[i] = "        print('HF_TOKEN loaded directly.')\n"

nb1['cells'][2]['source'] = source1
with open('notebooks/harsh_offline_pipeline.ipynb', 'w') as f:
    json.dump(nb1, f, indent=1)


# Patch harsh_week5_qwen3vl.ipynb
with open('notebooks/harsh_week5_qwen3vl.ipynb', 'r') as f:
    nb2 = json.load(f)

source2 = nb2['cells'][2]['source']
for i, line in enumerate(source2):
    if "def _g(n):" in line:
        source2[i] = "    pass\n"
    elif "from google.colab import userdata; return userdata.get(n)" in line:
        source2[i] = "    pass\n"
    elif "except: return os.environ.get(n)" in line:
        source2[i] = "    pass\n"
    elif "_hf_token = _g('HF_TOKEN') or _g('HUGGINGFACE_TOKEN')" in line:
        source2[i] = f"    _hf_token = '{TOKEN}'\n"

nb2['cells'][2]['source'] = source2
with open('notebooks/harsh_week5_qwen3vl.ipynb', 'w') as f:
    json.dump(nb2, f, indent=1)

print("Hardcoded HF_TOKEN into both notebooks successfully.")
