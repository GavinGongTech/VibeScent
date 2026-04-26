import json

with open('notebooks/harsh_week5_qwen3vl.ipynb', 'r') as f:
    nb = json.load(f)

# Fix Cell 1 (Stage 0) indentation
source1 = nb['cells'][1]['source']
for i, line in enumerate(source1):
    if "os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'" in line:
        source1[i] = "os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'\n"

# Check Cell 2 (Stage 1 Setup) for HF_HOME and add hf_transfer environment variable
source2 = nb['cells'][2]['source']
inserted_2 = False
for i, line in enumerate(source2):
    if "os.environ['HF_HOME'] = HF_CACHE" in line and not inserted_2:
        # get the indentation of this line
        indent = line[:len(line) - len(line.lstrip())]
        source2.insert(i+1, indent + "os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'\n")
        inserted_2 = True

nb['cells'][1]['source'] = source1
nb['cells'][2]['source'] = source2

with open('notebooks/harsh_week5_qwen3vl.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Fixed indentation and added hf_transfer environment to Cell 2 in week5 notebook.")
