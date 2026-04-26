import json

with open('notebooks/harsh_week5_qwen3vl.ipynb', 'r') as f:
    nb = json.load(f)

source = nb['cells'][2]['source']
new_source = []
skip = False
for i, line in enumerate(source):
    if line.startswith('def _secrets():'):
        new_source.append(line)
        new_source.append("    global _hf_token\n")
        new_source.append("    import os; _hf_token = os.environ.get('HF_TOKEN', '')\n")
        skip = True
    elif skip and line.startswith('_st = threading.Thread'):
        skip = False
        new_source.append(line)
    elif not skip:
        new_source.append(line)

nb['cells'][2]['source'] = new_source

with open('notebooks/harsh_week5_qwen3vl.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Fixed indentation in week5 notebook.")
