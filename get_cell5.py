import json
with open('notebooks/harsh_offline_pipeline.ipynb', 'r') as f:
    nb = json.load(f)
print("".join(nb['cells'][5]['source']))
