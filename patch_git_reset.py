import json

nb_path = 'notebooks/harsh_offline_pipeline.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

cell_1_src = nb['cells'][1]['source']
for i, line in enumerate(cell_1_src):
    if "subprocess.check_call(['git', '-C', REPO_DIR, 'pull'])" in line:
        cell_1_src[i] = "        subprocess.check_call(['git', '-C', REPO_DIR, 'fetch', '--all'])\n        subprocess.check_call(['git', '-C', REPO_DIR, 'reset', '--hard', 'origin/main'])\n"
        break

nb['cells'][1]['source'] = cell_1_src

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
