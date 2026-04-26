import json

def patch_notebook(nb_path):
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            for i, line in enumerate(source):
                if "sys.path.insert(0, REPO_DIR)" in line:
                    source[i] = line.replace("sys.path.insert(0, REPO_DIR)", "sys.path.insert(0, os.path.join(REPO_DIR, 'src'))")

    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)

patch_notebook('notebooks/harsh_offline_pipeline.ipynb')
patch_notebook('notebooks/harsh_week5_qwen3vl.ipynb')
print("Patched sys.path in both notebooks.")
