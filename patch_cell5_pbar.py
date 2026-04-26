import json

nb_path = 'notebooks/harsh_offline_pipeline.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

cell_5_src = nb['cells'][5]['source']
new_src = []
for line in cell_5_src:
    if "for i in range(0, len(texts_to_embed), CHUNK_SIZE):" in line:
        new_src.append('    from tqdm.auto import tqdm\n')
        new_src.append('    with tqdm(total=len(texts_to_embed), desc="Embedding (Outer Chunks)") as outer_pbar:\n')
        new_src.append('        for i in range(0, len(texts_to_embed), CHUNK_SIZE):\n')
    elif "elapsed = time.perf_counter() - t0" in line:
        new_src.append('            outer_pbar.update(len(chunk))\n')
        new_src.append(line)
    elif "np.save(ckpt_path, chunk_emb)" in line:
        new_src.append(line)
        # remove the print so we don't spam the console too much when using tqdm
    elif "print(f\"  -> Processing chunk {i//CHUNK_SIZE + 1} ({len(chunk)} rows)...\")\n" in line:
        pass
    elif "print(f\"  -> Saved checkpoint: {ckpt_path}\")\n" in line:
        pass
    else:
        new_src.append(line)

nb['cells'][5]['source'] = new_src

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
