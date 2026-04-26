import json

with open("notebooks/harsh_week5_qwen3vl.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

changes = 0
for cell in nb["cells"]:
    src = cell.get("source", [])
    new_src = []
    for line in src:
        new_line = line.replace("Settings(api_key=None)", "Settings()")
        if new_line != line:
            changes += 1
        new_src.append(new_line)
    cell["source"] = new_src

with open("notebooks/harsh_week5_qwen3vl.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Fixed {changes} Settings(api_key=None) occurrences")
