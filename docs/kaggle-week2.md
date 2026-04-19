# Running Week 2 Pipeline on Kaggle

> Pre-computed embeddings (stages 4 & 5) are already in the repo via Git LFS.
> You are picking up from **Stage 7**. Total runtime: ~4 hours on T4.

---

## Prerequisites

### 1. Clone the repo

```bash
git lfs install
git clone https://github.com/GavinGongTech/VibeScent.git
```

The `artifacts/week2_precomputed/` folder contains the fixed corpus and occasion embeddings. They download automatically via Git LFS — no separate file sharing needed.

### 2. Set up Kaggle Secrets

Go to **kaggle.com → top-right avatar → Settings → Secrets** and add:

| Secret name | Where to get it |
|---|---|
| `GEMINI_API_KEY` | Ask Harsh — it's the Google AI Studio key |
| `HF_TOKEN` | huggingface.co/settings/tokens → New token (read access) |

Also: go to [huggingface.co/google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) and **accept the license agreement** before running — otherwise the model download fails with a 401.

---

## Notebook Setup

### 3. Create the Kaggle Notebook

1. Go to [kaggle.com/code](https://kaggle.com/code) → **New Notebook**
2. Click **File → Import Notebook** → upload `notebooks/harsh_week2_pipeline.ipynb`
3. On the right panel, configure:
   - **Accelerator**: T4 x2 GPU
   - **Internet**: ON (required for model downloads and Gemini API)
   - **Persistence**: Files only
4. Under the **Secrets** tab on the right panel: toggle **on** both `GEMINI_API_KEY` and `HF_TOKEN`

### 4. Add two extra cells immediately after Stage 1

After the Stage 1 setup cell finishes (the one that ends with "All modules importable"), **add these two cells in order** before running Stage 2:

**Cell A — Copy pre-computed artifacts into place:**

```python
from pathlib import Path
import shutil

src = Path("/kaggle/working/vibescent/artifacts/week2_precomputed")
dst = Path("/kaggle/working/vibescent/artifacts")
dst.mkdir(parents=True, exist_ok=True)

for folder in ["fragrance_raw_full", "occasions"]:
    shutil.copytree(str(src / folder), str(dst / folder), dirs_exist_ok=True)

# Verify stage gates will pass
import json
for folder in ["fragrance_raw_full", "occasions"]:
    m = json.loads((dst / folder / "manifest.json").read_text())
    print(f"{folder}: {m['row_count']} rows, pipeline_version={m['pipeline_version']}")
```

Expected output:
```
fragrance_raw_full: 35889 rows, pipeline_version=v1
occasions: 8 rows, pipeline_version=v1
```

If you see those two lines, stages 3/4/5 will be skipped automatically.

**Cell B — HuggingFace login (needed to download Gemma):**

```python
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("HF login OK")
```

---

## Running the Pipeline

### 5. Run all cells top to bottom

What each stage does on T4 and how long it takes:

| Stage | What happens | Est. time |
|---|---|---|
| 1 | Setup, clone repo, install deps | ~5 min |
| 2 | Sanity check — loads CSV, validates files | <1 min |
| 3 | **Skipped** — embeddings already done | instant |
| 4 | **Skipped** — loads cached 35,889 corpus embeddings | ~5 sec |
| 5 | **Skipped** — loads cached 8 occasion embeddings | ~5 sec |
| 6 | Loads **Gemma 3 4B int4** via transformers (~4 GB VRAM, fits on T4) | ~8 min |
| 7 | Generates 20 benchmark labels via **Gemini API** | ~5 min |
| 8 | Enriches 500-row TIER C via Gemma | ~20 min |
| 9 | Retrieval comparison report (TIER A raw) | ~2 min |
| 10 | Enriches 2K TIER B via Gemma | ~90 min |
| 11–13 | Validation + intermediate checks | ~5 min |
| 14 | Multimodal embeddings — loads Qwen3-VL-8B int8, embeds TIER B+C | ~60 min |
| 17 | Multimodal embed TIER B | ~30 min |

**Total: ~4 hours.** Kaggle gives 30h/week free — this fits comfortably in one session.

---

## Saving Outputs Between Sessions

Kaggle wipes `/kaggle/working/` when the session ends. Before closing:

1. In the notebook editor, click the **folder icon** (Output panel) on the right
2. Download these files/folders from `/kaggle/working/vibescent/artifacts/`:
   - `benchmark_cases.json`
   - `vibescent_enriched_500_v2.csv`
   - `vibescent_enriched_2k_v2.csv`
   - `fragrance_enriched_500/` (whole folder)
   - `fragrance_enriched_2k/` (whole folder)
   - `multimodal_2k/` (whole folder)
3. Back them up to Google Drive or re-upload as a new Kaggle dataset version

If the session crashes mid-run, the notebook has checkpoint gates — re-running any cell resumes from where it left off within the same session. Across sessions, restore the output files using the copy cell pattern above.

---

## Troubleshooting

**Stage 6 OOM / kernel crash**
Confirm T4 x2 is selected (32 GB total VRAM). If still crashing, add this cell before Stage 6:
```python
USE_LIGHTWEIGHT_LLM = True
USE_LOCAL_LLM = False
```

**Stage 7 fails with "GEMINI_API_KEY not set"**
The secret wasn't toggled on for this notebook. Go to the Secrets tab, enable it, then restart the kernel and re-run from Stage 1.

**Stages 3/4/5 don't skip (shows "Resuming from batch..." instead of "loading cached")**
Cell A above didn't run, or ran before Stage 1 finished. Confirm `/kaggle/working/vibescent/artifacts/fragrance_raw_full/manifest.json` exists and contains `"pipeline_version": "v1"`.

**HF download 401 / gated model error**
- Re-run Cell B (HF login)
- Double-check you accepted the Gemma license at huggingface.co/google/gemma-4-E4B-it
- Confirm `HF_TOKEN` secret is enabled for this notebook

**Stage 9 retrieval comparison fails with FileNotFoundError on occasions.json or benchmark_briefs.json**
Main branch moved these files. Add this cell before Stage 9:
```python
import os
os.makedirs("examples", exist_ok=True)
import shutil
if not os.path.exists("examples/occasions.json"):
    shutil.copy("data/occasions.json", "examples/occasions.json")
```
