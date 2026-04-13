# VibeScent — Outfit Vibe Classifier

Multi-task deep learning pipeline that classifies outfit images across 5 vibe dimensions. Outputs structured vibe labels and 512-d embeddings for downstream use by teammates.

## What It Does

Given an outfit image, the model predicts:

| Dimension | Type | Classes |
|---|---|---|
| `formal` | classification | casual / smart casual / formal |
| `season` | classification | spring / summer / fall / winter |
| `gender` | classification | male / female / neutral |
| `time` | classification | day / night |
| `frequency` | classification | occasional / everyday |

It also outputs a **512-d embedding** per image for similarity search or downstream models.

---

## The 3 Experiments

### 1. CNN Baseline (`--model cnn`)
ResNet-50 pretrained on ImageNet. All layers frozen except `layer4` and the 5 task heads. Captures local texture and pattern features. Fastest to train.

### 2. CLIP Standalone (`--model clip`)
Frozen CLIP ViT-L/14 image encoder + a small projection MLP + 5 task heads. Leverages CLIP's semantic understanding of clothing concepts from image–text pretraining. No gradient flows into the encoder.

### 3. CNN + CLIP Hybrid (`--model hybrid`)
Concatenates ResNet-50 local features (2048-d) with frozen CLIP global features (768-d) → fusion MLP → 512-d → 5 task heads. Best of both: local texture detail + global semantic context.

---

## Setup

```bash
git clone https://github.com/GavinGongTech/VibeScent.git
cd VibeScent
pip install -r requirements.txt
```

---

## Step 1 — Generate Pseudo Labels

Runs CLIP zero-shot labeling over all images in `train/` using `configs/vibes.json` prompts. Saves labels to `data/labels/pseudo_labels.json`.

```bash
python src/clip_zero_shot.py
```

---

## Step 2 — Train a Model

```bash
# CNN Baseline
python train/train.py --model cnn --epochs 15 --batch_size 64 --lr 3e-5

# CLIP Standalone
python train/train.py --model clip --epochs 15 --batch_size 32 --lr 3e-5

# CNN + CLIP Hybrid
python train/train.py --model hybrid --epochs 15 --batch_size 32 --lr 3e-5
```

Checkpoints are saved to `checkpoints/{model}/best.pt`.
Loss history is saved to `logs/{model}_losses.json`.

Optional args: `--dropout` (default 0.5), `--data_dir` (default `.`), `--save_dir` (default `checkpoints/`).

---

## Step 3 — Evaluate

```bash
python src/evaluate.py --models cnn clip hybrid
```

Outputs:
- `outputs/metrics.json` — accuracy and macro F1 per model per dimension
- `outputs/plots/accuracy_comparison.png` — bar chart across all models
- `outputs/plots/cm_{model}_{dim}.png` — confusion matrices
- `outputs/plots/loss_curve_{model}.png` — training curves

---

## Step 4 — Run Inference

Single image:
```bash
python src/inference.py \
  --model hybrid \
  --checkpoint checkpoints/hybrid/best.pt \
  --input path/to/outfit.jpg
```

Entire folder:
```bash
python src/inference.py \
  --model hybrid \
  --checkpoint checkpoints/hybrid/best.pt \
  --input path/to/images/
```

---

## Output Format (for teammates)

**Vibe label** — saved to `outputs/vibes/{image_name}.json`:
```json
{
  "formal": "smart casual",
  "season": "fall",
  "gender": "female",
  "time": "day",
  "frequency": "everyday"
}
```

**Embedding** — saved to `outputs/embeddings/{image_name}.npy`:
- Shape: `(512,)`, dtype `float32`
- Load with: `np.load("outputs/embeddings/image_name.npy")`

---

## Project Structure

```
configs/          vibes.json — CLIP prompts for all 5 label dimensions
models/           cnn_baseline.py, clip_standalone.py, cnn_clip_hybrid.py
src/              clip_zero_shot.py, data_loader.py, evaluate.py, inference.py
train/            train.py — shared training script for all 3 models
notebooks/        train_colab.ipynb — Google Colab training notebook
data/labels/      pseudo_labels.json — CLIP-generated training labels
checkpoints/      best.pt per model
outputs/          vibes/, embeddings/, plots/, metrics.json
logs/             {model}_losses.json
```

---

## Google Colab

Open `notebooks/train_colab.ipynb` in Colab with GPU enabled. Follow the cells in order: clone → check GPU → upload labels → mount Drive → train → download checkpoints.
