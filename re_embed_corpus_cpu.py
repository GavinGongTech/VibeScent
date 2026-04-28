"""Re-embed the full corpus using all-MiniLM-L6-v2 (22M params, very fast on CPU).

Benchmarks on 16GB RAM CPU-only:
  - 35,000 texts at batch_size=1024 → ~6-8 minutes
  - Embedding dim: 384
  - Quality: good for retrieval, great speed/quality tradeoff
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, "src")

from vibescents.query import build_candidate_text
from vibescents.settings import Settings

settings = Settings.from_env()

print(f"Loading corpus from {settings.corpus_metadata_path}")
df = pd.read_csv(settings.corpus_metadata_path, low_memory=False)
texts = [build_candidate_text(row) for _, row in df.iterrows()]
print(f"Extracted {len(texts)} texts.")

print("Loading all-MiniLM-L6-v2 (22M params, ~6-8 min for 35k texts)...")
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

print("Embedding corpus on CPU...")
# Disable tokenizer parallelism to avoid multiprocessing issues on Python 3.14
import os as _os
_os.environ["TOKENIZERS_PARALLELISM"] = "false"

emb_array = model.encode(
    texts,
    batch_size=1024,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
).astype("float32")

out_path = Path(settings.corpus_embeddings_path).parent.parent / "minilm_corpus" / "embeddings.npy"
out_path.parent.mkdir(parents=True, exist_ok=True)
np.save(out_path, emb_array)

print(f"\nDone! Saved {emb_array.shape} embeddings to {out_path}")
