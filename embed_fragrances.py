import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Load the dataset
dataset_filename = 'data/processed/vibescent_unified.csv'
df = pd.read_csv(dataset_filename)

# Drop rows where we can't encode anything or don't know the name.
df = df.dropna(subset=['embedding_text', 'name'])
df = df.reset_index(drop=True)

# Embedding
embedding_model_name = 'Qwen/Qwen3-Embedding-8B' 
print(f"Loading encoder: {embedding_model_name}...")
model = SentenceTransformer(
    embedding_model_name, 
    trust_remote_code=True,
    model_kwargs={'torch_dtype': 'auto'}
)
print("Encoding pre-compiled text descriptions into vectors...")
embeddings = model.encode(
    df['embedding_text'].tolist(), 
    normalize_embeddings=True, 
    show_progress_bar=True
)

# Store as npy
output_filename = "embeddings/fragrance_embeddings.npy"
np.save(output_filename, embeddings)
print(f"Saved {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]} to {output_filename}")

# Nearest neighbor sanity check
print("\n--- Running Sanity Check ---")

knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(embeddings)

test_index = 0
test_perfume_name = df['name'].iloc[test_index] 
print(f"Querying nearest neighbors for: {test_perfume_name}\nDescription: {df['embedding_text'].iloc[test_index]}\n")

distances, indices = knn.kneighbors([embeddings[test_index]])

for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    if i == 0: continue 
    
    neighbor_name = df['name'].iloc[idx]
    # Update this print statement to use the new column name
    neighbor_desc = df['embedding_text'].iloc[idx]
    
    print(f"{i}. {neighbor_name} (Distance: {dist:.4f})")
    print(f"   {neighbor_desc}\n")