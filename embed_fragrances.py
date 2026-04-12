import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Load the dataset
dataset_path = 'data/vibescent_unified.csv'
df = pd.read_csv(dataset_path)

# Drop rows where we can't encode anything or don't know the name.
df = df.dropna(subset=['embedding_text', 'name'])
df = df.reset_index(drop=True)

# Encoding
encoder_model_name = 'all-MiniLM-L6-v2' 
print(f"Loading encoder: {encoder_model_name}...")
model = SentenceTransformer(encoder_model_name)

print("Encoding pre-compiled text descriptions into vectors...")
embeddings = model.encode(df['embedding_text'].tolist(), show_progress_bar=True)

# Store as npy
output_filename = "fragrance_embeddings.npy"
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