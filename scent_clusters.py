import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load data
embeddings = np.load("embeddings/fragrance_embeddings.npy")
df = pd.read_csv("data/processed/vibescent_unified.csv")
df = df.dropna(subset=['embedding_text', 'name']).reset_index(drop=True)

# Define Cluster Model
N_CLUSTERS = 150 
print(f"Running K-Means with {N_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')

# Fit Predict
cluster_labels = kmeans.fit_predict(embeddings)

# Label
df['vibe_cluster_id'] = cluster_labels

# Save clusters
df.to_csv("data/processed/vibescent_clustered.csv", index=False)

# Save to map future vectors
joblib.dump(kmeans, "models/kmeans_fragrance_model.pkl")

print("Clustering complete. Here is a sample of Cluster 42:")

# Sanity check
cluster_42 = df[df['vibe_cluster_id'] == 42]
print(cluster_42[['name', 'embedding_text']].head())