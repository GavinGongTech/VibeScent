import os
import time
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import json

# 1. Load environment variables from the .env file
load_dotenv()

# 2. Define the strict schema
class FragranceVibe(BaseModel):
    # Using string numbers here to be absolutely safe across all parsers
    Formality: Literal["0.3", "0.5", "0.8"]
    Season: Literal["Spring", "Summer", "Autumn", "Winter"]
    Gender: Literal["Male", "Female", "Unisex"]
    Time_of_Day: Literal["Day", "Night"]
    Frequency: Literal["Often", "Occasionally", "Rarely"]

def build_clustered_database():
    # 3. Initialize the OpenAI client pointed to Groq's servers
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    print("Loading 36,000 fragrances...")
    df = pd.read_csv("../data/vibescent_unified.csv")
    df['embedding_text'] = df['embedding_text'].fillna("")

    print("Generating dense embeddings... (This takes a minute)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['embedding_text'].tolist(), show_progress_bar=True)

    print("Clustering into 500 archetypes...")
    num_clusters = 500
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(embeddings)

    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)

    print("Extracting vibes via Groq (llama-3.1-8b-instant)...")
    cluster_vibes = {}

    for cluster_idx, row_idx in tqdm(enumerate(closest_indices), total=num_clusters):
        centroid_row = df.iloc[row_idx]

        prompt = f"""
        Analyze this fragrance and map it to our exact dimensions.
        Name: {centroid_row['name']}
        Brand: {centroid_row['brand']}
        Main Accords: {centroid_row['main_accords']}
        Top Notes: {centroid_row['top_notes']}
        Middle Notes: {centroid_row['middle_notes']}
        Base Notes: {centroid_row['base_notes']}
        """

        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are an expert fragrance evaluator. You MUST output ONLY valid JSON that perfectly matches this exact schema: {FragranceVibe.model_json_schema()}"
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}, # The universal Groq format!
                temperature=0.0
            )
            
            # 5. Extract the raw text and turn it into a Python dictionary
            raw_json_string = completion.choices[0].message.content
            vibe_data = json.loads(raw_json_string)
            
            # 6. Pass it through our Pydantic class just to guarantee it followed the rules
            validated_object = FragranceVibe(**vibe_data)
            
            # Store it
            cluster_vibes[cluster_idx] = validated_object.model_dump()
            
        except Exception as e:
            print(f"\nFailed on cluster {cluster_idx}: {e}")
            # Safe fallback
            cluster_vibes[cluster_idx] = {
                "Formality": "0.5", "Season": "Spring", "Gender": "Unisex",
                "Time_of_Day": "Day", "Frequency": "Often"
            }
            
        # 5. Pace the requests to respect the 30 RPM free tier limit!
        time.sleep(2.5) 

    print("Propagating labels to the full dataset...")
    vibes_df = pd.DataFrame.from_dict(cluster_vibes, orient='index')
    vibes_df.index.name = 'cluster_id'

    final_df = df.merge(vibes_df, on='cluster_id', how='left')
    final_df = final_df.drop(columns=['cluster_id'])

    final_df.to_csv("../data/vibescent_enriched.csv", index=False)
    print("Success! Saved vibescent_enriched.csv")

if __name__ == "__main__":
    build_clustered_database()