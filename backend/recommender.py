import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_fragrances(df, user_input, ai_model):
    """
    Partitions the dataframe using categorical keys, then uses an AI language 
    model to calculate semantic similarity for the nuanced vibe.
    """
    # 1. Extract the text query
    vibe_description = user_input.get('Description', '')

    # 2. Hard Filtering (Partitioning)
    cat_filters = {k: v for k, v in user_input.items() if k not in ['Name', 'Description', 'Longevity']}
    filtered_df = df.copy()

    for column, value in cat_filters.items():
        # Only filter if the column actually exists in our new CSV
        if column in filtered_df.columns and value is not None and str(value).strip() != '':
            if column == 'Formality':
                try:
                    target = float(value)
                    tolerance = 0.15
                    filtered_df = filtered_df[
                        (filtered_df[column] >= target - tolerance) &
                        (filtered_df[column] <= target + tolerance)
                    ]
                except ValueError:
                    print(f"Warning: Formality input '{value}' is not a valid number.")
            else:
                exact_word_pattern = rf"\b{value}\b"
                filtered_df = filtered_df[
                    filtered_df[column].astype(str).str.contains(exact_word_pattern, case=False, na=False, regex=True)
                ]

    if filtered_df.empty:
        return "No fragrances found matching those exact categories."

    # 3. Semantic Vectorization
    db_embeddings = ai_model.encode(filtered_df['Notes'].tolist())
    user_embedding = ai_model.encode([vibe_description])

    # 4. Calculate Cosine Similarity
    similarity_scores = cosine_similarity(user_embedding, db_embeddings).flatten()
    filtered_df = filtered_df.copy()
    filtered_df['Similarity_Score'] = similarity_scores

    # 5. Rank and Extract Top 3
    top_3 = filtered_df.sort_values(by='Similarity_Score', ascending=False).head(3)
    return top_3[['Name', 'brand', 'Similarity_Score', 'Notes']]

# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":
    print("Loading AI Model (this takes a few seconds the first time)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    df = pd.read_csv('vibescent_enriched.csv')
    df = df.rename(columns={'name': 'Name', 'embedding_text': 'Notes'})

    user_event_input = {
        'Name': 'My Holiday Party',
        'Formality': 0.3,
        'Season': 'Winter',
        'Gender': 'Male',
        'Time_of_Day': 'Day',
        'Frequency': 'Often',
        'Longevity': 'Long',
        'Description': 'I want to smell like a dark, mysterious forest.'
    }

    print("\nRunning Semantic Search...")
    results = recommend_fragrances(df, user_event_input, model)

    print("\nTop Recommendations:\n")
    if isinstance(results, str):
        print(results)
    else:
        print(results.to_string(index=False))