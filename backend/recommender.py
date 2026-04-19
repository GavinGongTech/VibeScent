import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_fragrances(df, user_input, ai_model):
    """
    Partitions the dataframe using categorical keys, then uses an AI language 
    model to calculate semantic similarity for the nuanced vibe.
    """
    # 1. Extract the text query and remove non-filtering fields
    vibe_description = user_input.get('Description', '')
    cat_filters = {k: v for k, v in user_input.items() if k not in ['Name', 'Description']}
    
    # 2. Hard Filtering (Partitioning)
    filtered_df = df.copy()
    
    for column, value in cat_filters.items():
        if value is not None and str(value).strip() != '':
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

    # 3. Semantic Vectorization (The AI Upgrade)
    # Convert the filtered database's 'Notes' into dense AI embeddings
    # We use .tolist() to feed the model a clean list of text strings
    db_embeddings = ai_model.encode(filtered_df['Notes'].tolist())
    
    # Convert the user's abstract description into an embedding
    user_embedding = ai_model.encode([vibe_description])
    
    # 4. Calculate Cosine Similarity
    # We compare the user's single vector against the matrix of database vectors
    similarity_scores = cosine_similarity(user_embedding, db_embeddings).flatten()
    
    # 5. Rank and Extract Top 3
    filtered_df['Similarity_Score'] = similarity_scores
    top_3 = filtered_df.sort_values(by='Similarity_Score', ascending=False).head(3)
    
    return top_3[['Name', 'Similarity_Score', 'Notes']]

# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":
    # --- IMPORTANT PRODUCTION NOTE ---
    # Loading the model takes a few seconds. Do this ONCE when your server starts up,
    # not inside the search function, otherwise every user search will be very slow!
    print("Loading AI Model (this takes a few seconds the first time)...")
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    # Load your actual CSV here: df = pd.read_csv('your_fragrance_db.csv')
    df = pd.read_csv('../data/mock_fragrances.csv')
    
    # Notice we can use wildly abstract descriptions now
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
    # Pass the pre-loaded model into the function
    results = recommend_fragrances(df, user_event_input, model)
    
    print("\nTop Recommendations:\n")
    if isinstance(results, str):
        print(results)
    else:
        print(results.to_string(index=False))