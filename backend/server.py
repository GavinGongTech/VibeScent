from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. Imports are correct
from recommender import recommend_fragrances
from clip_zero_shot import initialize_model, extract_vibe_dictionary

app = FastAPI()

# ==========================================
# WARM BOOT: Load AI & Data globally
# ==========================================
print("--- BOOTING ML PIPELINES ---")

# 🚨 THE FIX: Actually call the function to wake up CLIP! 🚨
initialize_model()

print("Loading Text Embedding Model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

print("Loading Fragrance Database...")
fragrance_df = pd.read_csv('../data/mock_fragrances.csv') 

# ==========================================
# PYDANTIC SCHEMA
# ==========================================
class RecommendRequest(BaseModel):
    image: str       
    mimeType: str
    context: str = "" 

# ==========================================
# MAIN PIPELINE ENDPOINT
# ==========================================
@app.post("/predict")
async def predict(request: RecommendRequest):
    try:
        # STEP 1: Vision - Pass the image and text to CLIP to build the dictionary
        user_event_input, clip_reasoning = extract_vibe_dictionary(request.image, request.context)

        # ---------------------------------------------------------
        # 🖨️ PRINT 1: THE CLIP VISION OUTPUT
        # ---------------------------------------------------------
        print("\n" + "="*50)
        print("👁️  PHASE 1: CLIP VISION MODULE OUTPUT")
        print("="*50)
        print("Extracted Vibe Dictionary:")
        for key, value in user_event_input.items():
            print(f"  - {key}: {value}")
        print(f"\nGenerated Reasoning String:\n  {clip_reasoning}")
        print("="*50 + "\n")

        # STEP 2: Semantic Search
        # 🚨 THE FIX: Changed from find_top_fragrances_semantic to recommend_fragrances 🚨
        results_df = recommend_fragrances(fragrance_df, user_event_input, embedding_model)

        if isinstance(results_df, str):
            raise HTTPException(status_code=404, detail=results_df)

        # ---------------------------------------------------------
        # 🖨️ PRINT 2: THE SEMANTIC RECOMMENDER OUTPUT
        # ---------------------------------------------------------
        print("\n" + "="*50)
        print("🧠 PHASE 2: SEMANTIC RECOMMENDER OUTPUT")
        print("="*50)
        print(results_df[['Name', 'Similarity_Score']].to_string(index=False))
        print("="*50 + "\n")

        # STEP 3: Format Output
        recommendations = []
        for index, row in results_df.reset_index().iterrows():
            recommendations.append({
                "rank": index + 1,
                "name": row['Name'],
                "score": float(row['Similarity_Score']), 
                "notes": row['Notes'].split(" | "), 
                "reasoning": f"{clip_reasoning} Paired with your specific notes, this is a {round(row['Similarity_Score']*100)}% aesthetic match.",
                "occasion": f"The {user_event_input['Season']} Edit"
            })

        return {"recommendations": recommendations}

    except Exception as e:
        import traceback
        print("\n" + "!"*50)
        print("🚨 CRITICAL PIPELINE CRASH 🚨")
        print("!"*50)
        traceback.print_exc()
        print("!"*50 + "\n")
        raise HTTPException(status_code=500, detail="Internal Server Error during curation.")