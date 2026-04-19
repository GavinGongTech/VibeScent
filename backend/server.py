import base64
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import clip_zero_shot

app = FastAPI(title="ScentAI Model API")

@app.on_event("startup")
def startup_event():
    print("\n" + "="*50)
    print("VIBESCENT WEB SERVER BOOTING UP", flush=True)
    print("="*50 + "\n", flush=True)
    
    # Tell the ML module to load the heavy weights into memory
    clip_zero_shot.initialize_model()

class RecommendRequest(BaseModel):
    image: str       
    mimeType: str
    context: str    

@app.post("/predict")
async def predict_fragrance(req: RecommendRequest):
    try:
        print("\n" + "*"*40, flush=True)
        print("WEB: Request received. Processing image...", flush=True)
        
        # 1. Decode base64 to PIL Image
        image_data = base64.b64decode(req.image.split(",")[1] if "," in req.image else req.image)
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # 2. Hand the image off to the ML module
        recommendations = clip_zero_shot.get_recommendations(img)
        
        print("WEB: Inference complete. Sending to Next.js.", flush=True)
        print("*"*40 + "\n", flush=True)

        return { "recommendations": recommendations }

    except Exception as e:
        print(f"Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))