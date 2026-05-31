"""
Run: python -m uvicorn main:app --reload
Doc: http://localhost:8000/docs
"""

import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx

AI_PATH = Path(__file__).parent / "ia-kcal"
sys.path.insert(0, str(AI_PATH))
os.chdir(str(AI_PATH))

from analyze import analyze

app = FastAPI(
    title="JARMY",
    description="API de l'application jarmy",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_TOKEN = os.getenv("KCAL_SECRET_TOKEN", "clesecrete")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/nateraw/food"
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token.")
    return credentials.credentials


class MealRequest(BaseModel):
    text: str = Field(..., example="266g of rice and chicken and for the dessert i ate an ice cream and 50g of apple", description="Description textuelle du repas à analyser")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "266g of rice and chicken and for the dessert i ate an ice cream and 50g of apple"
            }
        }


class FoodItemResponse(BaseModel):
    food: str = Field(..., example="rice", description="Nom de l'aliment détecté")
    grams: float = Field(..., example=266, description="Quantité détectée en grammes")
    kcal: float = Field(..., example=350, description="Calories estimées pour cet aliment")


class MealResponse(BaseModel):
    total_kcal: float = Field(..., example=900, description="Total des calories du repas")
    message: str = Field(..., example="Repas analysé avec succès", description="Message de retour")
    items: list[FoodItemResponse] = Field(..., description="Liste des aliments détectés")


@app.post("/analyze-image", response_model=MealResponse, summary="Analyser une photo de repas", tags=["Analyse IA"])
async def analyze_image_route(file: UploadFile = File(...), token: str = Depends(verify_token)):
    """
    Reçoit une image, utilise Hugging Face Vision pour identifier les aliments,
    puis utilise le NLP interne pour calculer les calories.
    """
    image_bytes = await file.read()

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(HF_API_URL, headers=headers, content=image_bytes)
            response.raise_for_status()
            predictions = response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur avec l'API de vision: {str(e)}")

    top_labels = [p["label"] for p in predictions if p["score"] > 0.1][:3]

    if not top_labels:
        raise HTTPException(status_code=400, detail="Aucun aliment reconnu sur la photo.")

    fake_text = " and ".join([f"100g of {label}" for label in top_labels])

    try:
        result = analyze(fake_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse NLP: {str(e)}")

    return MealResponse(
        total_kcal=result.total_kcal,
        message=f"Image analysée comme: {fake_text}. {result.message}",
        items=[
            FoodItemResponse(food=item["food"], grams=item["grams"], kcal=item["kcal"])
            for item in result.items
        ]
    )


@app.get("/")
def root():
    return {"status": "ok", "service": "JARMY API"}


@app.post("/analyze", response_model=MealResponse)
def analyze_meal(request: MealRequest, token: str = Depends(verify_token)):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Meal text cannot be empty.")

    try:
        result = analyze(request.text)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"AI model not ready: {str(e)}."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    return MealResponse(
        total_kcal=result.total_kcal,
        message=result.message,
        items=[
            FoodItemResponse(
                food=item["food"],
                grams=item["grams"],
                kcal=item["kcal"]
            )
            for item in result.items
        ]
    )