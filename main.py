"""
Run: python -m uvicorn main:app --reload
Doc: http://localhost:8000/docs
"""

import sys
import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pyctuator.pyctuator import Pyctuator

import vision
from nutrition_lookup import lookup as nutrition_lookup

AI_PATH = Path(__file__).parent / "ia-kcal"
sys.path.insert(0, str(AI_PATH))
os.chdir(str(AI_PATH))

from analyze import analyze  # noqa: E402

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

Pyctuator(
    app,
    "JARMY Kcal Service",
    app_url="http://localhost:8001",
    pyctuator_endpoint_url="http://localhost:8001/pyctuator",
    registration_url=None,
)

SECRET_TOKEN = os.getenv("KCAL_SECRET_TOKEN", "clesecrete")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/nateraw/food"
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
    proteines_g: Optional[float] = Field(None, description="Protéines estimées (g)")
    glucides_g: Optional[float] = Field(None, description="Glucides estimés (g)")
    lipides_g: Optional[float] = Field(None, description="Lipides estimés (g)")
    fibres_g: Optional[float] = Field(None, description="Fibres estimées (g)")
    source: Optional[str] = Field(None, description="Source nutrition : 'catalogue' (ETL) ou 'fallback'")


class MealResponse(BaseModel):
    total_kcal: float = Field(..., example=900, description="Total des calories du repas")
    message: str = Field(..., example="Repas analysé avec succès", description="Message de retour")
    items: list[FoodItemResponse] = Field(..., description="Liste des aliments détectés")
    total_proteines_g: Optional[float] = Field(None, description="Total protéines (g)")
    total_glucides_g: Optional[float] = Field(None, description="Total glucides (g)")
    total_lipides_g: Optional[float] = Field(None, description="Total lipides (g)")


def _food_db_kcal(food: str, grams: float) -> float:
    """Fallback calorique via le CSV embarqué (FOOD_DB), si le catalogue ETL ne connaît pas l'aliment."""
    try:
        from data.nutrition_data import FOOD_DB
        return FOOD_DB.get(food.lower(), 0) * grams / 100.0
    except Exception:
        return 0.0


def _enrich(food: str, grams, fallback_kcal=None) -> FoodItemResponse:
    """Construit un item enrichi des macros via le catalogue ETL ; fallback CSV sinon."""
    grams = float(grams) if grams is not None else 100.0
    factor = grams / 100.0
    info = nutrition_lookup(food)
    if info and info["calories_100g"] > 0:
        return FoodItemResponse(
            food=info["nom"] or food,
            grams=round(grams, 1),
            kcal=round(info["calories_100g"] * factor, 1),
            proteines_g=round(info["proteines_g"] * factor, 1),
            glucides_g=round(info["glucides_g"] * factor, 1),
            lipides_g=round(info["lipides_g"] * factor, 1),
            fibres_g=round(info["fibres_g"] * factor, 1),
            source="catalogue",
        )
    kcal = fallback_kcal if fallback_kcal is not None else _food_db_kcal(food, grams)
    return FoodItemResponse(food=food, grams=round(grams, 1), kcal=round(kcal or 0.0, 1), source="fallback")


def _meal_response(items: list[FoodItemResponse], message: str) -> MealResponse:
    return MealResponse(
        total_kcal=round(sum(i.kcal for i in items), 1),
        message=message,
        items=items,
        total_proteines_g=round(sum(i.proteines_g or 0 for i in items), 1),
        total_glucides_g=round(sum(i.glucides_g or 0 for i in items), 1),
        total_lipides_g=round(sum(i.lipides_g or 0 for i in items), 1),
    )


@app.post("/analyze-image", response_model=MealResponse, summary="Analyser une photo de repas", tags=["Analyse IA"])
async def analyze_image_route(file: UploadFile = File(...), token: str = Depends(verify_token)):
    """
    Reçoit une image, identifie les aliments via le fournisseur de vision
    (Claude ou HuggingFace selon VISION_PROVIDER, avec fallback), puis calcule
    calories + macros à partir du catalogue ETL.
    """
    image_bytes = await file.read()

    try:
        detected = vision.recognize(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur avec l'API de vision: {str(e)}")

    if not detected:
        raise HTTPException(status_code=400, detail="Aucun aliment reconnu sur la photo.")

    items = [_enrich(d["food"], d.get("grams")) for d in detected]
    noms = ", ".join(i.food for i in items)
    return _meal_response(items, message=f"Image analysée : {noms}.")


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

    # Enrichit chaque aliment via le catalogue ETL (macros), avec le kcal
    # du NLP/CSV comme fallback si l'aliment n'est pas au catalogue.
    items = [_enrich(item["food"], item["grams"], fallback_kcal=item["kcal"]) for item in result.items]
    return _meal_response(items, message=result.message)