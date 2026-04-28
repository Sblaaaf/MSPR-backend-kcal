"""
HealthAI — meal calorie analyzer.
Just describe what you ate, get the kcal back.

Usage:
    from analyze import analyze
    result = analyze("2 eggs and a banana")
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.nutrition_data import FOOD_DB
from nlp.parser import parse
from dataclasses import dataclass

@dataclass
class MealResult:
    items: list[dict]   # [{"food": str, "grams": float, "kcal": float}]
    total_kcal: float
    message: str


def analyze(text: str) -> MealResult:
    parsed = parse(text)
    items = []
    total = 0.0

    for item in parsed:
        food  = item["food"]
        grams = item["grams"]
        kcal_100 = FOOD_DB.get(food, 0)
        kcal = round(kcal_100 * grams / 100, 1)
        total += kcal
        items.append({"food": food, "grams": round(grams, 1), "kcal": kcal})

    total = round(total, 1)
    message = f"This meal contains approximately {total:.0f} kcal."
    if not items:
        message = "No food recognized. Try being more specific (e.g. '200g chicken and rice')."

    return MealResult(items=items, total_kcal=total, message=message)