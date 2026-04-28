"""
Nutritional database — loads kaggle_nutrition.csv.
All values normalized to kcal per 100g.
"""

import re, csv
from pathlib import Path

CSV_PATH = Path(__file__).parent / "kaggle_nutrition.csv"

# Used only to convert "per portion" values from the CSV into kcal/100g.
# Example: "Apple = 95 kcal" + "apple = 150g portion" → 63 kcal/100g
PORTION_WEIGHT_G = {
    "egg": 60, "apple": 150, "banana": 120, "orange": 180,
    "strawberry": 150, "grape": 100, "pear": 160, "mango": 200,
    "pineapple": 150, "kiwi": 80, "peach": 150, "blueberry": 100,
    "broccoli": 150, "carrot": 100, "tomato": 120, "lettuce": 80,
    "spinach": 100, "zucchini": 150, "pepper": 100, "onion": 80,
    "chicken": 150, "salmon": 150, "tuna": 120, "beef": 150,
    "shrimp": 100, "turkey": 150, "pork": 150,
    "rice": 180, "pasta": 200, "bread": 60, "toast": 60,
    "quinoa": 185, "oats": 60, "potato": 200, "sweet potato": 150,
    "milk": 200, "yogurt": 125, "cheese": 40, "butter": 15,
    "lentils": 200, "chickpeas": 150, "beans": 150, "tofu": 150,
    "almonds": 30, "walnuts": 30,
    "coffee": 200, "tea": 200, "juice": 200, "soda": 330, "water": 300,
    "pizza": 300, "burger": 200, "fries": 150, "chips": 50,
    "chocolate": 30, "honey": 15, "oil": 10,
    "default": 100,
}


def _normalize(raw: str) -> str:
    """Clean a food name from the CSV: remove parentheses, quantities, adjectives."""
    raw = re.sub(r"\(.*?\)", "", raw)
    raw = re.sub(r"\b\d+\s*(oz|g|ml|cup|cups|slice|slices|tbsp|tsp)\b", "", raw, flags=re.I)
    raw = re.sub(r"\b(grilled|steamed|baked|fried|boiled|cooked|raw|large|small|whole|black|white|brown|low-fat|non-fat|scrambled|hard-boiled|roasted|mixed)\b", "", raw, flags=re.I)
    return re.sub(r"\s+", " ", raw).strip().lower()


def _portion(name: str) -> int:
    """Return the reference portion weight in grams for a food name."""
    for key, weight in PORTION_WEIGHT_G.items():
        if key in name:
            return weight
    return PORTION_WEIGHT_G["default"]


def load() -> dict[str, float]:
    """
    Load the CSV and return a dict {food_name: kcal_per_100g}.
    Raises FileNotFoundError if the CSV is missing.
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            "Make sure kaggle_nutrition.csv is in the data/ folder."
        )

    db = {}
    errors = 0

    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = _normalize(row.get("Food_Item", ""))
            if not name:
                continue
            try:
                raw_kcal = str(row.get("Calories (kcal)", 0) or 0).strip()
                m = re.search(r"[\d.]+", raw_kcal)
                kcal_portion = float(m.group()) if m else 0.0
                portion_g = _portion(name)
                kcal_100g = round(kcal_portion * 100 / portion_g, 1)
                if name not in db:
                    db[name] = kcal_100g
            except Exception:
                errors += 1

    if errors:
        print(f"[nutrition] {errors} rows skipped (bad data)")

    print(f"[nutrition] {len(db)} foods loaded from CSV")
    return db


FOOD_DB = load()