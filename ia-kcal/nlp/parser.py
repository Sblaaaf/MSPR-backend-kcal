"""
NLP parser — spaCy NER only.
Run nlp/train_ner.py first to generate the model.
"""

import re, sys
from pathlib import Path
from difflib import get_close_matches

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.nutrition_data import FOOD_DB

FOODS     = sorted(FOOD_DB.keys(), key=len, reverse=True)
MODEL_DIR = Path(__file__).parent / "food_ner_model"

PORTIONS = {
    "egg": 60, "roast chicken": 150, "chicken": 150, "grilled chicken": 150, "beef steak": 200, "steak": 200,
    "salmon": 150, "tuna": 120, "beef": 150, "shrimp": 100, "turkey": 150,
    "rice": 180, "white rice": 180, "brown rice": 180, "spaghetti": 200, "pasta": 200,
    "bread": 60, "whole wheat bread": 60, "toast": 40,
    "potato": 200, "sweet potato": 150, "quinoa": 185, "overnight oats": 60, "oats": 60,
    "milk": 200, "greek yogurt": 125, "yogurt": 125, "cheddar cheese": 40, "cheese": 40, "butter": 15,
    "apple": 150, "banana": 120, "orange": 180, "strawberry": 150,
    "grapes": 100, "grape": 100, "pear": 160, "mango": 200, "kiwifruit": 80, "kiwi": 80,
    "broccoli": 150, "carrot": 100, "tomato": 120, "lettuce": 80,
    "spinach": 100, "pepper": 100, "onion": 80,
    "almonds": 30, "walnuts": 30,
    "soda": 330, "orange juice": 200, "coffee": 200, "bubble tea": 200, "tea": 200, "coconut water": 200, "water": 300,
    "pizza": 300, "burger": 200, "hamburger": 200, "french fries": 150, "chips": 50,
    "dark chocolate": 30, "honey": 15, "olive oil": 10,
}

UNITS = {
    "g":1,"gr":1,"gram":1,"grams":1,"kg":1000,
    "ml":1,"cl":10,"dl":100,"l":1000,
    "oz":28.35,"lb":453.6,
    "cup":240,"cups":240,"bowl":300,"glass":200,
    "slice":35,"slices":35,"piece":100,"pieces":100,
    "tbsp":15,"tsp":5,"handful":30,"scoop":35,
}

WORDS = {
    "one":1,"two":2,"three":3,"four":4,"five":5,
    "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "a":1,"an":1,"half":0.5,
}

IGNORED = {
    "grilled","baked","steamed","fried","boiled","cooked","raw","roasted",
    "scrambled","fresh","whole","large","small","with","and","a","an",
    "the","of","some","bit","little","had","ate","have","eat","just",
}

SYNONYMS = {
    "eggs": "egg",
    "salad": "lettuce",
    "fries": "french fries",
    "coke": "soda", "cola": "soda", "pepsi": "soda",
    "juice": "orange juice",
    "rice": "rice",
    "noodles": "spaghetti", "spaghetti": "spaghetti", "pasta": "spaghetti",
    "meat": "beef steak", "beef": "beef steak", "steak": "beef steak",
    "chicken": "roast chicken", "chiken": "roast chicken", "chikn": "roast chicken",
    "samon": "salmon", "salman": "salmon",
    "tomatoe": "tomato", "tomatoes": "tomato",
    "carrots": "carrot", "carots": "carrot",
    "potatoes": "potato",
    "burger": "burger", "burguer": "burger", "hamburger": "burger",
    "chocolate": "dark chocolate",
    "oil": "olive oil",
}


def _clean(raw: str) -> str:
    return " ".join(w for w in raw.strip().split() if w not in IGNORED and len(w) > 1)


def _match_food(raw: str) -> str | None:
    raw = raw.strip().lower()

    if raw in SYNONYMS:
        return SYNONYMS[raw]
    for syn, canonical in SYNONYMS.items():
        if syn in raw:
            return canonical

    raw = _clean(raw)
    if not raw:
        return None

    if raw in FOOD_DB:
        return raw
    if raw in SYNONYMS:
        return SYNONYMS[raw]

    for f in FOODS:
        if f == raw:
            return f
        if f in raw and len(f) >= 3:
            return f
        if raw in f and len(raw) >= 4:
            return f

    m = get_close_matches(raw, FOODS, n=1, cutoff=0.72)
    return m[0] if m else None


def _grams(food: str, qty: float, unit: str) -> float:
    unit = unit.lower().strip()
    if unit in ("g", "gr", "gram", "grams"):
        return qty
    if unit in UNITS:
        return qty * UNITS[unit]
    return qty * PORTIONS.get(food, 100)


def _dedup(items: list) -> list:
    result, seen = [], []
    for item in sorted(items, key=lambda x: len(x["food"]), reverse=True):
        f = item["food"]
        if not any(f in s or s in f for s in seen):
            result.append(item)
            seen.append(f)
    return result


def _qty_from_text(text: str) -> tuple[float, str]:
    text = text.strip().lower()
    m = re.match(r'^(\d+(?:\.\d+)?)\s*(g|gr|grams?|kg|ml|cl|oz|lb)$', text)
    if m:
        return float(m.group(1)), m.group(2)
    m = re.match(r'^(\d+(?:\.\d+)?)\s*(\w+)?$', text)
    if m:
        return float(m.group(1)), m.group(2) or ""
    if text in WORDS:
        return WORDS[text], ""
    return 1.0, ""


_nlp = None


def _load_model():
    global _nlp
    if _nlp is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {MODEL_DIR}\n"
                "Lance d'abord : python nlp/train_ner.py"
            )
        import spacy
        _nlp = spacy.load(MODEL_DIR)
    return _nlp


def parse(text: str) -> list[dict]:
    nlp = _load_model()
    doc = nlp(text.lower())
    items = []
    last_qty, last_unit = 1.0, ""

    for ent in doc.ents:
        if ent.label_ == "QUANTITY":
            last_qty, last_unit = _qty_from_text(ent.text)
        elif ent.label_ == "FOOD":
            food = _match_food(ent.text)
            if food:
                grams = _grams(food, last_qty, last_unit)
                items.append({"food": food, "grams": grams})
                last_qty, last_unit = 1.0, ""

    return _dedup(items)


def parser_info() -> str:
    return "spaCy NER (trained model)"