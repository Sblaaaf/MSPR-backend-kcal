"""
Lookup nutritionnel sur le catalogue `aliment` (alimenté par l'ETL), via le
service meal. Source unique des macros (calories + protéines/glucides/lipides/
fibres) pour les analyses texte et photo. Fallback silencieux si indisponible.
"""
import os
import re
import httpx
from difflib import get_close_matches

MEAL_SERVICE_URL = os.getenv("MEAL_SERVICE_URL", "http://meal:8003")

# Le catalogue (datasets Kaggle) est majoritairement en ANGLAIS : on cherche
# donc en anglais. Quelques alias pour rapprocher les labels Food-101/VLM des
# termes du catalogue (et NON des traductions FR, qui matcheraient des sous-mots).
SYNONYMS = {
    "fries": "french fries",
    "chips": "french fries",
    "soda": "soft drink",
    "kiwifruit": "kiwi",
}


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def lookup(name: str) -> dict | None:
    """
    Retourne {calories_100g, proteines_g, glucides_g, lipides_g, fibres_g, nom}
    pour l'aliment le plus proche du catalogue ETL, ou None si introuvable.
    """
    q = _normalize(name)
    q = SYNONYMS.get(q, q)

    rows = _search(q)
    if not rows and " " in q:
        # tente le dernier mot (souvent l'aliment principal)
        rows = _search(q.split()[-1])
    if not rows:
        return None
    return _shape(_best_match(q, rows))


def _tokens(s: str) -> list[str]:
    """Mots significatifs (>= 3 lettres) d'un libellé, sans ponctuation/parenthèses."""
    return [w for w in re.findall(r"[a-z]+", s.lower()) if len(w) >= 3]


def _best_match(q: str, rows: list[dict]) -> dict:
    """
    Classe les candidats par recouvrement de mots avec la requête :
      1. égalité exacte
      2. max de mots de la requête présents ; à égalité, moins de mots "en trop", puis plus court
      3. repli flou (difflib) si aucun mot commun
      4. nom le plus court (le moins "bruité")
    Évite les faux positifs type « red bell pepper » -> « Black Pepper ».
    """
    norm = [(str(r["nom"]).lower(), r) for r in rows]
    for n, r in norm:                                  # 1. égalité exacte
        if n == q:
            return r

    q_tokens = set(_tokens(q))
    best, best_key = None, None
    for n, r in norm:                                  # 2. recouvrement de mots
        n_tokens = set(_tokens(n))
        overlap = len(q_tokens & n_tokens)
        if overlap == 0:
            continue
        # plus de recouvrement > moins de mots superflus > nom plus court
        key = (overlap, -len(n_tokens - q_tokens), -len(n))
        if best_key is None or key > best_key:
            best, best_key = r, key
    if best is not None:
        return best

    close = get_close_matches(q, [n for n, _ in norm], n=1, cutoff=0.6)  # 3. flou
    if close:
        return dict(norm)[close[0]]
    return min(norm, key=lambda x: len(x[0]))[1]        # 4. nom le plus court


def _search(query: str) -> list[dict]:
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{MEAL_SERVICE_URL}/aliments", params={"query": query})
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return []


def _shape(row: dict) -> dict:
    def num(v):
        return float(v) if v is not None else 0.0
    return {
        "nom": row.get("nom"),
        "calories_100g": num(row.get("calories_100g")),
        "proteines_g": num(row.get("proteines_g")),
        "glucides_g": num(row.get("glucides_g")),
        "lipides_g": num(row.get("lipides_g")),
        "fibres_g": num(row.get("fibres_g")),
    }
