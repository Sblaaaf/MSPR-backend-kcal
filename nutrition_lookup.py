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


def _has_macros(row: dict) -> int:
    """1 si la ligne porte des macros exploitables (sinon des lignes 'calories seules'
    du catalogue gagneraient l'égalité exacte et afficheraient P0/G0/L0)."""
    v = row.get("proteines_g")
    return 1 if isinstance(v, (int, float)) and v > 0 else 0


def _best_match(q: str, rows: list[dict]) -> dict:
    """
    Classe les candidats LIKE par, dans l'ordre :
      1. présence de macros (préférer une ligne complète à une ligne 'calories seules')
      2. nom exactement égal à la requête
      3. nombre de mots de la requête présents (recouvrement)
      4. moins de mots "en trop", puis nom le plus court (le moins bruité)
    Repli flou (difflib) si aucun mot commun, sinon nom le plus court.
    Évite « red bell pepper » -> « Black Pepper » et « rice » -> ligne sans macros.
    """
    norm = [(str(r["nom"]).lower(), r) for r in rows]
    q_tokens = set(_tokens(q))
    best, best_key = None, None
    for n, r in norm:
        n_tokens = set(_tokens(n))
        overlap = len(q_tokens & n_tokens)
        exact = 1 if n == q else 0
        if overlap == 0 and not exact:
            continue
        key = (_has_macros(r), exact, overlap, -len(n_tokens - q_tokens), -len(n))
        if best_key is None or key > best_key:
            best, best_key = r, key
    if best is not None:
        return best

    close = get_close_matches(q, [n for n, _ in norm], n=1, cutoff=0.6)  # repli flou
    if close:
        return dict(norm)[close[0]]
    return min(norm, key=lambda x: len(x[0]))[1]        # repli : nom le plus court


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
