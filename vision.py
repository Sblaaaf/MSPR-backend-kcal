"""
Abstraction "fournisseur de vision" pour la reconnaissance d'aliments sur photo.

- huggingface : modèle nateraw/food (gratuit, 1 aliment, sans quantité) — défaut.
- claude      : VLM Claude (multi-aliments + estimation des grammes, JSON structuré).

Sélection via VISION_PROVIDER. Bascule automatique sur HuggingFace si Claude
échoue, n'a pas de clé, ou dépasse un plafond (garde-fou anti-surcoût).

Chaque provider renvoie : list[{"food": str, "grams": float | None, "confidence": float}]
"""
import base64
import io
import json
import os

import httpx

HF_API_URL = "https://router.huggingface.co/hf-inference/models/nateraw/food"

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "aliments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aliment": {"type": "string"},
                    "grammes": {"type": "number"},
                    "confiance": {"type": "number"},
                },
                "required": ["aliment", "grammes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["aliments"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# HuggingFace (gratuit)
# --------------------------------------------------------------------------- #
def recognize_hf(image_bytes: bytes) -> list[dict]:
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}",
        "Content-Type": "image/jpeg",
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(HF_API_URL, headers=headers, content=image_bytes)
        resp.raise_for_status()
        predictions = resp.json()

    labels = [p for p in predictions if p.get("score", 0) > 0.1][:3]
    return [
        {"food": p["label"].replace("_", " "), "grams": None, "confidence": p.get("score", 0.0)}
        for p in labels
    ]


# --------------------------------------------------------------------------- #
# Claude VLM (Phase 2 — activé quand ANTHROPIC_API_KEY est présent)
# --------------------------------------------------------------------------- #
def _resize_jpeg(image_bytes: bytes, max_px: int) -> bytes:
    from PIL import Image  # import paresseux : non requis pour le mode HF

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def recognize_claude(image_bytes: bytes) -> list[dict]:
    import anthropic  # import paresseux : non requis pour le mode HF

    max_px = int(os.getenv("VISION_MAX_IMAGE_PX", "768"))
    model = os.getenv("VISION_MODEL", "claude-haiku-4-5")
    resized = _resize_jpeg(image_bytes, max_px)
    b64 = base64.standard_b64encode(resized).decode("utf-8")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model,
        max_tokens=300,  # garde-fou coût : sortie courte
        output_config={"format": {"type": "json_schema", "schema": VISION_SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Identify each food visible in this meal photo and estimate its "
                            "weight in grams. Use generic English food names, singular, lowercase, "
                            "with no brand names (e.g. 'grilled chicken', 'cucumber', 'white rice'). "
                            "Respond only with the requested JSON."
                        ),
                    },
                ],
            }
        ],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    data = json.loads(text)
    return [
        {
            "food": a["aliment"],
            "grams": float(a.get("grammes")) if a.get("grammes") is not None else None,
            "confidence": float(a.get("confiance", 1.0)),
        }
        for a in data.get("aliments", [])
    ]


# --------------------------------------------------------------------------- #
# Sélecteur + fallback
# --------------------------------------------------------------------------- #
def recognize(image_bytes: bytes) -> list[dict]:
    provider = os.getenv("VISION_PROVIDER", "huggingface").lower()
    if provider == "claude" and os.getenv("ANTHROPIC_API_KEY") and _under_cap():
        try:
            result = recognize_claude(image_bytes)
            _increment_usage()
            if result:
                return result
        except Exception:
            pass  # bascule silencieuse sur HuggingFace
    return recognize_hf(image_bytes)


# --------------------------------------------------------------------------- #
# Garde-fou : plafond mensuel (compteur en mémoire pour la Phase 1 ;
# remplacé par un compteur persistant en base en Phase 2).
# --------------------------------------------------------------------------- #
_usage_count = 0


def _under_cap() -> bool:
    raw = os.getenv("VISION_MONTHLY_CAP")
    if not raw:                       # non défini / vide => illimité
        return True
    try:
        cap = int(raw)
    except ValueError:
        return True
    return _usage_count < cap          # cap=0 => toujours fallback (utile pour tester)


def _increment_usage() -> None:
    global _usage_count
    _usage_count += 1
