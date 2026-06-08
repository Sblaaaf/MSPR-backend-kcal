"""
Integration tests for the kcal FastAPI endpoints.
/analyze-image is tested with a mocked HuggingFace call.
"""
import sys
import os
from pathlib import Path
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

IA_PATH = Path(__file__).parent.parent / "ia-kcal"
sys.path.insert(0, str(IA_PATH))
os.chdir(str(IA_PATH))

os.environ.setdefault("KCAL_SECRET_TOKEN", "clesecrete")
os.environ.setdefault("HF_TOKEN", "fake-hf-token")

from main import app

client = TestClient(app)
AUTH = {"Authorization": "Bearer clesecrete"}
BAD_AUTH = {"Authorization": "Bearer wrongtoken"}


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /analyze — text endpoint
# ---------------------------------------------------------------------------

def test_analyze_success():
    resp = client.post("/analyze", json={"text": "200g of chicken and 150g of rice"}, headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert "total_kcal" in body
    assert "items" in body
    assert "message" in body
    assert body["total_kcal"] > 0


def test_analyze_missing_token():
    resp = client.post("/analyze", json={"text": "200g of chicken"})
    assert resp.status_code in (401, 403)


def test_analyze_wrong_token():
    resp = client.post("/analyze", json={"text": "200g of chicken"}, headers=BAD_AUTH)
    assert resp.status_code == 401


def test_analyze_empty_text():
    resp = client.post("/analyze", json={"text": "   "}, headers=AUTH)
    assert resp.status_code == 400


def test_analyze_missing_text_field():
    resp = client.post("/analyze", json={}, headers=AUTH)
    assert resp.status_code == 422


def test_analyze_items_structure():
    resp = client.post("/analyze", json={"text": "100g of salmon"}, headers=AUTH)
    assert resp.status_code == 200
    for item in resp.json()["items"]:
        assert "food" in item
        assert "grams" in item
        assert "kcal" in item


def test_analyze_total_matches_sum():
    resp = client.post("/analyze", json={"text": "100g of chicken and 100g of rice"}, headers=AUTH)
    body = resp.json()
    computed = round(sum(i["kcal"] for i in body["items"]), 1)
    assert body["total_kcal"] == computed


def test_analyze_complex_meal():
    resp = client.post(
        "/analyze",
        json={"text": "266g of rice and chicken and for the dessert i ate an ice cream and 50g of apple"},
        headers=AUTH,
    )
    assert resp.status_code == 200
    assert resp.json()["total_kcal"] > 0


# ---------------------------------------------------------------------------
# /analyze-image — vision endpoint
#
# La reconnaissance est déléguée à vision.recognize() (provider Claude ou
# HuggingFace selon VISION_PROVIDER, avec bascule automatique). On teste donc
# le comportement de la ROUTE en simulant ce point d'entrée unique, ce qui rend
# le test indépendant du provider réellement configuré.
# ---------------------------------------------------------------------------

# Ce que renvoie vision.recognize() : list[{"food", "grams", "confidence"}]
FAKE_DETECTED = [
    {"food": "rice", "grams": 150.0, "confidence": 0.85},
    {"food": "chicken", "grams": 120.0, "confidence": 0.72},
]


@pytest.fixture
def fake_image():
    return BytesIO(b"fake-image-bytes")


def test_analyze_image_success(fake_image):
    with patch("vision.recognize", return_value=FAKE_DETECTED):
        resp = client.post(
            "/analyze-image",
            headers=AUTH,
            files={"file": ("meal.jpg", fake_image, "image/jpeg")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "total_kcal" in body
    assert "items" in body
    assert len(body["items"]) > 0


def test_analyze_image_missing_token(fake_image):
    resp = client.post(
        "/analyze-image",
        files={"file": ("meal.jpg", fake_image, "image/jpeg")},
    )
    assert resp.status_code in (401, 403)


def test_analyze_image_provider_error(fake_image):
    # Le provider de vision lève une exception → 502 Bad Gateway
    with patch("vision.recognize", side_effect=Exception("vision API down")):
        resp = client.post(
            "/analyze-image",
            headers=AUTH,
            files={"file": ("meal.jpg", fake_image, "image/jpeg")},
        )

    assert resp.status_code == 502


def test_analyze_image_no_food_recognized(fake_image):
    # Aucun aliment reconnu sur la photo → 400 Bad Request
    with patch("vision.recognize", return_value=[]):
        resp = client.post(
            "/analyze-image",
            headers=AUTH,
            files={"file": ("meal.jpg", fake_image, "image/jpeg")},
        )

    assert resp.status_code == 400
