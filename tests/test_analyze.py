"""
Unit tests for the kcal NLP analysis pipeline.
No network calls — pure logic tests on analyze() and parse().
"""
import sys
from pathlib import Path

# Make ia-kcal importable
IA_PATH = Path(__file__).parent.parent / "ia-kcal"
sys.path.insert(0, str(IA_PATH))

import os  # noqa: E402
os.chdir(str(IA_PATH))

from analyze import analyze, MealResult  # noqa: E402


# ---------------------------------------------------------------------------
# analyze() — basic recognition
# ---------------------------------------------------------------------------

def test_analyze_returns_meal_result():
    result = analyze("200g of chicken")
    assert isinstance(result, MealResult)
    assert isinstance(result.items, list)
    assert isinstance(result.total_kcal, float)
    assert isinstance(result.message, str)


def test_analyze_single_food_with_weight():
    result = analyze("200g of rice")
    assert len(result.items) >= 1
    assert result.total_kcal > 0
    item = result.items[0]
    assert item["food"] != ""
    assert item["grams"] == 200.0
    assert item["kcal"] > 0


def test_analyze_multiple_foods():
    result = analyze("200g of chicken and 150g of rice")
    assert len(result.items) >= 2
    assert result.total_kcal > 0


def test_analyze_total_is_sum_of_items():
    result = analyze("100g of chicken and 100g of rice")
    computed = round(sum(i["kcal"] for i in result.items), 1)
    assert result.total_kcal == computed


def test_analyze_empty_string_returns_zero():
    result = analyze("   ")
    assert result.total_kcal == 0.0
    assert result.items == []


def test_analyze_unknown_food_returns_zero_kcal():
    result = analyze("300g of xyzunknownfood999")
    # Unknown food → 0 kcal/100g → kcal=0 or item absent
    total = result.total_kcal
    assert total == 0.0


def test_analyze_message_contains_kcal():
    result = analyze("100g of chicken")
    assert "kcal" in result.message.lower()


def test_analyze_banana_no_weight_uses_default():
    result = analyze("a banana")
    assert len(result.items) >= 1
    assert result.total_kcal > 0


def test_analyze_egg_portion():
    result = analyze("two eggs")
    assert len(result.items) >= 1
    assert result.total_kcal > 0


def test_analyze_gram_variations():
    r1 = analyze("100g of rice")
    r2 = analyze("100 grams of rice")
    r3 = analyze("100gr of rice")
    assert r1.total_kcal == r2.total_kcal == r3.total_kcal


def test_analyze_kg_conversion():
    # Parser handles integer kg: 1kg chicken = 1000g
    result_g = analyze("1000g of chicken")
    result_kg = analyze("1kg of chicken")
    assert abs(result_g.total_kcal - result_kg.total_kcal) < 1


def test_analyze_case_insensitive():
    r1 = analyze("200g of CHICKEN")
    r2 = analyze("200g of chicken")
    assert r1.total_kcal == r2.total_kcal


def test_analyze_complex_meal():
    result = analyze("266g of rice and chicken and for the dessert i ate an ice cream and 50g of apple")
    assert result.total_kcal > 0
    assert len(result.items) >= 2


def test_analyze_synonym_eggs():
    result = analyze("3 eggs")
    items = [i["food"] for i in result.items]
    assert any("egg" in f for f in items)


def test_analyze_items_have_required_keys():
    result = analyze("150g of salmon")
    for item in result.items:
        assert "food" in item
        assert "grams" in item
        assert "kcal" in item
