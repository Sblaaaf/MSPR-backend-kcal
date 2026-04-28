"""
Train spaCy NER model — FOOD and QUANTITY entities.
Combines:
  1. Auto-generated examples from kaggle_nutrition.csv
  2. Hand-annotated sentences from data/training_sentences.json

Run: python nlp/train_ner.py
"""

import sys, re, random, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from data.nutrition_data import FOOD_DB
from nlp.parser import SYNONYMS

OUTPUT_DIR   = Path(__file__).parent / "food_ner_model"
MANUAL_FILE  = Path(__file__).parent.parent / "data" / "training_sentences.json"

FOODS = sorted(FOOD_DB.keys(), key=len, reverse=True)

# ── Template bank ─────────────────────────────────────────────────────────────
TEMPLATES_SINGLE = [
    # qty + unit + food
    "200g of {food}", "150g {food}", "100 grams of {food}", "300g {food}",
    "2 cups of {food}", "a bowl of {food}", "a glass of {food}",
    "a slice of {food}", "2 slices of {food}", "a handful of {food}",
    "half a cup of {food}", "3 tbsp of {food}",
    # number + food
    "2 {food}", "3 {food}", "4 {food}", "one {food}", "two {food}", "three {food}",
    "a {food}", "an {food}",
    # food alone / with context
    "{food}", "some {food}", "grilled {food}", "baked {food}", "steamed {food}",
    "fried {food}", "roasted {food}", "boiled {food}", "fresh {food}",
    "ate {food}", "had {food}", "just {food}", "ate some {food}",
    "had some {food}", "snacked on {food}",
    # with meal context
    "{food} for breakfast", "{food} for lunch", "{food} for dinner",
    "morning {food}", "ate {food} today",
]

TEMPLATES_MULTI = [
    "{food} and {food2}",
    "{food} with {food2}",
    "2 {food} and {food2}",
    "200g {food} with {food2}",
    "{food} and {food2} and {food3}",
    "{food} with {food2} and {food3}",
    "ate {food} with {food2}",
    "had {food} and {food2} for lunch",
    "just ate {food} and {food2}",
    "{food} with some {food2}",
    "i had {food} with {food2} and {food3}",
    "ate 200g of {food} with {food2}",
    "2 {food} with {food2} and a {food3}",
    "{food} and {food2} for dinner",
    "breakfast was {food} and {food2}",
    "lunch was {food} with {food2}",
    "dinner was {food} and {food2} and {food3}",
]

# Typo variants to teach robustness
TYPO_VARIANTS = {
    "chicken": ["chiken", "chikken", "chickin"],
    "salmon":  ["samon", "salman", "slamon"],
    "banana":  ["bananaa", "bannana"],
    "broccoli":["brocoli", "brocolli"],
    "potato":  ["potatoe", "potatos"],
    "tomato":  ["tomatoe", "tomatos"],
    "carrot":  ["carot", "carrott"],
    "lettuce": ["letuce", "lettus"],
    "yogurt":  ["yoghurt", "yogourt"],
    "quinoa":  ["kinoa", "quinua"],
}


def _clean_spans(ents, text):
    ents = sorted(set(ents), key=lambda x: (x[0], -(x[1]-x[0])))
    result, last_end = [], -1
    for start, end, label in ents:
        if start >= last_end and end <= len(text) and start < end:
            result.append((start, end, label))
            last_end = end
    return result


def _make_entity_spans(text, food, qty_pattern=None):
    """Find all occurrences of food and optional qty in text."""
    ents = []
    idx = text.find(food)
    if idx != -1:
        ents.append((idx, idx + len(food), "FOOD"))
    if qty_pattern:
        for m in re.finditer(r'\b' + re.escape(qty_pattern) + r'\b', text):
            ents.append((m.start(), m.end(), "QUANTITY"))
    return ents


def generate_auto_data() -> list[tuple]:
    data = []
    foods = FOODS[:80]  # top 80 foods

    for food in foods:
        # Single food templates
        for tmpl in TEMPLATES_SINGLE:
            text = tmpl.replace("{food}", food)
            ents = []
            # Find food span
            idx = text.find(food)
            if idx != -1:
                ents.append((idx, idx + len(food), "FOOD"))
            # Find quantity spans
            for m in re.finditer(r'\b(\d+g|\d+\s*grams?|\d+\s*cups?|\d+\s*slices?|\d+\s*tbsp|\d+)\b', text):
                ents.append((m.start(), m.end(), "QUANTITY"))
            for word in ["a", "an", "two", "three", "half", "one", "handful", "bowl", "glass", "cup", "slice"]:
                for m in re.finditer(r'\b' + word + r'\b', text):
                    ents.append((m.start(), m.end(), "QUANTITY"))
            ents = _clean_spans(ents, text)
            if ents:
                data.append((text, {"entities": ents}))

        # Multi-food templates
        other_foods = [f for f in foods if f != food]
        for tmpl in TEMPLATES_MULTI:
            food2 = random.choice(other_foods)
            food3 = random.choice([f for f in other_foods if f != food2] or [food2])
            text = tmpl.replace("{food}", food).replace("{food2}", food2).replace("{food3}", food3)
            ents = []
            for target in [food, food2, food3]:
                start = 0
                while True:
                    idx = text.find(target, start)
                    if idx == -1: break
                    ents.append((idx, idx + len(target), "FOOD"))
                    start = idx + 1
            for m in re.finditer(r'\b(\d+g|\d+\s*grams?|\d+)\b', text):
                ents.append((m.start(), m.end(), "QUANTITY"))
            for word in ["a", "an", "two", "three", "half", "one"]:
                for m in re.finditer(r'\b' + word + r'\b', text):
                    ents.append((m.start(), m.end(), "QUANTITY"))
            ents = _clean_spans(ents, text)
            if ents:
                data.append((text, {"entities": ents}))

        # Typo variants
        typos = TYPO_VARIANTS.get(food, [])
        for typo in typos:
            text = f"had {typo} for dinner"
            ents = [(4, 4 + len(typo), "FOOD")]
            data.append((text, {"entities": ents}))

    # Synonym variants
    for syn, canonical in SYNONYMS.items():
        text = f"ate some {syn}"
        ents = [(9, 9 + len(syn), "FOOD")]
        data.append((text, {"entities": ents}))

    return data


def load_manual_data() -> list[tuple]:
    """Load hand-annotated sentences from JSON file."""
    if not MANUAL_FILE.exists():
        print(f"[train_ner] No manual file found at {MANUAL_FILE}")
        return []

    with open(MANUAL_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    data = []
    for item in raw:
        text = item["text"].lower()
        ents = []
        for surface, label in item["entities"]:
            surface = surface.lower()
            idx = text.find(surface)
            if idx != -1:
                ents.append((idx, idx + len(surface), label))
        ents = _clean_spans(ents, text)
        if ents:
            data.append((text, {"entities": ents}))

    print(f"[train_ner] Loaded {len(data)} manual sentences from {MANUAL_FILE.name}")
    return data


def train():
    print("[train_ner] Generating auto training data...")
    auto_data   = generate_auto_data()
    manual_data = load_manual_data()

    # Manual data repeated 3x — higher weight on real sentences
    all_data = auto_data + manual_data * 3
    random.shuffle(all_data)
    print(f"[train_ner] Total: {len(all_data)} examples ({len(auto_data)} auto + {len(manual_data)*3} manual×3)")

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    ner.add_label("FOOD")
    ner.add_label("QUANTITY")

    split     = int(len(all_data) * 0.9)
    train_set = all_data[:split]
    dev_set   = all_data[split:]

    optimizer = nlp.begin_training()

    print(f"[train_ner] Training on {len(train_set)} examples ({len(dev_set)} dev)...")
    best_loss = float("inf")
    for epoch in range(40):
        random.shuffle(train_set)
        losses = {}
        for batch in minibatch(train_set, size=compounding(8.0, 64.0, 1.001)):
            examples = []
            for text, annotations in batch:
                try:
                    examples.append(Example.from_dict(nlp.make_doc(text), annotations))
                except Exception:
                    continue
            nlp.update(examples, drop=0.25, losses=losses, sgd=optimizer)

        loss = losses.get("ner", 0)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}/40 — NER loss: {loss:.3f}")
        if loss < best_loss:
            best_loss = loss

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(OUTPUT_DIR)
    print(f"\n[train_ner] Model saved → {OUTPUT_DIR}")
    print(f"[train_ner] Best loss: {best_loss:.3f}")

    # Eval on dev
    correct = total = 0
    for text, annotations in dev_set[:100]:
        doc = nlp(text)
        predicted = {(e.start_char, e.end_char, e.label_) for e in doc.ents}
        expected  = {(s, e, l) for s, e, l in annotations["entities"]}
        if predicted & expected:
            correct += 1
        total += 1
    print(f"[train_ner] Dev accuracy: {correct}/{total} = {correct/max(1,total):.0%}")

    # Quick live test
    print("\n[train_ner] Quick test:")
    tests = [
        "2 eggs and a banana",
        "200g chicken with brown rice",
        "pizza and fries",
        "had chiken and rice",
        "steak and salad",
    ]
    for t in tests:
        doc = nlp(t)
        ents = [(e.text, e.label_) for e in doc.ents]
        print(f"  '{t}' → {ents}")


if __name__ == "__main__":
    train()