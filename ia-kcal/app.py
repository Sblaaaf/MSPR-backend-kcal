"""
Run: python app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from analyze import analyze
from nlp.parser import parser_info

TESTS = [
    "2 eggs and a banana",
    "200g of grilled chicken with brown rice and broccoli",
    "pizza and french fries with a soda",
    "a yogurt and an apple",
    "three scrambled eggs with whole wheat bread",
    "150g of salmon with spinach",
    "a hamburger with fries",
    "oats with milk and a banana",
    "i had chiken and rise",          # typos
    "had some salad and coffe",       # typos
]

def show(result):
    if not result.items:
        print("  ❌ Nothing recognized\n")
        return
    for item in result.items:
        print(f"  • {item['food']:<25} {item['grams']:>6.0f}g  →  {item['kcal']:.0f} kcal")
    print(f"  {'TOTAL':<25} {'':>9}  {result.total_kcal:.0f} kcal\n")

def interactive():
    print("=" * 52)
    print(f"  HealthAI — Calorie Counter")
    print(f"  Parser: {parser_info()}")
    print("  Type 'quit' to exit, 'test' for examples")
    print("=" * 52 + "\n")
    while True:
        try:
            meal = input("🍽️  What did you eat? ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!"); break
        if not meal: continue
        if meal.lower() in ("quit","q","exit"): print("Bye!"); break
        if meal.lower() == "test":
            for t in TESTS:
                print(f"\n📝 {t}")
                show(analyze(t))
            continue
        result = analyze(meal)
        print()
        show(result)
        print(f"  💬 {result.message}\n")

if __name__ == "__main__":
    if "--test" in sys.argv:
        print(f"Parser: {parser_info()}\n")
        for t in TESTS:
            print(f"📝 {t}")
            show(analyze(t))
    else:
        interactive()
