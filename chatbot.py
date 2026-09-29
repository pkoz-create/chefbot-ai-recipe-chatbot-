"""
chatbot.py
----------
Interactive command-line recipe chatbot. Loads the model trained by
train_model.py to classify what the user wants (small talk vs. a recipe
request), and for recipe requests, looks up the closest matching dish in
recipes.json and prints out ingredients + steps.

Usage:
    python chatbot.py
"""

import json
import random
import os
import sys
import re
import difflib
import joblib

MODEL_PATH = "chatbot_model.joblib"
INTENTS_CACHE_PATH = "intents_cache.json"
RECIPES_PATH = "recipes.json"

# Confidence threshold below which we treat the message as "unknown".
# Kept low because softmax probability gets spread thin across many classes
# with only a small number of training examples per class.
CONFIDENCE_THRESHOLD = 0.15


def load_resources():
    missing = [p for p in (MODEL_PATH, INTENTS_CACHE_PATH, RECIPES_PATH) if not os.path.exists(p)]
    if MODEL_PATH in missing or INTENTS_CACHE_PATH in missing:
        print("No trained model found. Please run 'python train_model.py' first.")
        sys.exit(1)
    if RECIPES_PATH in missing:
        print(f"Missing {RECIPES_PATH}. Make sure it's in the same folder as chatbot.py.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    with open(INTENTS_CACHE_PATH, "r", encoding="utf-8") as f:
        intents_data = json.load(f)
    responses_by_tag = {
        intent["tag"]: intent["responses"] for intent in intents_data["intents"]
    }

    with open(RECIPES_PATH, "r", encoding="utf-8") as f:
        recipes_data = json.load(f)["recipes"]

    return model, responses_by_tag, recipes_data


def build_lookup_table(recipes_data):
    """Maps every alias/name (lowercased) -> recipe dict, for matching."""
    lookup = {}
    for recipe in recipes_data:
        names = [recipe["name"].lower()] + [a.lower() for a in recipe.get("aliases", [])]
        for n in names:
            lookup[n] = recipe
    return lookup


def find_recipe(user_input, recipes_data, lookup):
    text = user_input.lower()

    # 1. Direct substring match against any known name/alias
    for alias, recipe in lookup.items():
        if alias in text:
            return recipe

    # 2. Fuzzy match: compare each word/phrase in the message against known aliases
    candidates = list(lookup.keys())
    words = re.findall(r"[a-zA-Z]+", text)
    # Try matching against the whole cleaned message first
    close = difflib.get_close_matches(text, candidates, n=1, cutoff=0.5)
    if close:
        return lookup[close[0]]
    # Then try individual words (helps with e.g. "guac" vs "guacamole")
    for w in words:
        close = difflib.get_close_matches(w, candidates, n=1, cutoff=0.6)
        if close:
            return lookup[close[0]]

    return None


def format_recipe(recipe):
    lines = [f"\n=== {recipe['name']} ==="]
    lines.append(f"Servings: {recipe['servings']}  |  Prep time: {recipe['prep_time']}")
    lines.append("\nIngredients:")
    for ing in recipe["ingredients"]:
        lines.append(f"  - {ing}")
    lines.append("\nSteps:")
    for i, step in enumerate(recipe["steps"], 1):
        lines.append(f"  {i}. {step}")
    return "\n".join(lines)


def format_recipe_list(recipes_data):
    names = [r["name"] for r in recipes_data]
    return "Here's what I can make:\n  - " + "\n  - ".join(names) + \
           "\n\nJust ask, e.g. 'recipe for " + names[0].lower() + "'"


def get_response(model, responses_by_tag, recipes_data, lookup, user_input):
    proba = model.predict_proba([user_input])[0]
    classes = model.classes_
    best_idx = proba.argmax()
    best_tag = classes[best_idx]
    confidence = proba[best_idx]

    tag = best_tag if confidence >= CONFIDENCE_THRESHOLD else "unknown"

    # Even if the classifier is unsure, check if the message clearly names a
    # known dish -- that's a strong enough signal to treat it as a recipe request.
    if tag == "unknown":
        maybe_recipe = find_recipe(user_input, recipes_data, lookup)
        if maybe_recipe:
            tag = "get_recipe"

    if tag == "get_recipe":
        recipe = find_recipe(user_input, recipes_data, lookup)
        if recipe:
            return format_recipe(recipe), tag, confidence
        else:
            return ("I don't have that recipe yet. Try 'list recipes' to see what I know!",
                    tag, confidence)

    if tag == "list_recipes":
        return format_recipe_list(recipes_data), tag, confidence

    responses = responses_by_tag.get(tag, responses_by_tag.get("unknown", ["I'm not sure."]))
    return random.choice(responses), tag, confidence


def main():
    model, responses_by_tag, recipes_data = load_resources()
    lookup = build_lookup_table(recipes_data)

    print("=" * 55)
    print(" ChefBot: AI Recipe Assistant ".center(55, "="))
    print("=" * 55)
    print("Ask me for a recipe (e.g. 'how do I make pancakes?')")
    print("or say 'list recipes' to see everything I know.")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! Happy cooking!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("Bot: Goodbye! Happy cooking!")
            break

        response, tag, confidence = get_response(
            model, responses_by_tag, recipes_data, lookup, user_input
        )
        print(f"Bot: {response}\n")
        # Uncomment to debug intent predictions:
        # print(f"     [debug: tag={tag}, confidence={confidence:.2f}]")


if __name__ == "__main__":
    main()
