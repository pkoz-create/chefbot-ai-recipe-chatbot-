"""
train_model.py
----------------
Trains a scikit-learn text-classification model on intents.json
and saves the trained pipeline to disk (chatbot_model.joblib).

Run this once (or whenever you edit intents.json) before using chatbot.py.
"""

import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_training_data(path="intents.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(pattern)
            labels.append(intent["tag"])
    return texts, labels, data


def train():
    texts, labels, raw_data = load_training_data()

    if len(set(labels)) < 2:
        raise ValueError("Need at least 2 intent classes with patterns to train.")

    # Build a pipeline: TF-IDF vectorizer -> Logistic Regression classifier
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=10))
    ])

    # Split for a quick sanity-check evaluation (small dataset, so this is illustrative)
    if len(texts) >= 15:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
    else:
        X_train, y_train = texts, labels
        X_test, y_test = texts, labels

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Training complete. Sanity-check accuracy: {acc:.2f}\n")
    print(classification_report(y_test, preds, zero_division=0))

    # Refit on ALL data before saving (use every example for the final model)
    pipeline.fit(texts, labels)

    joblib.dump(pipeline, "chatbot_model.joblib")

    # Save the intents data alongside so chatbot.py can look up responses by tag
    with open("intents_cache.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    print("\nModel saved to chatbot_model.joblib")
    print("Run 'python chatbot.py' to start chatting.")


if __name__ == "__main__":
    train()
