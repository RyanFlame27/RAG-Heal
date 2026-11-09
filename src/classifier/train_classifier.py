"""
Train Multi-Class Failure Classifier.

Reads 4-signal features from the synthetic failure dataset and trains a
RandomForest classifier to predict failure types:
  0: CLEAN
  1: RETRIEVAL_FAILURE
  2: GROUNDING_FAILURE
  3: CONFIDENCE_FAILURE
  4: COVERAGE_FAILURE
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

LABEL_MAP = {
    "clean": 0,
    "retrieval_failure": 1,
    "grounding_failure": 2,
    "confidence_failure": 3,
    "coverage_failure": 4,
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def extract_features_and_labels(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature vectors [retrieval_sim, entailment, entropy, coverage] and labels from JSON.
    """
    if not dataset_path.exists():
        print(f"Dataset path {dataset_path} not found. Generating synthetic features for training...")
        return generate_synthetic_feature_dataset()

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    X, y = [], []
    for item in data:
        signals = item.get("signals", {})
        label = item.get("label", "clean")

        if label not in LABEL_MAP:
            continue

        # Extract features with fallback for missing/nan values
        ret_sim = signals.get("retrieval_similarity", 0.5)
        entail = signals.get("entailment_score", 0.5)
        entropy = signals.get("mean_entropy", 0.5)
        if np.isnan(entropy) or np.isinf(entropy):
            entropy = 0.5
        coverage = signals.get("coverage_score", 0.5)

        X.append([ret_sim, entail, entropy, coverage])
        y.append(LABEL_MAP[label])

    return np.array(X), np.array(y)


def generate_synthetic_feature_dataset(num_per_class: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Generate realistic feature vectors matching signal distributions for training."""
    np.random.seed(42)
    X, y = [], []

    for label_name, label_id in LABEL_MAP.items():
        for _ in range(num_per_class):
            if label_name == "clean":
                ret = np.random.uniform(0.55, 0.95)
                ent = np.random.uniform(0.60, 0.98)
                ent_tok = np.random.uniform(0.1, 0.8)
                cov = np.random.uniform(0.55, 0.95)
            elif label_name == "retrieval_failure":
                ret = np.random.uniform(0.05, 0.35)
                ent = np.random.uniform(0.01, 0.30)
                ent_tok = np.random.uniform(0.5, 1.8)
                cov = np.random.uniform(0.40, 0.80)
            elif label_name == "grounding_failure":
                ret = np.random.uniform(0.50, 0.90)
                ent = np.random.uniform(0.0, 0.15)
                ent_tok = np.random.uniform(0.3, 1.2)
                cov = np.random.uniform(0.50, 0.90)
            elif label_name == "confidence_failure":
                ret = np.random.uniform(0.45, 0.85)
                ent = np.random.uniform(0.40, 0.85)
                ent_tok = np.random.uniform(2.1, 4.5)
                cov = np.random.uniform(0.45, 0.85)
            elif label_name == "coverage_failure":
                ret = np.random.uniform(0.10, 0.35)
                ent = np.random.uniform(0.01, 0.25)
                ent_tok = np.random.uniform(1.0, 3.5)
                cov = np.random.uniform(0.05, 0.30)

            X.append([ret, ent, ent_tok, cov])
            y.append(label_id)

    return np.array(X), np.array(y)


def train_failure_classifier(
    dataset_path: Path = config.FAILURE_DATASET_PATH,
    model_save_path: Path = config.DATA_DIR / "failure_classifier.pkl",
) -> RandomForestClassifier:
    """Train, evaluate, and save the failure classifier."""
    print("Extracting features from dataset...")
    X, y = extract_features_and_labels(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    cv_scores = cross_val_score(clf, X, y, cv=5)

    print("\n" + "=" * 60)
    print(" 🎯 FAILURE CLASSIFIER EVALUATION REPORT")
    print("=" * 60)
    print(f"Cross-Validation Accuracy (5-fold): {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print("\nClassification Report:")
    target_names = [REVERSE_LABEL_MAP[i] for i in range(len(LABEL_MAP))]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save model
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump({"model": clf, "label_map": LABEL_MAP, "reverse_map": REVERSE_LABEL_MAP}, f)

    print(f"✅ Trained model saved to: {model_save_path}")
    return clf


if __name__ == "__main__":
    train_failure_classifier()
