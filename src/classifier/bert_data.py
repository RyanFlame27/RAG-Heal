"""Shared BERT inputs and leakage-resistant splits; no ML dependencies."""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict

LABELS = ("clean", "retrieval_failure", "grounding_failure",
          "confidence_failure", "coverage_failure")
LABEL_MAP = {name: i for i, name in enumerate(LABELS)}
INPUT_SCHEMA = "question-answer-context-v1"


def question_key(query):
    return re.sub(r"\s+", " ", query.strip().casefold())


def classifier_text(query, contexts, answer):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("BERT needs the actual query, not signals alone.")
    if not isinstance(answer, str):
        raise ValueError("BERT needs the actual answer text.")
    if not isinstance(contexts, (list, tuple)) or not all(isinstance(c, str) for c in contexts):
        raise ValueError("BERT needs the retrieved context texts.")
    # Keep the answer before lengthy contexts so truncation retains the claim.
    # Neither dataset labels nor label-derived numeric features appear here.
    return f"Question: {query}\nAnswer: {answer}\nContext: " + "\n".join(contexts)


def dataset_hash(data):
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def grouped_split(data, seed=42):
    groups = defaultdict(list)
    for i, row in enumerate(data):
        if row.get("label") not in LABEL_MAP:
            raise ValueError(f"Unknown label at row {i}.")
        classifier_text(row["query"], row["context_chunks"], row["answer"])
        groups[question_key(row["query"])].append(i)
    # Group by question first, then balance the two generator families:
    # QA questions with clean/corrupted variants, and coverage-only questions.
    families = defaultdict(list)
    for key, indices in groups.items():
        family = tuple(sorted({data[i]["label"] for i in indices}))
        families[family].append(key)
    split = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for family in sorted(families):
        keys = sorted(families[family])
        if len(keys) < 3:
            raise ValueError("Each question family needs at least 3 distinct questions for a grouped split.")
        rng.shuffle(keys)
        n_validation = max(1, round(len(keys) * 0.15))
        n_test = max(1, round(len(keys) * 0.15))
        partitions = {"validation": keys[:n_validation],
                      "test": keys[n_validation:n_validation + n_test],
                      "train": keys[n_validation + n_test:]}
        for name, selected in partitions.items():
            split[name].extend(i for key in selected for i in groups[key])
    for name in split:
        split[name].sort()
        present = {data[i]["label"] for i in split[name]}
        if present != set(LABELS):
            raise ValueError(f"{name} lacks a class; cannot report a five-class evaluation.")
    return {"seed": seed, "dataset_sha256": dataset_hash(data),
            "indices": split,
            "class_counts": {name: dict(Counter(data[i]["label"] for i in ids))
                             for name, ids in split.items()},
            "grouping": "normalised question; all corruption variants stay together"}
