"""
Synthetic Failure-Injection Dataset Generator.

Creates a labeled dataset of clean and corrupted QA examples
for validating the 4 failure detection signals.

Failure types:
    1. Retrieval failure — replace context with irrelevant chunks.
    2. Grounding failure — use context that doesn't support the answer.
    3. Confidence failure — use ambiguous/truncated context.
    4. Coverage failure — use questions with no KB coverage.

Source: Natural Questions (via HuggingFace datasets).
"""

import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def load_natural_questions(
    num_examples: int = config.NUM_QA_EXAMPLES,
) -> List[Dict[str, Any]]:
    """
    Load QA pairs from HuggingFace dataset or synthetic fallback.
    """
    print(f"Loading QA dataset ({num_examples} examples)...")
    examples = []
    
    # Try SQuAD (Parquet format, ultra-stable)
    try:
        dataset = load_dataset("rajpurkar/squad", split=f"train[:{num_examples * 2}]")
        for row in dataset:
            q = row.get("question", "")
            answers = row.get("answers", {}).get("text", [])
            a = answers[0] if answers else ""
            if q.strip() and a.strip():
                examples.append({"question": q, "answer": a})
            if len(examples) >= num_examples:
                break
    except Exception as e:
        print(f"  HF dataset load skipped: {e}")

    # Fallback to high-quality synthetic template QA generator
    if len(examples) < num_examples:
        needed = num_examples - len(examples)
        print(f"  Generating {needed} synthetic QA pairs as backup...")
        sample_topics = [
            ("What is the capital of France?", "Paris"),
            ("Who invented the telephone?", "Alexander Graham Bell"),
            ("What is the chemical symbol for gold?", "Au"),
            ("Which planet is known as the Red Planet?", "Mars"),
            ("What is the largest ocean on Earth?", "Pacific Ocean"),
            ("Who wrote Romeo and Juliet?", "William Shakespeare"),
            ("What is the speed of light?", "299,792,458 meters per second"),
            ("What process do plants use to convert sunlight into food?", "Photosynthesis"),
            ("What is the hardest natural substance on Earth?", "Diamond"),
            ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ]
        for i in range(needed):
            q, a = sample_topics[i % len(sample_topics)]
            examples.append({"question": f"{q} (ID #{i+1})", "answer": a})

    print(f"Loaded {len(examples)} QA pairs.")
    return examples


def load_context_pool(
    num_articles: int = 1000,
) -> List[str]:
    """
    Load a pool of text chunks for corruption.
    """
    print(f"Loading context pool ({num_articles} Wikipedia articles)...")
    context_pool = []
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split=f"train[:{num_articles}]",
        )
        for row in dataset:
            text = row.get("text", "")
            if len(text.strip()) >= 200:
                context_pool.append(text[:300])
    except Exception as e:
        print(f"  Wikipedia context pool notice: using synthetic pool ({e})")

    if not context_pool:
        context_pool = [
            f"Article #{i}: The history of science and technology encompasses the development of science from ancient times to the present day."
            for i in range(200)
        ]

    print(f"Context pool: {len(context_pool)} chunks.")
    return context_pool


def generate_clean_example(
    qa: Dict[str, str],
    context_pool: List[str],
) -> Dict[str, Any]:
    """Generate a clean (non-corrupted) example with plausible context."""
    # Use a context chunk that contains or relates to the answer
    # Since we don't have true gold context, we synthesize it
    synthetic_context = (
        f"According to reliable sources, the answer to the question "
        f"'{qa['question']}' is {qa['answer']}. "
        f"This is well-documented and widely accepted."
    )
    return {
        "query": qa["question"],
        "context_chunks": [synthetic_context],
        "answer": qa["answer"],
        "label": "clean",
    }


def generate_retrieval_failure(
    qa: Dict[str, str],
    context_pool: List[str],
) -> Dict[str, Any]:
    """
    Retrieval failure: swap in completely irrelevant context.
    The retrieved chunks have nothing to do with the query.
    """
    # Pick random irrelevant chunks
    irrelevant_chunks = random.sample(context_pool, min(3, len(context_pool)))
    return {
        "query": qa["question"],
        "context_chunks": irrelevant_chunks,
        "answer": qa["answer"],
        "label": "retrieval_failure",
    }


def generate_grounding_failure(
    qa: Dict[str, str],
    context_pool: List[str],
) -> Dict[str, Any]:
    """
    Grounding failure: context is topically adjacent but doesn't
    actually support the answer. The answer is fabricated.
    """
    # Use a context that mentions the topic but gives wrong info
    wrong_answer = random.choice([
        "Tokyo", "Berlin", "London", "Sydney", "Cairo",
        "1776", "1945", "2001", "1492", "1969",
        "Albert Einstein", "Marie Curie", "Isaac Newton",
        "the Amazon River", "the Pacific Ocean",
    ])

    misleading_context = (
        f"Many sources discuss {qa['question'].rstrip('?')}. "
        f"However, the details are often debated and there is "
        f"significant disagreement among experts on this topic."
    )

    return {
        "query": qa["question"],
        "context_chunks": [misleading_context],
        "answer": wrong_answer,  # Forced wrong answer
        "label": "grounding_failure",
    }


def generate_confidence_failure(
    qa: Dict[str, str],
    context_pool: List[str],
) -> Dict[str, Any]:
    """
    Confidence failure: ambiguous or truncated context that should
    produce uncertain/hedging model output.
    """
    ambiguous_context = (
        f"There are multiple perspectives on this topic. "
        f"Some sources suggest one answer while others disagree. "
        f"The evidence is inconclusive and further research is needed. "
        f"It could be {qa['answer']} but it might also be something else entirely."
    )

    return {
        "query": qa["question"],
        "context_chunks": [ambiguous_context[:150]],  # Truncated for extra ambiguity
        "answer": f"It is unclear, but possibly {qa['answer']}",
        "label": "confidence_failure",
    }


def generate_coverage_failure(
    context_pool: List[str],
) -> Dict[str, Any]:
    """
    Coverage failure: query about a topic not in the KB at all.
    Uses obscure/invented queries that won't match any KB content.
    """
    obscure_queries = [
        "What is the melting point of Unobtanium in the 2154 periodic table?",
        "Who won the 2087 Intergalactic Chess Championship?",
        "What is the population of the underwater city of Rapture?",
        "How many moons does planet Kepler-442b have according to the 2050 survey?",
        "What is the GDP of Wakanda in current US dollars?",
        "Who invented the flux capacitor and when?",
        "What is the chemical formula for Vibranium?",
        "How tall is the Babel Tower on Mars Colony Alpha?",
        "What programming language does the Matrix simulation use?",
        "What is the speed limit on the Rainbow Bridge of Asgard?",
        "Who was the first person to teleport in 2099?",
        "What is the exchange rate between Galleons and US dollars?",
        "How many dragons are registered in the Welsh Dragon Preserve?",
        "What is the top speed of a 2077 Quadra V-Tech?",
        "Who holds the record for fastest lightsaber duel?",
    ]

    query = random.choice(obscure_queries)
    return {
        "query": query,
        "context_chunks": random.sample(context_pool, min(2, len(context_pool))),
        "answer": "I don't have enough information to answer this question.",
        "label": "coverage_failure",
    }


def generate_failure_dataset(
    num_per_class: int = config.NUM_QA_EXAMPLES,
    output_path: Optional[Path] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate the full failure-injection dataset.

    Creates num_per_class examples for each of the 5 classes:
    clean, retrieval_failure, grounding_failure, confidence_failure, coverage_failure.

    Args:
        num_per_class: Number of examples per failure type.
        output_path: Path to save the dataset JSON.
        seed: Random seed for reproducibility.

    Returns:
        List of labeled examples.
    """
    random.seed(seed)

    config.ensure_dirs()

    if output_path is None:
        output_path = config.FAILURE_DATASET_PATH

    # Load source data
    qa_pairs = load_natural_questions(num_examples=num_per_class)
    context_pool = load_context_pool()

    if len(qa_pairs) < num_per_class:
        print(f"Warning: only {len(qa_pairs)} QA pairs available, adjusting num_per_class.")
        num_per_class = len(qa_pairs)

    dataset = []

    print(f"\nGenerating {num_per_class} examples per class (5 classes)...")

    # Clean examples
    for qa in tqdm(qa_pairs[:num_per_class], desc="Clean"):
        dataset.append(generate_clean_example(qa, context_pool))

    # Retrieval failures
    for qa in tqdm(qa_pairs[:num_per_class], desc="Retrieval failure"):
        dataset.append(generate_retrieval_failure(qa, context_pool))

    # Grounding failures
    for qa in tqdm(qa_pairs[:num_per_class], desc="Grounding failure"):
        dataset.append(generate_grounding_failure(qa, context_pool))

    # Confidence failures
    for qa in tqdm(qa_pairs[:num_per_class], desc="Confidence failure"):
        dataset.append(generate_confidence_failure(qa, context_pool))

    # Coverage failures
    for _ in tqdm(range(num_per_class), desc="Coverage failure"):
        dataset.append(generate_coverage_failure(context_pool))

    # Shuffle
    random.shuffle(dataset)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Summary
    label_counts = {}
    for ex in dataset:
        label_counts[ex["label"]] = label_counts.get(ex["label"], 0) + 1

    print(f"\nDataset saved to {output_path}")
    print(f"Total examples: {len(dataset)}")
    print(f"Label distribution: {label_counts}")

    return dataset


if __name__ == "__main__":
    dataset = generate_failure_dataset()

    # Show a sample from each class
    seen_labels = set()
    print("\n" + "=" * 60)
    print("SAMPLE EXAMPLES:")
    print("=" * 60)
    for ex in dataset:
        if ex["label"] not in seen_labels:
            seen_labels.add(ex["label"])
            print(f"\n[{ex['label'].upper()}]")
            print(f"  Query:   {ex['query'][:80]}...")
            print(f"  Answer:  {ex['answer'][:80]}...")
            print(f"  Context: {ex['context_chunks'][0][:80]}...")
        if len(seen_labels) == 5:
            break
