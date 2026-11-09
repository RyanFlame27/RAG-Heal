"""Allowlisted faulty starting states for a live, explicitly injected demo.

These are stored corrupted answer/context triples, not newly generated baseline
answers. No reference answer or recorded recovery output is provided to inference.
"""
import json

import config

CASE_SPECS = (
    ("publication-year", 123, "Wrong answer: publication year"),
    ("primary-seminary", 537, "Wrong answer: primary seminary"),
    ("college-president", 168, "Wrong answer: college president"),
)


def demo_cases():
    rows = json.loads((config.BENCHMARK_DIR / "failure_dataset.json").read_text(encoding="utf-8"))
    cases = []
    for case_id, index, title in CASE_SPECS:
        stored = rows[index]
        # Only the corrupted state is loaded. Do not look up clean siblings,
        # reference answers, saved final responses or expected classifier labels.
        cases.append({"case_id": case_id, "title": title, "sample_index": index,
                      "query": stored["query"], "injected_answer": stored["answer"],
                      "injected_contexts": list(stored["context_chunks"]),
                      "fault_description": "A stored wrong answer and non-answering context are supplied before BERT diagnosis.",
                      "source": "Stored corrupted triple from the project's failure-injected dataset"})
    return cases


def get_demo_case(case_id):
    return next((case for case in demo_cases() if case["case_id"] == case_id), None)


def injected_initial(case, matrix):
    contexts = list(case["injected_contexts"])
    answer = case["injected_answer"]
    # This answer was supplied, not sampled from Mistral in this request. Token
    # entropy therefore has no measurement; never invent or reuse logits.
    signals = matrix._audit(case["query"], answer, contexts, token_logits=None)
    signals["mean_entropy"] = None
    return {"answer": answer, "context_texts": contexts, "signals": signals,
            "retrieved_chunks": [{"text": text, "rank": rank, "score": None,
                                  "metadata": {"source": "injected_demo_context"}}
                                 for rank, text in enumerate(contexts)]}
