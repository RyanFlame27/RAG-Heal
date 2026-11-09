"""
Benchmark Evaluation Script: Vanilla RAG vs. Self-Healing RAG.

Runs a small diagnostic query suite; it is not a balanced performance benchmark.
Compares initial vs. healed metrics and computes net improvement statistics.
"""

import math
import json
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


BENCHMARK_PROMPTS = [
    # Clean Cases
    {"category": "clean", "query": "What is photosynthesis and how do plants convert sunlight into energy?"},
    {"category": "clean", "query": "How does gravity work according to general relativity?"},
    {"category": "clean", "query": "Who was Albert Einstein and what is he best known for?"},
    {"category": "clean", "query": "What caused the fall of the Western Roman Empire?"},

    # Grounding / Hallucination Cases
    {"category": "grounding_failure", "query": "Why did the Eiffel Tower collapse in 2019?"},
    {"category": "grounding_failure", "query": "Why was oxygen removed from the periodic table in 2018?"},
    {"category": "grounding_failure", "query": "How many people died when the Great Wall of China collapsed last year?"},
    {"category": "grounding_failure", "query": "Why did Japan ban the use of electricity in 2021?"},

    # Coverage / Out-of-Scope Cases
    {"category": "coverage_failure", "query": "What is the wifi password for Starbucks in Mumbai airport?"},
    {"category": "coverage_failure", "query": "How many Infinity Stones did Thanos collect on Planet Xandar?"},
    {"category": "coverage_failure", "query": "How do I fix error code 0x80070057 on Windows 11?"},
    {"category": "coverage_failure", "query": "What were the top 10 trending TikTok sounds yesterday?"},

    # Gibberish is an unanswerable/coverage case, not a retrieval injection.
    {"category": "coverage_failure", "query": "qwzxy7890 vbnm 9988 zxcvbnm 123456789 qwertyuiop zzzzzzzz"},
    {"category": "coverage_failure", "query": "zzzxxxyyyaaabbb 77788899 jjjkkklll mmmnnnooo"},
]


def run_benchmark_evaluation(pipeline, classifier, recovery_matrix, cases=None):
    """Compare identical signal-policy criteria, not factual-answer accuracy."""
    from src.detection.response_check import assess_response
    import time

    cases = BENCHMARK_PROMPTS if cases is None else cases
    records, details = [], []
    out_dir = config.VALIDATION_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in cases:
        query = item["query"]
        allow_refusal = item.get("allow_refusal", item["category"] != "clean")
        started = time.perf_counter()
        print(f"[Test] {item['category']}: {query}")
        try:
            initial = pipeline.run_rag_with_detection(query, **item.get("pipeline_kwargs", {}))
            prediction = classifier.predict(initial["signals"], query=query,
                                            contexts=initial["context_texts"], answer=initial["answer"])
            recovered = recovery_matrix.heal(query, initial, prediction["predicted_label"])
            before = assess_response(initial["answer"], initial["signals"], allow_refusal)
            after = assess_response(recovered["healed_answer"], recovered["post_signals"], allow_refusal)
            record = {
                "category": item["category"], "query": query,
                "predicted_failure": prediction["predicted_label"],
                "action_taken": recovered["action_taken"],
                "baseline_accepted": before["accepted"], "recovered_accepted": after["accepted"],
                "baseline_grounded": before["grounded_answer"], "recovered_grounded": after["grounded_answer"],
                "baseline_refusal": before["refusal"], "recovered_refusal": after["refusal"],
                "baseline_entropy": initial["signals"].get("mean_entropy"),
                "recovered_entropy": recovered["post_signals"].get("mean_entropy"),
                "baseline_grounding": initial["signals"]["entailment_score"],
                "recovered_grounding": recovered["post_signals"]["entailment_score"],
                "elapsed_seconds": time.perf_counter() - started, "error": "",
            }
            details.append({"case": item, "baseline_answer": initial["answer"],
                            "baseline_contexts": initial["context_texts"],
                            "baseline_signals": initial["signals"], "recovery": recovered,
                            "baseline_assessment": before, "recovered_assessment": after})
        except Exception as error:
            record = {"category": item["category"], "query": query,
                      "baseline_accepted": False, "recovered_accepted": False,
                      "baseline_grounded": False, "recovered_grounded": False,
                      "baseline_refusal": False, "recovered_refusal": False,
                      "error": str(error), "elapsed_seconds": time.perf_counter() - started}
            details.append({"case": item, "error": str(error)})
            print(f"  ERROR: {error}")
        records.append(record)
        pd.DataFrame(records).to_csv(out_dir / "comparative_evaluation_report.csv", index=False)
        with (out_dir / "comparative_details.json").open("w", encoding="utf-8") as stream:
            json.dump(details, stream, indent=2, ensure_ascii=False, allow_nan=False)

    df = pd.DataFrame(records)
    print("Automated policy acceptance; not independently judged factual accuracy.")
    for side in ["baseline", "recovered"]:
        print(f"{side}: {int(df[f'{side}_accepted'].sum())}/{len(df)} accepted; "
              f"{int(df[f'{side}_grounded'].sum())} grounded; "
              f"{int(df[f'{side}_refusal'].sum())} abstentions")
    print(f"Execution errors: {int(df['error'].ne('').sum())}")
    return df


if __name__ == "__main__":
    from src.pipeline.rag import RAGPipeline
    from src.classifier.predict import FailureClassifier
    from src.recovery.recovery_matrix import RecoveryMatrix
    pipeline = RAGPipeline()
    run_benchmark_evaluation(pipeline, FailureClassifier(), RecoveryMatrix(pipeline))
