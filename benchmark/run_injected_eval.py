"""Measure stored-answer detection and live recovery on the injected dataset.

The stored wrong answer and a freshly generated answer are different test objects.
Keep both. Injection labels describe constructed conditions, not independently
verified errors in the freshly generated response. No signal is simulated here.
"""

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
import random
import sys
import time

LABELS = ("clean", "retrieval_failure", "grounding_failure",
          "confidence_failure", "coverage_failure")
SCHEMA = 1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def finite_signals(signals):
    """Unavailable values remain null; they never turn into passing zeroes."""
    result = {}
    for key, value in signals.items():
        if key == "is_refusal":
            result[key] = bool(value)
        elif value is None:
            result[key] = None
        else:
            number = float(value)
            result[key] = number if math.isfinite(number) else None
    return result


def threshold_flag(value, threshold, higher=False):
    if value is None or not math.isfinite(value):
        return None
    return value >= threshold if higher else value < threshold


def load_dataset(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Dataset must be a nonempty JSON list.")
    for i, item in enumerate(data):
        if item.get("label") not in LABELS:
            raise ValueError(f"Unknown label at sample {i}.")
        if not isinstance(item.get("query"), str) or not item["query"].strip():
            raise ValueError(f"Missing query at sample {i}.")
        if not isinstance(item.get("answer"), str):
            raise ValueError(f"Missing stored answer at sample {i}.")
        contexts = item.get("context_chunks")
        if not isinstance(contexts, list) or not contexts or not all(
                isinstance(text, str) for text in contexts):
            raise ValueError(f"Invalid injected contexts at sample {i}.")
    return data


def selected_sample_indices(size, limit=None, sample_indices=None):
    """Select original dataset IDs without renumbering or changing the split."""
    if sample_indices is None:
        indices = list(range(size)) if limit is None else list(range(size))[:limit]
    else:
        if limit is not None:
            raise ValueError("Use either limit or sample_indices, not both.")
        indices = list(sample_indices)
        if any(type(i) is not int or i < 0 or i >= size for i in indices):
            raise ValueError("Sample indices must be valid original dataset IDs.")
        if len(set(indices)) != len(indices):
            raise ValueError("Sample indices must be unique.")
    if not indices:
        raise ValueError("Select at least one sample.")
    return sorted(indices)


def detector_counts(records, injected_label, flag_key):
    """Compare one constructed failure condition with clean examples only."""
    counts = dict(true_positive=0, false_negative=0, false_positive=0,
                  true_negative=0, unavailable=0)
    for row in records:
        if row.get("error") or row["injected_label"] not in ("clean", injected_label):
            continue
        flag = row["detector_flags"][flag_key]
        if flag is None:
            counts["unavailable"] += 1
            continue
        injected = row["injected_label"] == injected_label
        key = ("true_positive" if flag else "false_negative") if injected else (
            "false_positive" if flag else "true_negative")
        counts[key] += 1
    tp, fn = counts["true_positive"], counts["false_negative"]
    fp, tn = counts["false_positive"], counts["true_negative"]
    counts["recall_on_constructed_condition"] = tp / (tp + fn) if tp + fn else None
    counts["clean_false_alarm_rate"] = fp / (fp + tn) if fp + tn else None
    return counts


def summarise(records, expected):
    successful = [r for r in records if not r.get("error")]
    confusion = {actual: {pred: 0 for pred in LABELS} for actual in LABELS}
    for row in successful:
        confusion[row["injected_label"]][row["predicted_label"]] += 1
    result = {
        "expected_samples": expected, "processed_samples": len(records),
        "successful_samples": len(successful),
        "execution_errors": len(records) - len(successful),
        "all_samples_completed_without_error": len(successful) == expected,
        "processed_class_counts": dict(Counter(r["injected_label"] for r in records)),
        "live_initial_outcomes": dict(Counter(r["initial_assessment"]["outcome"] for r in successful)),
        "live_final_outcomes": dict(Counter(r["final_assessment"]["outcome"] for r in successful)),
        "injection_label_vs_live_route": confusion,
        "detectors": {},
        "interpretation": [
            "Grounding uses the stored dataset answer, preserving the injected wrong answer.",
            "Retrieval and coverage use real embeddings and the actual FAISS index.",
            "Confidence uses raw-logit entropy from the newly generated answer, not the stored answer.",
            "Injection labels do not prove a live answer is wrong or uncertain; route agreement is not factual accuracy.",
            "Ambiguous text need not cause high entropy. Clean constructed context need not have index coverage.",
            "Refusals are counted separately; no aggregate refusal-as-success score is reported.",
        ],
    }
    for label, key in (("retrieval_failure", "retrieval"),
                       ("grounding_failure", "stored_grounding"),
                       ("confidence_failure", "generated_confidence"),
                       ("coverage_failure", "coverage")):
        result["detectors"][label] = detector_counts(records, label, key)
    held_out = [r for r in successful if r.get("classifier_split") == "test"]
    stored_confusion = {actual: {pred: 0 for pred in LABELS} for actual in LABELS}
    for row in held_out:
        stored_confusion[row["injected_label"]][row["stored_answer_prediction"]["predicted_label"]] += 1
    result["bert_stored_answer_held_out_test"] = {
        "samples": len(held_out), "confusion_matrix": stored_confusion,
        "label_accuracy": sum(stored_confusion[label][label] for label in LABELS) / len(held_out) if held_out else None,
        "scope": "Constructed injection labels on stored question/context/answer triples; excludes training and validation rows."}
    return result


def run_evaluation(pipeline, classifier, recovery_matrix, dataset_path,
                   output_dir, seed=42, limit=None, retry_errors=False, sample_indices=None):
    import torch
    import config
    from src.detection.nli_entailment import detect_grounding_failure
    from src.detection.response_check import assess_response

    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    dataset = load_dataset(dataset_path)
    selected = selected_sample_indices(len(dataset), limit, sample_indices)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_dir = output_dir / "samples"
    record_dir.mkdir(exist_ok=True)
    root = Path(config.__file__).resolve().parent
    source_paths = [root / "config.py", Path(__file__).resolve()]
    source_paths += sorted((root / "src").rglob("*.py"))
    thresholds = dict(retrieval=config.RETRIEVAL_SIM_THRESHOLD,
                      grounding=config.ENTAILMENT_THRESHOLD,
                      confidence=config.ENTROPY_THRESHOLD,
                      coverage=config.COVERAGE_SIM_THRESHOLD)
    signature = {
        "schema": SCHEMA, "dataset_sha256": digest(dataset_path),
        "seed": seed, "thresholds": thresholds,
        "source_sha256": {str(p.relative_to(root)) if p.is_relative_to(root)
                          else p.name: digest(p) for p in source_paths},
        "models": {key: getattr(config, key) for key in
                   ("LLM_MODEL_NAME", "EMBEDDING_MODEL_NAME", "NLI_MODEL_NAME")},
        "top_k": pipeline.top_k, "index_vectors": int(pipeline.index.ntotal),
        "pipeline_protocol": getattr(pipeline, "evaluation_protocol", {"recovery_version": 1}),
        "index_assets_sha256": {name: digest(config.FAISS_INDEX_DIR / name)
                                for name in ("index.faiss", "chunks.pkl")
                                if (config.FAISS_INDEX_DIR / name).is_file()},
        "torch_version": torch.__version__,
        "classifier": getattr(classifier.impl, "description", {"backend": classifier.backend}),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    if sample_indices is not None:
        signature["selected_original_sample_indices"] = selected
    classifier_impl = getattr(classifier, "impl", classifier)
    split = None
    if hasattr(classifier_impl, "manifest"):
        from src.classifier.bert_data import dataset_hash
        if classifier_impl.manifest["dataset_sha256"] != dataset_hash(dataset):
            raise ValueError("Classifier split belongs to another dataset; provide an explicit external-test protocol.")
        split = json.loads((classifier_impl.path / "split.json").read_text(encoding="utf-8"))
    split_lookup = {i: name for name, indices in (split or {}).get("indices", {}).items() for i in indices}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError("Run configuration changed. Choose a new output directory.")
    else:
        write_json(manifest_path, {"signature": signature,
                                  "dataset_class_counts": dict(Counter(x["label"] for x in dataset)),
                                  "protocol": "stored-answer detector audit plus live generation and one-action recovery"})

    records = []
    for position, i in enumerate(selected):
        item = dataset[i]
        path = record_dir / f"{i:05d}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["sample_index"] != i or row["input"] != item:
                raise ValueError(f"Checkpoint does not match sample {i}.")
            if not (retry_errors and row.get("error")):
                records.append(row)
                continue
        row = {"sample_index": i, "injected_label": item["label"],
               "input": item, "seed": seed + i, "classifier_split": split_lookup.get(i, "external")}
        started = time.perf_counter()
        try:
            random.seed(seed + i)
            torch.manual_seed(seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + i)
            # Compute the NLI score BEFORE generation can replace the wrong answer.
            stored_grounding = float(detect_grounding_failure(
                item["context_chunks"], item["answer"], query=item["query"]))
            if not math.isfinite(stored_grounding):
                stored_grounding = None
            initial = pipeline.run_rag_with_detection(
                item["query"], custom_chunks=item["context_chunks"])
            signals = finite_signals(initial["signals"])
            prediction = classifier.predict(initial["signals"], query=item["query"],
                                            contexts=initial["context_texts"], answer=initial["answer"])
            stored_prediction = classifier.predict(
                {}, query=item["query"], contexts=item["context_chunks"], answer=item["answer"])
            recovered = recovery_matrix.heal(item["query"], initial,
                                             prediction["predicted_label"])
            after = finite_signals(recovered["post_signals"])
            # Refusals remain visible; the dataset does not label answerability.
            before_assessment = assess_response(initial["answer"], signals, allow_refusal=False)
            after_assessment = assess_response(recovered["healed_answer"], after, allow_refusal=False)
            row.update(
                stored_answer_grounding=stored_grounding,
                stored_answer_prediction=stored_prediction,
                detector_flags={
                    "retrieval": threshold_flag(signals.get("retrieval_similarity"), thresholds["retrieval"]),
                    "stored_grounding": threshold_flag(stored_grounding, thresholds["grounding"]),
                    "generated_confidence": threshold_flag(signals.get("mean_entropy"), thresholds["confidence"], higher=True),
                    "coverage": threshold_flag(signals.get("coverage_score"), thresholds["coverage"]),
                },
                initial_answer=initial["answer"], initial_contexts=initial["context_texts"],
                initial_raw_answer=initial.get('raw_llm_output', {}).get('raw_text', initial['answer']),
                initial_signals=signals, initial_assessment=before_assessment,
                predicted_label=prediction["predicted_label"],
                recovery_action=recovered["action_taken"],
                recovery_trace=recovered.get("recovery_trace"),
                final_answer=recovered["healed_answer"],
                final_contexts=recovered["post_context_texts"],
                final_signals=after, final_assessment=after_assessment, error=None)
            del initial, recovered
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        row["elapsed_seconds"] = time.perf_counter() - started
        write_json(path, row)
        records.append(row)
        write_json(output_dir / "summary.json", summarise(records, len(selected)))
        status = row["error"] or row["predicted_label"]
        print(f"[{position + 1}/{len(selected)}; original ID {i}] {item['label']} -> {status} ({row['elapsed_seconds']:.1f}s)", flush=True)
        if len(records) >= 10:
            recent = records[-10:]
            if all(r.get("error") for r in recent):
                raise RuntimeError("Ten consecutive errors; stopped. Fix the cause and resume with retry_errors=True.")
    summary = summarise(records, len(selected))
    write_json(output_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config
    from src.pipeline.rag import RAGPipeline
    from src.classifier.predict import FailureClassifier
    from src.recovery.recovery_matrix import RecoveryMatrix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.FAILURE_DATASET_PATH)
    parser.add_argument("--output", type=Path,
                        default=config.VALIDATION_RESULTS_DIR / "full_injected")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    active_pipeline = RAGPipeline()
    result = run_evaluation(active_pipeline, FailureClassifier(), RecoveryMatrix(active_pipeline),
                            args.dataset, args.output, limit=args.limit,
                            retry_errors=args.retry_errors)
    print(json.dumps(result, indent=2))
