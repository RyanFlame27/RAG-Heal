"""Checkpoint baseline outputs and compare all systems on identical sample IDs."""

from collections import Counter
import json
from pathlib import Path
import random
import re
import string
import time

from benchmark.run_injected_eval import digest, write_json, load_dataset, finite_signals, selected_sample_indices


def normalise(text):
    text = text.casefold().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(answer, reference):
    predicted, gold = normalise(answer), normalise(reference)
    p_tokens, g_tokens = predicted.split(), gold.split()
    common = sum((Counter(p_tokens) & Counter(g_tokens)).values())
    if not p_tokens or not g_tokens:
        f1 = float(p_tokens == g_tokens)
    elif not common:
        f1 = 0.0
    else:
        precision, recall = common / len(p_tokens), common / len(g_tokens)
        f1 = 2 * precision * recall / (precision + recall)
    return {"exact_match": float(predicted == gold), "token_f1": f1}


def reference_answers(dataset):
    references = {}
    for row in dataset:
        if row["label"] == "clean":
            previous = references.setdefault(row["query"], row["answer"])
            if previous != row["answer"]:
                raise ValueError("Conflicting clean reference answers.")
    for row in dataset:
        if row["label"] != "coverage_failure" and row["query"] not in references:
            raise ValueError("A corrupted sample lacks a matching clean reference.")
    return references


def run_baseline(adapter, auditor, dataset_path, output_dir, limit=None, retry_errors=False, sample_indices=None):
    import torch
    from src.detection.response_check import assess_response
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    data = load_dataset(dataset_path)
    selected = selected_sample_indices(len(data), limit, sample_indices)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = output_dir / "samples"
    records_dir.mkdir(exist_ok=True)
    import config
    signature = {"dataset_sha256": digest(dataset_path), "adapter": adapter.description,
                 "source_sha256": {
                     "comparison_baselines.py": digest(Path(__file__).with_name("comparison_baselines.py")),
                     "run_baseline_comparison.py": digest(__file__),
                     **{str(p.relative_to(config.PROJECT_ROOT)): digest(p)
                        for p in sorted(config.PROJECT_ROOT.glob("src/**/*.py"))}},
                 "config_sha256": digest(config.__file__), "torch_version": torch.__version__}
    if sample_indices is not None:
        signature["selected_original_sample_indices"] = selected
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != signature:
            raise ValueError("Baseline settings changed. Use a new output directory.")
    else:
        write_json(manifest_path, signature)
    records = []
    for position, i in enumerate(selected):
        item = data[i]
        path = records_dir / f"{i:05d}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["input"] != item:
                raise ValueError("Baseline checkpoint input mismatch.")
            if not (retry_errors and row.get("error")):
                records.append(row)
                continue
        started = time.perf_counter()
        row = {"sample_index": i, "input": item, "method": adapter.name}
        try:
            random.seed(42 + i)
            torch.manual_seed(42 + i)
            generated = adapter.answer(item)
            signals = finite_signals(auditor._audit(
                item["query"], generated["answer"], generated["contexts"],
                generated.get("token_logits")))
            row.update(answer=generated["answer"], contexts=generated["contexts"],
                       signals=signals, trace=generated["trace"],
                       assessment=assess_response(generated["answer"], signals, allow_refusal=False), error=None)
            del generated
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        row["elapsed_seconds"] = time.perf_counter() - started
        write_json(path, row)
        records.append(row)
        print(f"{adapter.name} [{position + 1}/{len(selected)}; original ID {i}] {row['error'] or 'saved'}", flush=True)
        if len(records) >= 10 and all(r.get("error") for r in records[-10:]):
            raise RuntimeError("Ten consecutive baseline errors; fix and resume with retry_errors=True.")
    write_json(output_dir / "completion.json", {"expected": len(selected), "processed": len(records),
               "execution_errors": sum(bool(r.get("error")) for r in records)})


def comparison_report(dataset_path, ours_dir, baseline_dirs, output_path, split_path=None, sample_indices=None):
    dataset = load_dataset(dataset_path)
    references = reference_answers(dataset)
    def read_records(directory):
        records = {}
        for path in sorted((Path(directory) / "samples").glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            i = row["sample_index"]
            if i >= len(dataset) or row["input"] != dataset[i]:
                raise ValueError("Comparison input mismatch.")
            records[i] = row
        return records
    ours = read_records(ours_dir)
    methods = {"vanilla_mistral": (ours, "initial"), "ours_bert_recovery": (ours, "final")}
    methods.update({name: (read_records(directory), None) for name, directory in baseline_dirs.items()})
    held_out = None
    if split_path:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        from src.classifier.bert_data import dataset_hash
        if split["dataset_sha256"] != dataset_hash(dataset):
            raise ValueError("Held-out split belongs to another dataset.")
        held_out = set(split["indices"]["test"])
    expected = selected_sample_indices(len(dataset), sample_indices=sample_indices)
    common = set(expected)
    for records, _ in methods.values():
        common &= {i for i, row in records.items() if not row.get("error")}

    def aggregate(ids, records, side):
        scored, outcomes = [], Counter()
        by_class = {}
        for i in sorted(ids):
            row = records[i]
            answer = row[f"{side}_answer"] if side else row["answer"]
            assessment = row[f"{side}_assessment"] if side else row["assessment"]
            outcomes[assessment["outcome"]] += 1
            label = dataset[i]["label"]
            group = by_class.setdefault(label, {"samples": 0, "refusals": 0, "exact_match_sum": 0., "token_f1_sum": 0., "reference_samples": 0})
            group["samples"] += 1
            group["refusals"] += int(assessment["refusal"])
            if label != "coverage_failure":
                score = answer_scores(answer, references[dataset[i]["query"]])
                scored.append(score)
                group["reference_samples"] += 1
                for key, value in score.items():
                    group[key + "_sum"] += value
        return {"samples": len(ids), "outcomes": dict(outcomes), "reference_samples": len(scored),
                "exact_match": sum(s["exact_match"] for s in scored) / len(scored) if scored else None,
                "token_f1": sum(s["token_f1"] for s in scored) / len(scored) if scored else None,
                "by_injected_class": by_class}
    reports = {}
    for name, (records, side) in methods.items():
        reports[name] = {"processed": len(records), "errors": sum(bool(r.get("error")) for r in records.values()),
                         "common_completed_samples": aggregate(common, records, side)}
        if held_out is not None:
            reports[name]["common_held_out_test"] = aggregate(common & held_out, records, side)
    report = {"dataset_size": len(dataset), "evaluation_sample_count": len(expected),
              "selected_original_sample_indices": expected,
              "common_successful_samples": len(common),
              "all_methods_complete_without_error": len(common) == len(expected),
              "methods": reports,
              "interpretation": [
                  "Only common successful sample IDs are compared. Errors and missing rows remain explicit.",
                  "Primary BERT comparison uses common_held_out_test, not training examples.",
                  "References come from clean siblings, never from deliberately wrong stored answers.",
                  "Coverage samples have no verified factual reference and are excluded from answer EM/F1.",
                  "Token overlap and automatic support checks are not independent factual judgement.",
                  "Local CRAG is modified. Self-RAG uses a different pretrained generator and a 4-bit short-form port.",
                  "Mistral systems and Self-RAG differ in checkpoint and decoding; results do not isolate architecture alone.",
              ]}
    write_json(output_path, report)
    return report
