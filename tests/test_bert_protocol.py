"""Dataset leakage, classifier wiring and benchmark bookkeeping checks."""

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classifier.bert_data import classifier_text, grouped_split, question_key, LABELS
from src.classifier.predict import FailureClassifier
from benchmark.run_baseline_comparison import answer_scores, reference_answers, comparison_report
from benchmark.comparison_baselines import retrieval_action
from benchmark.run_injected_eval import finite_signals, summarise
from benchmark.run_injected_eval import run_evaluation, selected_sample_indices


def examples():
    rows = []
    for i in range(20):
        for label in LABELS[:-1]:
            rows.append(dict(query=f"Question {i}?", answer="Paris" if label == "clean" else "Tokyo",
                             context_chunks=["Question-specific evidence."], label=label))
    for i in range(12):
        for _ in range(2):
            rows.append(dict(query=f"Future fictional question {i}?", answer="Unknown",
                             context_chunks=["Other evidence."], label="coverage_failure"))
    return rows


class BertProtocolTests(unittest.TestCase):
    def test_runtime_training_target_keeps_conditions_and_marks_clean_refusals(self):
        from src.classifier.train_runtime_bert import runtime_target
        import importlib.util
        spec=importlib.util.spec_from_file_location('runtime_policy_test',Path(__file__).resolve().parents[1]/'src/detection/response_check.py')
        policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
        with patch.dict(sys.modules,{'src.detection':ModuleType('src.detection'),'src.detection.response_check':policy}):
            self.assertEqual(runtime_target('retrieval_failure',"I don't have enough information to answer this question."),'retrieval_failure')
            self.assertEqual(runtime_target('coverage_failure',"I don't have enough information to answer this question."),'coverage_failure')
            self.assertEqual(runtime_target('clean',"I don't have enough information to answer this question."),'grounding_failure')
            self.assertEqual(runtime_target('clean','Paris'),'clean')

    def test_subset_selection_retains_original_ids_and_rejects_invalid_ids(self):
        self.assertEqual(selected_sample_indices(10, sample_indices=[8, 2]), [2, 8])
        for invalid in ([2, 2], [-1], [10], [True], []):
            with self.assertRaises(ValueError):
                selected_sample_indices(10, sample_indices=invalid)
        with self.assertRaises(ValueError):
            selected_sample_indices(10, limit=3, sample_indices=[2])

    def test_subset_comparison_requires_only_selected_ids_for_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = examples()
            source = root / "input.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            (root/"ours/samples").mkdir(parents=True)
            (root/"baseline/samples").mkdir(parents=True)
            assessment = {"outcome": "supported", "refusal": False}
            i = 4
            ours = dict(sample_index=i, input=data[i], error=None,
                        initial_answer="Paris", final_answer="Paris",
                        initial_assessment=assessment, final_assessment=assessment)
            baseline = dict(sample_index=i, input=data[i], error=None,
                            answer="Paris", assessment=assessment)
            (root/"ours/samples/00004.json").write_text(json.dumps(ours), encoding="utf-8")
            (root/"baseline/samples/00004.json").write_text(json.dumps(baseline), encoding="utf-8")
            report = comparison_report(source, root/"ours", {"baseline": root/"baseline"},
                                       root/"report.json", sample_indices=[i])
            self.assertTrue(report["all_methods_complete_without_error"])
            self.assertEqual(report["evaluation_sample_count"], 1)
            self.assertEqual(report["dataset_size"], len(data))
            self.assertEqual(report["selected_original_sample_indices"], [i])

    def test_grouped_split_keeps_all_variants_and_duplicates_together(self):
        data = examples()
        split = grouped_split(data)
        sets = {name: {question_key(data[i]["query"]) for i in ids}
                for name, ids in split["indices"].items()}
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])
        ids = [i for subset in split["indices"].values() for i in subset]
        self.assertEqual(sorted(ids), list(range(len(data))))
        self.assertEqual(split, grouped_split(data))

    def test_classifier_input_retains_answer_and_excludes_labels(self):
        row = examples()[1]
        text = classifier_text(row["query"], row["context_chunks"], row["answer"])
        self.assertLess(text.index("Answer:"), text.index("Context:"))
        self.assertNotIn(row["label"], text)
        with self.assertRaises(ValueError):
            classifier_text(None, [], None)

    def test_default_bert_missing_checkpoint_fails_instead_of_using_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "No trained BERT"):
                FailureClassifier(model_path=Path(directory), backend="bert")

    def test_router_passes_full_text_state_to_bert_implementation(self):
        with patch("src.classifier.bert_classifier.BertFailureClassifier") as cls:
            router = FailureClassifier(backend="bert")
            router.predict({"mean_entropy": None}, query="Question", contexts=["Context"], answer="Answer")
            cls.return_value.predict.assert_called_once_with(
                {"mean_entropy": None}, query="Question", contexts=["Context"], answer="Answer")

    def test_reference_comes_from_clean_sibling_not_injected_wrong_answer(self):
        data = examples()
        refs = reference_answers(data)
        self.assertEqual(refs["Question 0?"], "Paris")
        self.assertEqual(answer_scores("Tokyo", refs["Question 0?"])["exact_match"], 0)
        self.assertEqual(answer_scores("The Paris.", "Paris")["exact_match"], 1)

    def test_crag_all_three_routes_and_threshold_sign(self):
        self.assertEqual(retrieval_action([0.7]), "correct")
        self.assertEqual(retrieval_action([0.0]), "ambiguous")
        self.assertEqual(retrieval_action([-1.1]), "incorrect")
        with self.assertRaises(ValueError):
            retrieval_action([float("nan")])

    def test_unavailable_measurement_and_empty_results_do_not_look_successful(self):
        self.assertIsNone(finite_signals({"mean_entropy": float("nan")})["mean_entropy"])
        summary = summarise([], 1000)
        self.assertFalse(summary["all_samples_completed_without_error"])
        self.assertIsNone(summary["bert_stored_answer_held_out_test"]["label_accuracy"])

    def test_missing_baseline_does_not_produce_comparative_performance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.json"
            dataset_path.write_text(json.dumps(examples()), encoding="utf-8")
            report = comparison_report(dataset_path, root / "ours", {"selfrag": root / "selfrag"}, root / "report.json")
            self.assertFalse(report["all_methods_complete_without_error"])
            self.assertIsNone(report["methods"]["selfrag"]["common_completed_samples"]["exact_match"])

    def test_full_runner_preserves_injected_answer_and_resumes_without_regeneration(self):
        # Stub inference only; exercise the real runner, actual response policy,
        # saved records, resume guard and stored-vs-generated answer distinction.
        import config
        torch_stub = SimpleNamespace(__version__="test-stub", manual_seed=lambda seed: None,
                                    cuda=SimpleNamespace(is_available=lambda: False))
        detection_package = ModuleType("src.detection")
        detection_package.__path__ = [str(Path(config.__file__).parent / "src/detection")]
        seen_answers = []
        def grounding(contexts, answer, query):
            seen_answers.append(answer)
            return 0.1 if answer == "Tokyo" else 0.9
        nli_stub = SimpleNamespace(detect_grounding_failure=grounding)
        signals = dict(retrieval_similarity=0.8, entailment_score=0.9,
                       mean_entropy=0.1, coverage_score=0.8)
        calls = []
        def generate(query, custom_chunks):
            calls.append(query)
            return dict(answer="Paris", context_texts=custom_chunks, signals=signals)
        pipeline = SimpleNamespace(top_k=5, index=SimpleNamespace(ntotal=1), run_rag_with_detection=generate)
        classifier = SimpleNamespace(backend="test-stub", impl=SimpleNamespace(description={"backend":"test-stub"}),
            predict=lambda signals, **state: {"predicted_label":"clean", "probabilities":{"clean":1.0}})
        matrix = SimpleNamespace(heal=lambda query, initial, label: dict(
            healed_answer=initial["answer"], post_context_texts=initial["context_texts"],
            post_signals=signals, action_taken="None"))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
                "torch":torch_stub, "src.detection":detection_package,
                "src.detection.nli_entailment":nli_stub}):
            root = Path(directory)
            data = [dict(query="Capital?", answer="Tokyo", context_chunks=["The capital is Paris."],
                         label="grounding_failure")]
            source = root / "input.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            run_evaluation(pipeline, classifier, matrix, source, root/"run")
            result = json.loads((root/"run/samples/00000.json").read_text(encoding="utf-8"))
            self.assertEqual(result["input"]["answer"], "Tokyo")
            self.assertEqual(result["initial_answer"], "Paris")
            self.assertEqual(result["stored_answer_grounding"], 0.1)
            self.assertEqual(result["initial_signals"]["entailment_score"], 0.9)
            run_evaluation(pipeline, classifier, matrix, source, root/"run")
            self.assertEqual(len(calls), 1)
            self.assertEqual(seen_answers, ["Tokyo"])
            # Sparse selection must retain its original ID and per-ID seed.
            data = data * 3
            source.write_text(json.dumps(data), encoding="utf-8")
            run_evaluation(pipeline, classifier, matrix, source, root/"subset", sample_indices=[2])
            sparse = json.loads((root/"subset/samples/00002.json").read_text(encoding="utf-8"))
            self.assertEqual(sparse["sample_index"], 2)
            self.assertEqual(sparse["seed"], 44)
            self.assertFalse((root/"subset/samples/00000.json").exists())


if __name__ == "__main__":
    unittest.main()
