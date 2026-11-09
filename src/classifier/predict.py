"""BERT failure routing, with the previous rule router available explicitly."""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
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

_cached_classifier = None
_cached_key = None


class RuleBasedFailureClassifier:
    """Real-time classifier wrapper for predicting failure modes from signals."""

    def __init__(self, model_path: Path = None):
        self.label_map = LABEL_MAP
        self.reverse_map = REVERSE_LABEL_MAP

    def predict(self, signals: Dict[str, float], **text_state) -> Dict[str, Any]:
        """
        Predict failure class and probabilities from a signals dict.

        Priority order:
        1. Coverage Failure: Both retrieval and coverage low -> completely out-of-scope
        2. Coverage Failure: Coverage strictly below threshold (< 0.45)
        3. Retrieval Failure: Retrieval low but topic exists in index
        4. Grounding Failure: Retrieved context contradicts or does not support answer
        5. Confidence Failure: High generation entropy
        6. Clean: Passed all signal audits
        """
        ret = signals.get("retrieval_similarity", 0.5)
        ent = signals.get("entailment_score", 0.5)
        tok = signals.get("mean_entropy")
        refusal = signals.get("is_refusal", False)
        cov = signals.get("coverage_score", 0.5)

        ret_thresh = config.RETRIEVAL_SIM_THRESHOLD
        cov_thresh = config.COVERAGE_SIM_THRESHOLD
        ent_thresh = config.ENTAILMENT_THRESHOLD
        tok_thresh = config.ENTROPY_THRESHOLD

        # Decision tree with priority routing
        if ret < ret_thresh and cov < cov_thresh:
            pred_label = "coverage_failure"
        elif cov < cov_thresh:
            pred_label = "coverage_failure"
        elif ret < ret_thresh:
            pred_label = "retrieval_failure"
        elif not refusal and (not np.isfinite(ent) or ent < ent_thresh):
            pred_label = "grounding_failure"
        elif not refusal and (tok is None or not np.isfinite(tok) or tok >= tok_thresh):
            pred_label = "confidence_failure"
        else:
            pred_label = "clean"

        class_id = self.label_map[pred_label]
        probs = {lbl: (1.0 if lbl == pred_label else 0.0) for lbl in self.label_map}

        return {
            "predicted_label": pred_label,
            "class_id": class_id,
            "probabilities": probs,
            "is_failure": pred_label != "clean",
        }


class FailureClassifier:
    """Configured router; BERT requires trained weights, with no silent fallback."""

    def __init__(self, model_path=None, backend=None):
        self.backend = backend or config.FAILURE_CLASSIFIER_BACKEND
        if self.backend == "rules":
            self.impl = RuleBasedFailureClassifier()
        elif self.backend == "bert":
            from .bert_classifier import BertFailureClassifier
            self.impl = BertFailureClassifier(model_path)
        else:
            raise ValueError(f"Unknown classifier backend: {self.backend}")

    def predict(self, signals, *, query=None, contexts=None, answer=None):
        return self.impl.predict(signals, query=query, contexts=contexts, answer=answer)


def load_classifier() -> FailureClassifier:
    global _cached_classifier, _cached_key
    key = (config.FAILURE_CLASSIFIER_BACKEND, str(config.BERT_CLASSIFIER_DIR))
    if _cached_classifier is None or _cached_key != key:
        _cached_classifier = FailureClassifier()
        _cached_key = key
    return _cached_classifier
