"""Detection module — four signal-based failure detectors."""

from .embedding_similarity import detect_retrieval_failure
from .nli_entailment import detect_grounding_failure
from .token_entropy import detect_confidence_failure
from .kb_coverage import detect_coverage_failure

__all__ = [
    "detect_retrieval_failure",
    "detect_grounding_failure",
    "detect_confidence_failure",
    "detect_coverage_failure",
]
