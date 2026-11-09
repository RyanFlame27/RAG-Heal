"""
Signal 3: Token Entropy Detector — detects confidence failure.

Computes the Shannon entropy of the LLM's output token probability
distributions. High entropy means the model was uncertain about which
token to generate -> low confidence in the answer.

Returns:
    - Mean entropy across all generated tokens.
    - High entropy -> confidence failure (model is uncertain).
    - Low entropy -> model is confident in its generation.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


def detect_confidence_failure(
    token_logits: Optional[List[Dict[str, Any]]],
    aggregation: str = "mean",
) -> Optional[float]:
    """
    Compute aggregated token entropy from LLM generation logits.
    """
    if token_logits is None or len(token_logits) == 0:
        return None

    entropies = [t.get("entropy") for t in token_logits]
    if any(e is None or not np.isfinite(e) or e < 0 for e in entropies):
        return None

    if aggregation == "mean":
        val = float(np.mean(entropies))
    elif aggregation == "max":
        val = float(np.max(entropies))
    elif aggregation == "median":
        val = float(np.median(entropies))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return val if np.isfinite(val) else None


def compute_top_prob_stats(
    token_logits: Optional[List[Dict[str, Any]]],
) -> Dict[str, float]:
    """
    Compute statistics about top-token probabilities.
    """
    if token_logits is None or len(token_logits) == 0:
        return {"mean_top_prob": 0.0, "min_top_prob": 0.0, "pct_below_50": 1.0}

    top_probs = [t["top_prob"] for t in token_logits if t.get("top_prob") is not None]
    if not top_probs:
        return {"mean_top_prob": None, "min_top_prob": None, "pct_below_50": None}

    return {
        "mean_top_prob": float(np.mean(top_probs)),
        "min_top_prob": float(np.min(top_probs)),
        "pct_below_50": float(np.mean([p < 0.5 for p in top_probs])),
    }


def is_confidence_failure(
    score: Optional[float],
    threshold: float = config.ENTROPY_THRESHOLD,
) -> bool:
    """Check if the entropy score indicates a confidence failure."""
    return score is None or not np.isfinite(score) or score >= threshold
