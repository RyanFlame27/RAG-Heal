"""
Signal 4: KB Coverage Detector — detects coverage failure.

Checks how close the query is to anything in the knowledge base
by computing the maximum cosine similarity between the query
and all vectors in the FAISS index.

If the best match is still very low, the query's topic is not
covered by the KB at all → coverage failure.

Returns:
    - Maximum similarity score between query and KB.
    - Low score → the KB doesn't contain relevant information.
    - High score → there's at least some relevant content in the KB.
"""

import numpy as np
from typing import Optional

import faiss

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config
from src.retrieval.embedder import Embedder


def detect_coverage_failure(
    query: str,
    embedder: Embedder,
    index: faiss.Index,
    k: int = 10,
    aggregation: str = "max",
) -> float:
    """
    Compute KB coverage score for a query.

    Retrieves the top-k most similar vectors from the full KB
    and returns the aggregated similarity as a coverage indicator.

    Args:
        query: User query string.
        embedder: Embedder instance.
        index: FAISS index of the full knowledge base.
        k: Number of nearest neighbors to consider.
        aggregation: How to compute the coverage score ("max" or "mean").

    Returns:
        Coverage score (float). Lower = less coverage in the KB.
    """
    query_embedding = embedder.embed_query(query)
    scores, _ = index.search(query_embedding, k)

    similarities = scores[0]
    # Filter out -1 padding entries
    valid = similarities[similarities > -1e6]

    if len(valid) == 0:
        return 0.0

    if aggregation == "max":
        return float(np.max(valid))
    elif aggregation == "mean":
        return float(np.mean(valid))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def compute_coverage_density(
    query: str,
    embedder: Embedder,
    index: faiss.Index,
    k: int = 20,
    sim_threshold: float = 0.3,
) -> dict:
    """
    Compute additional density metrics for coverage analysis.

    Args:
        query: User query string.
        embedder: Embedder instance.
        index: FAISS index.
        k: Number of neighbors to analyze.
        sim_threshold: Minimum similarity to count as "relevant".

    Returns:
        Dict with max_sim, mean_sim, num_above_threshold, density_ratio.
    """
    query_embedding = embedder.embed_query(query)
    scores, _ = index.search(query_embedding, k)

    similarities = scores[0]
    valid = similarities[similarities > -1e6]

    if len(valid) == 0:
        return {
            "max_sim": 0.0,
            "mean_sim": 0.0,
            "num_above_threshold": 0,
            "density_ratio": 0.0,
        }

    num_above = int(np.sum(valid >= sim_threshold))

    return {
        "max_sim": float(np.max(valid)),
        "mean_sim": float(np.mean(valid)),
        "num_above_threshold": num_above,
        "density_ratio": num_above / len(valid),
    }


def is_coverage_failure(
    score: float,
    threshold: float = config.COVERAGE_SIM_THRESHOLD,
) -> bool:
    """Check if the coverage score indicates no relevant KB content."""
    return score < threshold


if __name__ == "__main__":
    print("Testing KB coverage detector...")
    print("(Requires FAISS index on disk)")
    try:
        from src.retrieval.index import load_index

        embedder = Embedder()
        index, chunks = load_index()

        # Test with a query likely IN the KB
        covered_score = detect_coverage_failure("What is photosynthesis?", embedder, index)
        covered_density = compute_coverage_density("What is photosynthesis?", embedder, index)

        # Test with a query likely NOT in the KB
        uncovered_score = detect_coverage_failure(
            "What is the airspeed velocity of an unladen swallow?", embedder, index
        )
        uncovered_density = compute_coverage_density(
            "What is the airspeed velocity of an unladen swallow?", embedder, index
        )

        print(f"Covered query score:   {covered_score:.4f} (failure={is_coverage_failure(covered_score)})")
        print(f"Covered density: {covered_density}")
        print(f"Uncovered query score: {uncovered_score:.4f} (failure={is_coverage_failure(uncovered_score)})")
        print(f"Uncovered density: {uncovered_density}")
    except Exception as e:
        print(f"Test failed (expected if no index): {e}")
