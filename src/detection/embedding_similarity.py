"""
Signal 1: Embedding Similarity Detector — detects retrieval failure.

Computes cosine similarity between the query embedding and each
retrieved chunk embedding. Low similarity indicates that the retriever
returned irrelevant documents.

Returns:
    - Mean cosine similarity across top-k chunks.
    - Values close to 0 or negative → retrieval failure.
    - Values close to 1 → retrieval is on-topic.
"""

import numpy as np
from typing import Optional

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config


def detect_retrieval_failure(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    aggregation: str = "mean",
) -> float:
    """
    Compute embedding similarity between query and retrieved chunks.

    Args:
        query_embedding: Query embedding of shape (1, dim).
        chunk_embeddings: Chunk embeddings of shape (k, dim).
        aggregation: How to aggregate per-chunk scores ("mean" or "max").

    Returns:
        Aggregated cosine similarity score (float).
        Lower values indicate potential retrieval failure.
    """
    if query_embedding is None or chunk_embeddings is None:
        return 0.0

    if len(chunk_embeddings) == 0:
        return 0.0

    # Ensure 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Cosine similarity = dot product for L2-normalized vectors
    similarities = (chunk_embeddings @ query_embedding.T).flatten()

    if aggregation == "mean":
        return float(np.mean(similarities))
    elif aggregation == "max":
        return float(np.max(similarities))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def is_retrieval_failure(
    score: float,
    threshold: float = config.RETRIEVAL_SIM_THRESHOLD,
) -> bool:
    """Check if the retrieval similarity score indicates failure."""
    return score < threshold


if __name__ == "__main__":
    # Quick unit test with synthetic data
    np.random.seed(42)
    dim = 384

    # Simulate good retrieval (high similarity)
    query = np.random.randn(1, dim).astype(np.float32)
    query /= np.linalg.norm(query)
    good_chunks = query + np.random.randn(5, dim).astype(np.float32) * 0.1
    good_chunks /= np.linalg.norm(good_chunks, axis=1, keepdims=True)

    # Simulate bad retrieval (random chunks)
    bad_chunks = np.random.randn(5, dim).astype(np.float32)
    bad_chunks /= np.linalg.norm(bad_chunks, axis=1, keepdims=True)

    good_score = detect_retrieval_failure(query, good_chunks)
    bad_score = detect_retrieval_failure(query, bad_chunks)

    print(f"Good retrieval score: {good_score:.4f} (failure={is_retrieval_failure(good_score)})")
    print(f"Bad retrieval score:  {bad_score:.4f} (failure={is_retrieval_failure(bad_score)})")
    print(f"Threshold: {config.RETRIEVAL_SIM_THRESHOLD}")
