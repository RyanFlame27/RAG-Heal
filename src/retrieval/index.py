"""
FAISS index management — build, save, load, and retrieve.

Uses IndexFlatIP (inner product) with L2-normalized vectors,
which is equivalent to cosine similarity.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.retrieval.embedder import Embedder


def build_index(
    chunks: List[Dict[str, Any]],
    embedder: Embedder,
    batch_size: int = 256,
) -> faiss.Index:
    """
    Build a FAISS index from document chunks.

    Args:
        chunks: List of chunk dicts with 'text' key.
        embedder: Embedder instance for generating embeddings.
        batch_size: Batch size for embedding computation.

    Returns:
        Populated FAISS IndexFlatIP.
    """
    dimension = embedder.dimension
    index = faiss.IndexFlatIP(dimension)  # Inner product (= cosine sim for normalized vecs)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks for indexing...")
    # Process in batches to manage memory
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i : i + batch_size]
        batch_embeddings = embedder.embed_documents(batch, show_progress=False)
        all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Embedding matrix shape: {embeddings.shape}")

    # Add to FAISS index
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    return index


def save_index(
    index: faiss.Index,
    chunks: List[Dict[str, Any]],
    index_dir: Optional[Path] = None,
):
    """
    Persist FAISS index and chunk metadata to disk.

    Args:
        index: FAISS index to save.
        chunks: Chunk metadata (saved alongside for retrieval).
        index_dir: Directory to save into (defaults to config).
    """
    if index_dir is None:
        index_dir = config.FAISS_INDEX_DIR
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "chunks.pkl"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index ({index.ntotal} vectors) to {index_path}")
    print(f"Saved chunk metadata to {meta_path}")


def load_index(
    index_dir: Optional[Path] = None,
) -> tuple:
    """
    Load a persisted FAISS index and chunk metadata.

    Args:
        index_dir: Directory containing saved index (defaults to config).

    Returns:
        Tuple of (faiss.Index, List[chunk_dicts]).
    """
    if index_dir is None:
        index_dir = config.FAISS_INDEX_DIR
    index_dir = Path(index_dir)

    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "chunks.pkl"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run the indexing pipeline first."
        )

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded FAISS index ({index.ntotal} vectors) from {index_path}")
    return index, chunks


def retrieve(
    query: str,
    embedder: Embedder,
    index: faiss.Index,
    chunks: List[Dict[str, Any]],
    k: int = config.TOP_K,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most relevant chunks for a query.

    Args:
        query: User query string.
        embedder: Embedder instance.
        index: FAISS index.
        chunks: Chunk metadata list (aligned with index).
        k: Number of results to return.

    Returns:
        List of dicts: {text, metadata, score, rank}.
    """
    query_embedding = embedder.embed_query(query)  # shape: (1, dim)
    scores, indices = index.search(query_embedding, k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "score": float(score),
            "rank": rank,
        })

    return results


def retrieve_with_embeddings(
    query: str,
    embedder: Embedder,
    index: faiss.Index,
    chunks: List[Dict[str, Any]],
    k: int = config.TOP_K,
) -> Dict[str, Any]:
    """
    Retrieve top-k chunks along with query and chunk embeddings.

    Used by the detection layer which needs the raw embeddings
    for similarity calculations.

    Returns:
        Dict with keys: results, query_embedding, chunk_embeddings.
    """
    query_embedding = embedder.embed_query(query)
    scores, indices = index.search(query_embedding, k)

    results = []
    chunk_texts = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "score": float(score),
            "rank": rank,
        })
        chunk_texts.append(chunk["text"])

    # Re-embed the retrieved chunks for detection signals
    chunk_embeddings = embedder.embed_documents(chunk_texts, show_progress=False)

    return {
        "results": results,
        "query_embedding": query_embedding,
        "chunk_embeddings": chunk_embeddings,
    }


if __name__ == "__main__":
    # Quick test: build a small index and retrieve
    embedder = Embedder()

    test_chunks = [
        {"text": "Paris is the capital of France.", "metadata": {"title": "France", "chunk_index": 0, "source": "test"}},
        {"text": "Berlin is the capital of Germany.", "metadata": {"title": "Germany", "chunk_index": 0, "source": "test"}},
        {"text": "Python is a programming language.", "metadata": {"title": "Python", "chunk_index": 0, "source": "test"}},
    ]

    index = build_index(test_chunks, embedder)
    results = retrieve("What is the capital of France?", embedder, index, test_chunks, k=2)

    print("\nQuery: 'What is the capital of France?'")
    for r in results:
        print(f"  [{r['rank']}] (score={r['score']:.4f}) {r['text']}")
