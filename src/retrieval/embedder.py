"""
Embedding wrapper using sentence-transformers.

Provides a unified interface for embedding documents and queries
using all-MiniLM-L6-v2 (384-dimensional embeddings).
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config


class Embedder:
    """
    Wrapper around sentence-transformers for document and query embedding.

    Attributes:
        model_name: Name of the sentence-transformers model.
        model: Loaded SentenceTransformer instance.
        dimension: Output embedding dimension.
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        """
        Initialize the embedder.

        Args:
            model_name: HuggingFace model name for sentence-transformers.
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded. Dimension: {self.dimension}")

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of document texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show a progress bar.
            normalize: Whether to L2-normalize embeddings (required for cosine sim via dot product).

        Returns:
            np.ndarray of shape (len(texts), dimension).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single query string.

        Args:
            query: The query text.
            normalize: Whether to L2-normalize the embedding.

        Returns:
            np.ndarray of shape (1, dimension) — 2D for FAISS compatibility.
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embedding.astype(np.float32)

    def embed_batch(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Embed one or more texts (convenience method).

        Args:
            texts: Single string or list of strings.
            normalize: Whether to L2-normalize.

        Returns:
            np.ndarray of shape (n, dimension).
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.embed_documents(texts, show_progress=False, normalize=normalize)


if __name__ == "__main__":
    embedder = Embedder()

    # Quick test
    test_texts = [
        "The capital of France is Paris.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
    ]
    embeddings = embedder.embed_documents(test_texts, show_progress=False)
    print(f"\nEmbedded {len(test_texts)} texts → shape: {embeddings.shape}")

    query_emb = embedder.embed_query("What is the capital of France?")
    print(f"Query embedding shape: {query_emb.shape}")

    # Cosine similarity (dot product since embeddings are normalized)
    similarities = embeddings @ query_emb.T
    print(f"Similarities to 'What is the capital of France?': {similarities.flatten()}")
