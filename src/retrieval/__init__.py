"""Retrieval module — chunking, embedding, and FAISS indexing."""

from .chunker import load_and_chunk_wikipedia
from .embedder import Embedder
from .index import build_index, save_index, load_index, retrieve

__all__ = [
    "load_and_chunk_wikipedia",
    "Embedder",
    "build_index",
    "save_index",
    "load_index",
    "retrieve",
]
