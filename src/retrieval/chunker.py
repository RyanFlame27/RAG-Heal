"""
Document chunker for Wikipedia articles.

Loads a subset of Wikipedia from HuggingFace datasets,
splits articles into overlapping chunks of ~300 tokens each,
and returns structured Document objects for downstream embedding.
"""

import pickle
from typing import List, Dict, Any, Optional

from datasets import load_dataset
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config


def load_wikipedia_articles(
    num_articles: int = config.NUM_WIKI_ARTICLES,
    dataset_name: str = config.WIKI_DATASET_NAME,
    dataset_config: str = config.WIKI_DATASET_CONFIG,
) -> List[Dict[str, str]]:
    """
    Load a subset of Wikipedia articles from HuggingFace.

    Args:
        num_articles: Number of articles to load.
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration / language version.

    Returns:
        List of dicts with keys 'title' and 'text'.
    """
    print(f"Loading {num_articles} Wikipedia articles...")
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=f"train[:{num_articles}]",
    )

    articles = []
    for row in tqdm(dataset, desc="Loading articles"):
        text = row.get("text", "")
        title = row.get("title", "Unknown")

        # Skip very short articles (likely stubs)
        if len(text.strip()) < 100:
            continue

        articles.append({"title": title, "text": text})

    print(f"Loaded {len(articles)} articles (skipped {num_articles - len(articles)} stubs).")
    return articles


def chunk_documents(
    articles: List[Dict[str, str]],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    separators: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Split articles into overlapping text chunks.

    Args:
        articles: List of dicts with 'title' and 'text'.
        chunk_size: Target chunk size in characters (~= tokens for English).
        chunk_overlap: Overlap between consecutive chunks.
        separators: Text split boundaries (defaults to config).

    Returns:
        List of chunk dicts: {text, metadata: {title, chunk_index, source}}.
    """
    if separators is None:
        separators = config.CHUNK_SEPARATORS

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )

    all_chunks = []
    for article in tqdm(articles, desc="Chunking articles"):
        splits = splitter.split_text(article["text"])
        for idx, chunk_text in enumerate(splits):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "title": article["title"],
                    "chunk_index": idx,
                    "source": "wikipedia",
                },
            })

    print(f"Created {len(all_chunks)} chunks from {len(articles)} articles.")
    return all_chunks


def save_chunks(chunks: List[Dict[str, Any]], path=None):
    """Persist chunk metadata to disk."""
    if path is None:
        path = config.CHUNK_METADATA_PATH
    path = __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to {path}")


def load_chunks(path=None) -> List[Dict[str, Any]]:
    """Load persisted chunk metadata from disk."""
    if path is None:
        path = config.CHUNK_METADATA_PATH
    with open(path, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def load_and_chunk_wikipedia(
    num_articles: int = config.NUM_WIKI_ARTICLES,
    save: bool = True,
) -> List[Dict[str, Any]]:
    """
    End-to-end: load Wikipedia → chunk → optionally save.

    This is the main entry point for building the knowledge base.
    """
    config.ensure_dirs()
    articles = load_wikipedia_articles(num_articles=num_articles)
    chunks = chunk_documents(articles)
    if save:
        save_chunks(chunks)
    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk_wikipedia()
    # Quick sanity check
    print(f"\nSample chunk:")
    print(f"  Title : {chunks[0]['metadata']['title']}")
    print(f"  Index : {chunks[0]['metadata']['chunk_index']}")
    print(f"  Text  : {chunks[0]['text'][:200]}...")
