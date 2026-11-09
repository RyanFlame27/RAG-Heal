"""
Centralized configuration for the final BERT-guided Self-Healing RAG.

All hyperparameters, model names, and paths live here so every module
pulls from a single source of truth.
"""

import os
from pathlib import Path

# ── Project root ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RELEASE_VERSION = "4.2"

# ── Data paths ─────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
SOURCE_FAISS_INDEX_DIR = DATA_DIR / "faiss_index_with_source_passages"
SOURCE_DOCUMENTS_PATH = DATA_DIR / "source_documents.json"
CHUNK_METADATA_PATH = PROCESSED_DATA_DIR / "chunk_metadata.pkl"

# ── Benchmark paths ───────────────────────────────────────────
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
FAILURE_DATASET_PATH = BENCHMARK_DIR / "failure_dataset.json"
VALIDATION_RESULTS_DIR = BENCHMARK_DIR / "results"

# ── Embedding model ───────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # output dim of all-MiniLM-L6-v2

# ── LLM ───────────────────────────────────────────────────────
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LLM_MAX_NEW_TOKENS = 256
LLM_TEMPERATURE = 0.1
LLM_QUANTIZATION_BITS = 4  # 4-bit quantization via bitsandbytes

# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE = 300          # characters per chunk (length_function=len)
CHUNK_OVERLAP = 50        # character overlap between consecutive chunks
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ── Retrieval ─────────────────────────────────────────────────
TOP_K = 5                 # number of chunks to retrieve
RECOVERY_VERSION = 2     # Validated evidence recovery; use recovery_version=1 for the frozen original run.

# ── Wikipedia dataset ─────────────────────────────────────────
NUM_WIKI_ARTICLES = 5000  # number of Wikipedia articles to index
WIKI_DATASET_NAME = "wikimedia/wikipedia"
WIKI_DATASET_CONFIG = "20231101.en"

# ── NLI model (grounding detection) ──────────────────────────
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# Failure routing uses a separately fine-tuned BERT model. NLI is not this model.
FAILURE_CLASSIFIER_BACKEND = "bert"
BERT_CLASSIFIER_BASE = "google-bert/bert-base-uncased"
BERT_CLASSIFIER_DIR = DATA_DIR / "bert_failure_classifier_runtime_v4"
BERT_CLASSIFIER_MAX_LENGTH = 384
BERT_CLASSIFIER_DEVICE = "cpu"  # Reserve GPU memory for Mistral and NLI at inference.

# ── Detection thresholds (tuned for sharp failure separation) ─
RETRIEVAL_SIM_THRESHOLD = 0.45     # below = retrieval failure
ENTAILMENT_THRESHOLD = 0.50        # below = grounding failure
ENTROPY_THRESHOLD = 2.00           # above = confidence failure
COVERAGE_SIM_THRESHOLD = 0.45     # below = coverage failure

# ── Failure dataset generation ────────────────────────────────
QA_DATASET_NAME = "natural_questions"
NUM_QA_EXAMPLES = 200             # per-class target (clean + 4 failure types)

# ── Prompt template ───────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """<s>[INST] You are a helpful assistant. Use ONLY the provided context to answer the question. If the context does not contain enough information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer: [/INST]"""


def ensure_dirs():
    """Create all required data directories if they don't exist."""
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
              FAISS_INDEX_DIR, BENCHMARK_DIR, VALIDATION_RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
