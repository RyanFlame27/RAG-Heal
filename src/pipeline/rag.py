"""
End-to-end RAG pipeline orchestration.

Connects retrieval, generation, and detection into a single
callable interface: run_rag(query) -> full result dict.
"""

from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config
from src.retrieval.embedder import Embedder
from src.retrieval.index import load_index, retrieve, retrieve_with_embeddings
from src.generation.llm import LLMWrapper


class RAGPipeline:
    """
    Vanilla RAG pipeline with optional failure detection signals.

    Flow: query → retrieve(query, k) → build prompt → LLM generate → answer

    This serves as both the working pipeline and the baseline
    for comparison against the self-healing version in Phase 2.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        llm: Optional[LLMWrapper] = None,
        index=None,
        chunks=None,
        top_k: int = config.TOP_K,
        recovery_version: Optional[int] = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            embedder: Pre-loaded Embedder (loads default if None).
            llm: Pre-loaded LLMWrapper (loads default if None).
            index: Pre-loaded FAISS index (loads from disk if None).
            chunks: Pre-loaded chunk metadata (loads from disk if None).
            top_k: Number of chunks to retrieve.
        """
        self.top_k = top_k
        self.recovery_version = config.RECOVERY_VERSION if recovery_version is None else recovery_version
        self._evidence_retriever = None

        # Load components lazily
        print("Initializing RAG Pipeline...")

        self.embedder = embedder or Embedder()

        if index is not None and chunks is not None:
            self.index = index
            self.chunks = chunks
        else:
            source_dir = config.SOURCE_FAISS_INDEX_DIR
            if self.recovery_version == 2 and (source_dir/'index.faiss').is_file() and (source_dir/'chunks.pkl').is_file():
                self.index, self.chunks = load_index(source_dir)
            else:
                self.index, self.chunks = load_index()
                if self.recovery_version == 2:
                    print('Using the base corpus. To add the evaluated static source passages, '
                          'run: python -m benchmark.prepare_evidence_index')

        self.llm = llm or LLMWrapper()

        print(f"RAG Pipeline ready. Index: {self.index.ntotal} vectors, top_k={self.top_k}")

    @property
    def evidence_retriever(self):
        if self._evidence_retriever is None:
            from src.retrieval.evidence import EvidenceRetriever
            self._evidence_retriever = EvidenceRetriever(self)
        return self._evidence_retriever

    @property
    def evaluation_protocol(self):
        return {"recovery_version": self.recovery_version,
                "answer_style": "concise extractive" if self.recovery_version == 2 else "original",
                "recovery_candidates": 24 if self.recovery_version == 2 else None,
                "recovery_evidence_passages": 3 if self.recovery_version == 2 else None}

    def run_rag(
        self,
        query: str,
        return_logits: bool = True,
        return_embeddings: bool = False,
        custom_chunks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a single query.

        Args:
            query: User's question.
            return_logits: Whether to capture token-level logits (for entropy detector).
            return_embeddings: Whether to return raw embeddings (for similarity detectors).
            custom_chunks: If provided, skip retrieval and use these chunks instead.
                          Used by the failure-injection benchmark.

        Returns:
            Dict with keys:
                - query: Original query string.
                - answer: Generated answer text.
                - retrieved_chunks: List of retrieved chunk dicts.
                - raw_llm_output: Full LLM output dict.
                - context_texts: List of context text strings used in prompt.
                - prompt: The full prompt sent to the LLM.
                - query_embedding: (optional) Query embedding vector.
                - chunk_embeddings: (optional) Retrieved chunk embeddings.
        """
        # ── Step 1: Retrieve ──────────────────────────────────────
        if custom_chunks is not None:
            # Use injected chunks (for benchmark / failure testing)
            retrieved = [
                {"text": c, "metadata": {"title": "injected", "chunk_index": i, "source": "custom"}, "score": 0.0, "rank": i}
                for i, c in enumerate(custom_chunks)
            ]
            context_texts = custom_chunks
            query_embedding = None
            chunk_embeddings = None

            if return_embeddings:
                query_embedding = self.embedder.embed_query(query)
                chunk_embeddings = self.embedder.embed_documents(
                    custom_chunks, show_progress=False
                )
        elif return_embeddings:
            retrieval_result = retrieve_with_embeddings(
                query, self.embedder, self.index, self.chunks, k=self.top_k
            )
            retrieved = retrieval_result["results"]
            query_embedding = retrieval_result["query_embedding"]
            chunk_embeddings = retrieval_result["chunk_embeddings"]
            context_texts = [r["text"] for r in retrieved]
        else:
            retrieved = retrieve(
                query, self.embedder, self.index, self.chunks, k=self.top_k
            )
            context_texts = [r["text"] for r in retrieved]
            query_embedding = None
            chunk_embeddings = None

        # ── Step 2: Build prompt ──────────────────────────────────
        if self.recovery_version == 2:
            from src.recovery.evidence_recovery import build_short_prompt
            prompt = build_short_prompt(query, context_texts)
        else:
            prompt = self.llm.build_rag_prompt(query, context_texts)

        # ── Step 3: Generate ──────────────────────────────────────
        if self.recovery_version == 2:
            llm_output = self.llm.generate(prompt, max_new_tokens=64, temperature=0.0,
                                          return_logits=return_logits)
            from src.recovery.evidence_recovery import clean_generated_answer
            llm_output = dict(llm_output, raw_text=llm_output['text'],
                              text=clean_generated_answer(llm_output['text']))
        else:
            llm_output = self.llm.generate(prompt, return_logits=return_logits)

        # ── Assemble result ───────────────────────────────────────
        result = {
            "query": query,
            "answer": llm_output["text"],
            "retrieved_chunks": retrieved,
            "raw_llm_output": llm_output,
            "context_texts": context_texts,
            "prompt": prompt,
        }

        if return_embeddings:
            result["query_embedding"] = query_embedding
            result["chunk_embeddings"] = chunk_embeddings

        return result

    def run_rag_with_detection(
        self,
        query: str,
        custom_chunks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run RAG pipeline and compute all 4 failure detection signals.

        This is the main entry point for the validation benchmark.

        Returns:
            Dict with all run_rag fields plus:
                - signals: Dict with retrieval_sim, entailment, entropy, coverage scores.
        """
        from src.detection import (
            detect_retrieval_failure,
            detect_grounding_failure,
            detect_confidence_failure,
            detect_coverage_failure,
        )

        # Run pipeline with embeddings and logits
        result = self.run_rag(
            query,
            return_logits=True,
            return_embeddings=True,
            custom_chunks=custom_chunks,
        )

        # ── Compute detection signals ─────────────────────────────
        signals = {}

        # Signal 1: Retrieval similarity
        signals["retrieval_similarity"] = detect_retrieval_failure(
            query_embedding=result["query_embedding"],
            chunk_embeddings=result["chunk_embeddings"],
        )

        # Signal 2: NLI entailment
        signals["entailment_score"] = detect_grounding_failure(
            context_texts=result["context_texts"],
            answer=result["answer"],
            query=query,
        )

        # Signal 3: Token entropy
        signals["mean_entropy"] = detect_confidence_failure(
            token_logits=result["raw_llm_output"].get("token_logits"),
        )

        # Signal 4: KB coverage
        signals["coverage_score"] = detect_coverage_failure(
            query=query,
            embedder=self.embedder,
            index=self.index,
        )

        result["signals"] = signals
        from src.detection.response_check import is_refusal
        signals["is_refusal"] = is_refusal(result["answer"])
        return result


if __name__ == "__main__":
    print("Testing RAG Pipeline...")
    print("(Requires FAISS index on disk and GPU for LLM)")
    try:
        pipeline = RAGPipeline()
        result = pipeline.run_rag("What is the capital of France?")
        print(f"\nQuery: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Retrieved {len(result['retrieved_chunks'])} chunks")
    except Exception as e:
        print(f"Pipeline test failed: {e}")
