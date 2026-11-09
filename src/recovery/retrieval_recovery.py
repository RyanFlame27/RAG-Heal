"""
Recovery Action 1: Retrieval Failure Recovery.

Uses LLM query rewriting to reformulate the user query into better search
keywords, then re-queries the FAISS index for relevant chunks.
"""

from typing import Dict, Any, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.retrieval.index import retrieve


def recover_retrieval_failure(
    query: str,
    pipeline,
) -> Dict[str, Any]:
    """
    Perform Query Rewriting and Re-retrieval.

    Args:
        query: Original user question that failed retrieval.
        pipeline: RAGPipeline instance.

    Returns:
        Dict with:
            - rewritten_query: Formulated search query.
            - new_context_texts: List of updated context chunk strings.
            - new_retrieved_chunks: List of retrieved chunk dicts.
            - action_taken: Description string.
    """
    print(f"\n🛠️ [RECOVERY 1] Executing Query Rewriting for: '{query}'")

    # Prompt LLM to extract key entities / rewrite for search
    rewrite_prompt = (
        f"<s>[INST] Rephrase the following user question into 3-5 concise, specific search keywords "
        f"suitable for retrieving Wikipedia facts. Output ONLY the search keywords.\n\n"
        f"Question: {query} [/INST]"
    )

    try:
        llm_res = pipeline.llm.generate(rewrite_prompt, max_new_tokens=40, return_logits=False)
        rewritten_query = llm_res["text"].strip().replace('"', '')
    except Exception as e:
        print(f"  Warning during query rewrite generation: {e}")
        rewritten_query = f"{query} details background summary"

    print(f"  Rewritten Search Query: '{rewritten_query}'")

    # Re-query FAISS with rewritten query
    new_retrieved = retrieve(
        query=rewritten_query,
        embedder=pipeline.embedder,
        index=pipeline.index,
        chunks=pipeline.chunks,
        k=config.TOP_K,
    )
    new_context_texts = [r["text"] for r in new_retrieved]

    return {
        "rewritten_query": rewritten_query,
        "new_context_texts": new_context_texts,
        "new_retrieved_chunks": new_retrieved,
        "action_taken": f"Query Rewriting: '{query}' -> '{rewritten_query}'",
    }
