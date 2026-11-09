"""
Recovery Action 3: Confidence Failure Recovery.

Performs K-Expansion (expands k from 3 to 10) to supply broader context,
and sets sampling temperature to 0.0 for deterministic generation.
"""

from typing import Dict, Any, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.retrieval.index import retrieve


def recover_confidence_failure(
    query: str,
    pipeline,
    expanded_k: int = 10,
) -> Dict[str, Any]:
    """
    Perform K-Expansion & Deterministic Re-sampling.

    Args:
        query: User question.
        pipeline: RAGPipeline instance.
        expanded_k: New retrieval depth k (default 10).

    Returns:
        Dict with:
            - expanded_context_texts: List of expanded context strings.
            - new_answer: Regenerated answer.
            - action_taken: Description string.
    """
    print(f"\n🛠️ [RECOVERY 3] Executing K-Expansion (k={config.TOP_K} -> {expanded_k}) & Deterministic Sampling...")

    # Re-retrieve with larger K
    expanded_chunks = retrieve(
        query=query,
        embedder=pipeline.embedder,
        index=pipeline.index,
        chunks=pipeline.chunks,
        k=expanded_k,
    )
    expanded_context_texts = [r["text"] for r in expanded_chunks]

    prompt = pipeline.llm.build_rag_prompt(query, expanded_context_texts)
    llm_output = pipeline.llm.generate(prompt, max_new_tokens=200, temperature=0.0, return_logits=True)

    return {
        "expanded_context_texts": expanded_context_texts,
        "new_answer": llm_output["text"],
        "token_logits": llm_output.get("token_logits"),
        "action_taken": f"K-Expansion (k={expanded_k}) + Temperature set to 0.0",
    }
