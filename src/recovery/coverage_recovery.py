"""
Recovery Action 4: Coverage Failure Recovery.

Handles out-of-domain queries by returning a polite structured fallback response
or escalating to external web search.
"""

from typing import Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


def recover_coverage_failure(
    query: str,
    pipeline=None,
    use_web_search_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Perform Fallback Escalation or Structured Out-of-Domain Response.

    Args:
        query: User question.
        pipeline: RAGPipeline instance (optional).
        use_web_search_fallback: If True, attempts external retrieval fallback.

    Returns:
        Dict with:
            - fallback_answer: Polite out-of-domain response string.
            - action_taken: Description string.
    """
    print(f"\n🛠️ [RECOVERY 4] Executing Out-of-Domain Fallback Escalation for: '{query}'")

    if use_web_search_fallback:
        raise ValueError("External web search is not implemented.")
    fallback_answer = "I don't have enough information in the indexed knowledge base to answer this question."
    action_desc = "Domain Fallback Response Issued (Graceful Degradation)"

    return {
        "fallback_answer": fallback_answer,
        "action_taken": action_desc,
    }
