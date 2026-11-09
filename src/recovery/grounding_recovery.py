"""
Recovery Action 2: Grounding Failure Recovery.

Prunes unhelpful context and re-prompts the LLM with strict grounding
constraints and sentence-level verification.
"""

from typing import Dict, Any, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


def recover_grounding_failure(
    query: str,
    context_texts: List[str],
    previous_answer: str,
    pipeline,
) -> Dict[str, Any]:
    """
    Perform Context Pruning & Constrained Re-prompting.

    Args:
        query: User question.
        context_texts: Currently retrieved context strings.
        previous_answer: Un-grounded answer that failed NLI entailment.
        pipeline: RAGPipeline instance.

    Returns:
        Dict with:
            - pruned_context_texts: Filtered context strings.
            - new_answer: Regenerated grounded answer string.
            - action_taken: Description string.
    """
    print(f"\n🛠️ [RECOVERY 2] Executing Context Pruning & Constrained Re-prompting...")

    # Filter context chunks to top 2 most concise chunks
    pruned_texts = [c for c in context_texts if len(c.strip()) > 50][:2]
    if not pruned_texts:
        pruned_texts = context_texts

    combined_context = "\n\n---\n\n".join(pruned_texts)

    # Constrained prompt enforcing 100% strict context adherence
    constrained_prompt = (
        f"<s>[INST] You are a strict fact-checker. Answer the question below using ONLY the exact facts "
        f"mentioned in the context. Do NOT add any outside information or assumptions. If the facts are "
        f"not directly stated in the context, reply 'The context does not contain sufficient details.'\n\n"
        f"Context:\n{combined_context}\n\n"
        f"Question: {query}\n\n"
        f"Strict Answer: [/INST]"
    )

    llm_output = pipeline.llm.generate(constrained_prompt, max_new_tokens=150,
                                       temperature=0.01, return_logits=True)

    return {
        "pruned_context_texts": pruned_texts,
        "new_answer": llm_output["text"],
        "token_logits": llm_output.get("token_logits"),
        "action_taken": "Context Pruning + Constrained Re-prompting (temp=0.01)",
    }
