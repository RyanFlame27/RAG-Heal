"""Signal 2: probability that the retrieved evidence entails the answer.

Refusals are classified separately; neutral content is not treated as support.
"""

import torch
import re
from typing import List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

_nli_model = None
_nli_tokenizer = None

def _load_nli_model():
    """Load the NLI model and tokenizer (singleton pattern)."""
    global _nli_model, _nli_tokenizer

    if _nli_model is not None:
        return _nli_model, _nli_tokenizer

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = config.NLI_MODEL_NAME
    print(f"Loading NLI model: {model_name}...")

    _nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _nli_model = _nli_model.to(device)
    _nli_model.eval()
    print(f"NLI model loaded on {device}.")

    return _nli_model, _nli_tokenizer


def detect_grounding_failure(
    context_texts: List[str],
    answer: str,
    max_context_length: int = 3000,
    query: Optional[str] = None,
) -> float:
    """
    Compute the NLI entailment probability for context and answer.
    Returns a float in [0.0, 1.0]. High = supported; low = contradiction or neutral/unsupported content.
    """
    if not context_texts or not answer or not answer.strip():
        return 0.0

    model, tokenizer = _load_nli_model()
    device = next(model.parameters()).device

    combined_context = " ".join(context_texts)
    if len(combined_context) > max_context_length:
        combined_context = combined_context[:max_context_length]

    # Truncate answer to fit token window
    # QA datasets often store a bare answer span (e.g. a name). Restore the
    # question relation before asking NLI to judge that span as a proposition.
    hypothesis = answer
    if query and len(answer.split()) <= 12 and not re.search(r"[.!?]", answer):
        hypothesis = f"The answer to the question '{query}' is {answer}."
    truncated_answer = hypothesis[:512]

    inputs = tokenizer(
        combined_context,
        truncated_answer,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)[0]

    # Identify contradiction and entailment indices dynamically
    contradiction_idx = 0
    entailment_idx = 1
    if hasattr(model.config, "id2label") and model.config.id2label:
        for idx, label in model.config.id2label.items():
            lbl = label.lower()
            if "contradict" in lbl:
                contradiction_idx = int(idx)
            elif "entail" in lbl:
                entailment_idx = int(idx)

    contradiction_prob = probs[contradiction_idx].item()
    entailment_prob = probs[entailment_idx].item()

    # Neutral text is not evidence of support. Refusals are handled separately.
    return float(entailment_prob)


def is_grounding_failure(
    score: float,
    threshold: float = config.ENTAILMENT_THRESHOLD,
) -> bool:
    """Check if the entailment score indicates a grounding failure."""
    return score < threshold
