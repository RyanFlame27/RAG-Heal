"""Shared response checks; abstention is recorded separately from grounding."""

import math
import re
import config


def is_refusal(answer: str) -> bool:
    """Recognize conservative abstentions without accepting appended factual claims."""
    text = re.sub(r"\s+", " ", answer.strip().lower())
    if not text or re.search(r"\b(but|however|although)\b", text):
        return False
    sentences = re.split(r"(?<=[.!?])\s+", text)
    patterns = (
        r"i (?:don't|do not) have (?:enough|sufficient) information\b.*",
        r"i (?:cannot|can't|am unable to) answer\b.*",
        r"(?:the |this |provided )?context(?: provided)? (?:does not|doesn't|contains no)\b.*",
        r"(?:there is |there's )?(?:not enough|insufficient|no) information\b.*",
    )
    return all(any(re.fullmatch(pattern, sentence) for pattern in patterns)
               for sentence in sentences)


def assess_response(answer, signals, allow_refusal=True):
    """Apply identical acceptance rules before and after recovery.

    This is a signal-based policy check, not an independent factuality judgment.
    A refusal never counts as a grounded answer.
    """
    refusal = is_refusal(answer)
    names = ("retrieval_similarity", "entailment_score", "mean_entropy", "coverage_score")
    values = [signals.get(name) for name in names]
    available = all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in values)
    supported = bool(answer.strip()) and not refusal and available and (
        values[0] >= config.RETRIEVAL_SIM_THRESHOLD
        and values[1] >= config.ENTAILMENT_THRESHOLD
        and values[2] < config.ENTROPY_THRESHOLD
        and values[3] >= config.COVERAGE_SIM_THRESHOLD
    )
    return {
        "accepted": bool(supported or (refusal and allow_refusal)),
        "grounded_answer": bool(supported),
        "refusal": refusal,
        "signals_available": available,
        "outcome": "refusal" if refusal else ("supported" if supported else "unverified"),
    }
