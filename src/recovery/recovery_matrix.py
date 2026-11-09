"""Route one recovery action and audit its actual output."""

from src.detection import (
    detect_retrieval_failure, detect_grounding_failure,
    detect_confidence_failure, detect_coverage_failure,
)
from src.detection.response_check import assess_response, is_refusal
from .retrieval_recovery import recover_retrieval_failure
from .grounding_recovery import recover_grounding_failure
from .confidence_recovery import recover_confidence_failure
from .coverage_recovery import recover_coverage_failure


class RecoveryMatrix:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def _audit(self, query, answer, contexts, token_logits):
        embedder = self.pipeline.embedder
        query_embedding = embedder.embed_query(query)
        chunk_embeddings = embedder.embed_documents(contexts, show_progress=False) if contexts else []
        return {
            "retrieval_similarity": detect_retrieval_failure(query_embedding, chunk_embeddings),
            "entailment_score": detect_grounding_failure(contexts, answer, query=query),
            "mean_entropy": detect_confidence_failure(token_logits),
            "coverage_score": detect_coverage_failure(query, embedder, self.pipeline.index),
            "is_refusal": is_refusal(answer),
        }

    def heal(self, query, initial_result, predicted_failure):
        if getattr(self.pipeline, "recovery_version", 1) == 2:
            from .evidence_recovery import heal_with_evidence
            return heal_with_evidence(self, query, initial_result, predicted_failure)
        before = initial_result["signals"]
        contexts = initial_result["context_texts"]
        answer = initial_result["answer"]
        action = "None (initial response retained)"
        healed = predicted_failure != "clean"

        if predicted_failure == "clean":
            after = dict(before)
        elif predicted_failure == "coverage_failure":
            result = recover_coverage_failure(query, self.pipeline)
            answer = result["fallback_answer"]
            action = result["action_taken"]
            after = dict(before, is_refusal=is_refusal(answer))
            after["mean_entropy"] = None
            after["entailment_score"] = 0.0
        else:
            if predicted_failure == "retrieval_failure":
                result = recover_retrieval_failure(query, self.pipeline)
                contexts = result["new_context_texts"]
                generated = self.pipeline.llm.generate(
                    self.pipeline.llm.build_rag_prompt(query, contexts), return_logits=True)
                answer = generated["text"]
                token_logits = generated.get("token_logits")
            elif predicted_failure == "grounding_failure":
                result = recover_grounding_failure(query, contexts, answer, self.pipeline)
                contexts = result["pruned_context_texts"]
                answer = result["new_answer"]
                token_logits = result.get("token_logits")
            elif predicted_failure == "confidence_failure":
                result = recover_confidence_failure(query, self.pipeline, expanded_k=10)
                contexts = result["expanded_context_texts"]
                answer = result["new_answer"]
                token_logits = result.get("token_logits")
            else:
                raise ValueError(f"Unknown failure mode: {predicted_failure}")
            action = result["action_taken"]
            after = self._audit(query, answer, contexts, token_logits)

        assessment = assess_response(answer, after)
        return {
            "healed_answer": answer,
            "action_taken": action,
            "healed": healed,
            "recovery_mode": predicted_failure,
            "verified": assessment["accepted"],
            "assessment": assessment,
            "pre_signals": before,
            "post_signals": after,
            "post_context_texts": contexts,
            "verification_notes": {
                "supported": "Response passed all signal checks.",
                "refusal": "Abstention issued; no factual-answer verification claimed.",
                "unverified": "Response did not pass all signal checks.",
            }[assessment["outcome"]],
        }
