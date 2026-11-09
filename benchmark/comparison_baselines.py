"""GPU adapters for Self-RAG and an explicitly modified, local-only CRAG.

Sources:
https://github.com/AkariAsai/self-rag (MIT; short-form reflection scoring)
https://github.com/HuskyInSalt/CRAG (T5 evaluator and evidence refinement)

These are Transformers/4-bit ports for this experiment, not paper reproductions.
Local CRAG requires the authors' released evaluator; no substitute is loaded.
"""

import math
from pathlib import Path
import re

SELF_RAG_MODEL = "selfrag/selfrag_llama2_7b"
CRAG_EVALUATOR_URL = "https://drive.google.com/drive/folders/1CRFGsyNguXJwKSvFvJm_82GOOlkWSkW7"
REFLECTION = {
    "retrieve": ("[Retrieval]", "[No Retrieval]"),
    "relevance": ("[Relevant]", "[Irrelevant]"),
    "grounding": ("[Fully supported]", "[Partially supported]", "[No support / Contradictory]"),
    "utility": tuple(f"[Utility:{n}]" for n in range(1, 6)),
}


def retrieval_action(scores, upper=0.592, lower=-0.995):
    if not lower < upper:
        raise ValueError("CRAG lower threshold must be below upper threshold.")
    if not scores or not all(math.isfinite(x) for x in scores):
        raise ValueError("CRAG evaluator returned empty or non-finite scores.")
    best = max(scores)
    return "correct" if best >= upper else ("ambiguous" if best >= lower else "incorrect")


def sentence_strips(contexts):
    result = []
    for context in contexts:
        sentences = re.split(r"(?<=[.!?])\s+", context.strip())
        for i in range(0, len(sentences), 3):
            text = " ".join(sentences[i:i + 3]).strip()
            if text and text not in result:
                result.append(text)
    return result


class LocalCrag:
    """Original T5 evaluator; local FAISS retrieval replaces external web search.

Other declared adaptations: same Mistral generator/prompt as our system,
    no GPT query rewriting, Transformers inference and sentence-based strips.
    PopQA evaluator thresholds are fixed in advance, not tuned on this dataset.
    """

    name = "crag_local_modified"

    def __init__(self, pipeline, evaluator_path, upper=0.592, lower=-0.995,
                 evaluator_device="cpu"):
        import torch
        from transformers import T5Tokenizer, T5ForSequenceClassification
        evaluator_path = Path(evaluator_path)
        if not (evaluator_path / "config.json").is_file():
            raise FileNotFoundError("Download the released CRAG evaluator first; no proxy evaluator is substituted.")
        self.pipeline = pipeline
        self.upper, self.lower = upper, lower
        self.tokenizer = T5Tokenizer.from_pretrained(evaluator_path)
        # Float32 preserves the released evaluator's precision. GPU placement
        # is optional when enough memory remains beside the generator and NLI.
        self.device = torch.device(evaluator_device)
        self.model = T5ForSequenceClassification.from_pretrained(
            evaluator_path, num_labels=1, torch_dtype=torch.float32).to(self.device).eval()
        self.description = {
            "name": self.name, "evaluator_source": CRAG_EVALUATOR_URL,
            "evaluator_device": str(self.device), "evaluator_dtype": "float32",
            "upper": upper, "lower": lower,
            "generator": pipeline.llm.model_name,
            "generator_revision": getattr(pipeline.llm.model.config, "_commit_hash", None),
            "adaptations": ["local FAISS retrieval instead of web search", "no GPT query rewriting",
                            "project Mistral generator and prompt", "sentence strips grouped in threes",
                            "original PopQA thresholds, not calibrated here"],
        }
        from benchmark.run_injected_eval import digest
        self.description["evaluator_files_sha256"] = {
            p.name: digest(p) for p in sorted(evaluator_path.iterdir())
            if p.is_file() and p.suffix in (".json", ".safetensors", ".bin", ".model")}

    def score(self, query, contexts):
        import torch
        values = []
        for text in contexts:
            if len(text.split()) < 4:
                values.append(-1.0)
                continue
            inputs = self.tokenizer(query + " [SEP] " + text, return_tensors="pt",
                                    padding="max_length", truncation=True, max_length=512).to(self.device)
            with torch.inference_mode():
                value = float(self.model(**inputs).logits.reshape(-1)[0])
            if not math.isfinite(value):
                raise ValueError("CRAG evaluator produced a non-finite relevance score.")
            values.append(value)
        return values

    def answer(self, item):
        from src.retrieval.index import retrieve
        initial = item["context_chunks"]
        scores = self.score(item["query"], initial)
        action = retrieval_action(scores, self.upper, self.lower)
        local = []
        if action != "correct":
            hits = retrieve(item["query"], self.pipeline.embedder,
                            self.pipeline.index, self.pipeline.chunks, k=self.pipeline.top_k)
            local = [hit["text"] for hit in hits]
        candidates = initial if action == "correct" else (
            local if action == "incorrect" else initial + local)
        strips = sentence_strips(candidates)
        if not strips:
            raise ValueError("No evidence available for local CRAG generation.")
        strip_scores = self.score(item["query"], strips)
        ranked = sorted(range(len(strips)), key=lambda i: strip_scores[i], reverse=True)[:6]
        contexts = [strips[i] for i in ranked]
        generated = self.pipeline.llm.generate(
            self.pipeline.llm.build_rag_prompt(item["query"], contexts), return_logits=True)
        return {"answer": generated["text"], "contexts": contexts,
                "token_logits": generated.get("token_logits"),
                "trace": {"decision": action, "initial_evaluator_scores": scores,
                          "selected_strip_scores": [strip_scores[i] for i in ranked]}}


class SelfRag:
    """Released reflection-trained checkpoint with short-form candidate scoring.

    Uses the supplied injected passages, adaptive retrieval threshold 0.2,
    and relevance/support/utility plus sequence scoring. No reflection training
    or generic self-critique prompt is substituted for the released checkpoint.
    """

    name = "selfrag_7b_4bit_shortform"

    def __init__(self, model_name=SELF_RAG_MODEL, max_new_tokens=256, threshold=0.2):
        import torch
        from src.generation.llm import LLMWrapper
        self.wrapper = LLMWrapper(model_name=model_name)
        self.model, self.tokenizer = self.wrapper.model, self.wrapper.tokenizer
        self.max_new_tokens, self.threshold = max_new_tokens, threshold
        self.groups = {}
        for key, tokens in REFLECTION.items():
            ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
            if any(i is None or i == self.tokenizer.unk_token_id for i in ids):
                raise ValueError("Checkpoint is missing Self-RAG reflection tokens.")
            if any(self.tokenizer.encode(token, add_special_tokens=False) != [i]
                   for token, i in zip(tokens, ids)):
                raise ValueError("Reflection tokens must each be one token.")
            self.groups[key] = ids
        self.control_ids = set(sum(self.groups.values(), []))
        for token in ("[Continue to Use Evidence]", "<paragraph>", "</paragraph>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                self.control_ids.add(token_id)
        self.description = {
            "name": self.name, "model": model_name,
            "model_revision": getattr(self.model.config, "_commit_hash", None),
            "quantization": "NF4 4-bit", "retrieval_threshold": threshold,
            "weights": {"relevance": 1.0, "support": 1.0, "utility": 0.5, "sequence": 1.0},
            "max_new_tokens": max_new_tokens,
            "adaptations": ["Transformers port of short-form scoring, not original vLLM runtime",
                            "supplied injected passages instead of Contriever",
                            "no long-form segment beam search", "greedy decoding"],
        }

    def _inputs(self, prompt):
        return self.tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=2048).to(self.model.device)

    def _generate(self, prompt):
        import torch
        inputs = self._inputs(prompt)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True, output_logits=True)
        ids = outputs.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        logits = outputs.logits
        chosen_logprobs = [float(torch.log_softmax(step[0].float(), dim=-1)[token])
                           for step, token in zip(logits, ids)]
        sequence_score = math.exp(sum(chosen_logprobs) / len(ids)) if ids else 0.0
        group_scores = {}
        for group in ("relevance", "grounding", "utility"):
            group_ids = self.groups[group]
            position = next((i for i, token in enumerate(ids) if token in group_ids), None)
            if position is None:
                group_scores[group] = None
            else:
                group_scores[group] = torch.softmax(
                    logits[position][0, group_ids].float(), dim=-1).cpu().tolist()
        text_ids = [token for token in ids if token not in self.control_ids]
        answer = self.tokenizer.decode(text_ids, skip_special_tokens=True).strip()
        token_info = self.wrapper._extract_token_logits(logits)
        token_info = [info for info, token in zip(token_info, ids)
                      if token not in self.control_ids and token not in self.tokenizer.all_special_ids]
        rel = (group_scores["relevance"] or [0.0])[0]
        grounding = group_scores["grounding"] or [0.0, 0.0, 0.0]
        utility = group_scores["utility"] or [0.0] * 5
        support = grounding[0] + 0.5 * grounding[1]
        usefulness = sum(p * weight for p, weight in zip(utility, (-1, -0.5, 0, 0.5, 1)))
        score = sequence_score + rel + support + 0.5 * usefulness
        return {"answer": answer, "token_logits": token_info,
                "score": score, "reflection_probabilities": group_scores}

    def answer(self, item):
        import torch
        prompt = f"### Instruction:\n{item['query']}\n\n### Response:\n"
        with torch.inference_mode():
            next_logits = self.model(**self._inputs(prompt)).logits[0, -1]
            probabilities = torch.softmax(next_logits[self.groups["retrieve"]].float(), dim=-1)
            retrieval_probability = float(probabilities[0])
        if retrieval_probability < self.threshold:
            result = self._generate(prompt + "[No Retrieval]")
            result["contexts"] = []
            trace = {"retrieval_probability": retrieval_probability, "used_retrieval": False}
        else:
            candidates = []
            for context in item["context_chunks"]:
                candidate = self._generate(prompt + "[Retrieval]<paragraph>" + context + "</paragraph>")
                candidate["contexts"] = [context]
                candidates.append(candidate)
            result = max(candidates, key=lambda c: c["score"])
            trace = {"retrieval_probability": retrieval_probability, "used_retrieval": True,
                     "candidate_scores": [c["score"] for c in candidates]}
        trace["reflection_probabilities"] = result["reflection_probabilities"]
        return {"answer": result["answer"], "contexts": result["contexts"],
                "token_logits": result["token_logits"], "trace": trace}
