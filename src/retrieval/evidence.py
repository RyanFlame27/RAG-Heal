"""Retrieve local evidence, restore adjacent context and rerank question/passage pairs."""

import math
import re

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def unique_texts(texts):
    seen, result = set(), []
    for text in texts:
        text = text.strip()
        key = re.sub(r"\s+", " ", text).casefold()
        if key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def merge_overlapping(parts):
    """Remove only exact character overlap; retain source wording."""
    result = ""
    for part in parts:
        part = part.strip()
        overlap = 0
        for length in range(min(len(result), len(part)), 9, -1):
            if result.endswith(part[:length]):
                overlap = length
                break
        result += (" " if result and not overlap else "") + part[overlap:]
    return result


class EvidenceRetriever:
    def __init__(self, pipeline, model_name=RERANKER_MODEL, device="cuda:0", full_source_paragraphs=False):
        self.pipeline = pipeline
        self.model_name, self.device = model_name, device
        self.model = self.tokenizer = None
        self.locations = {}
        source_parts = {}
        for chunk in pipeline.chunks:
            metadata = chunk.get("metadata", {})
            if "title" in metadata and "chunk_index" in metadata:
                key = (metadata.get("source"), metadata["title"], metadata["chunk_index"])
                self.locations[key] = chunk["text"]
                if full_source_paragraphs and metadata.get('source') == 'squad_original_source_passage':
                    source_parts.setdefault(key[:2], []).append((metadata['chunk_index'],chunk['text']))
        # A short source paragraph is one evidence unit. Returning overlapping
        # windows from it can waste all three slots and omit the answer sentence.
        self.source_passages = {key:merge_overlapping(text for _,text in sorted(parts))
                                for key,parts in source_parts.items()}

    def _load(self):
        if self.model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name).to(self.device).eval()

    def score(self, query, passages):
        import torch
        self._load()
        values = []
        for start in range(0, len(passages), 16):
            batch = passages[start:start+16]
            features = self.tokenizer([query]*len(batch), batch, padding=True,
                                      truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                values.extend(self.model(**features).logits.flatten().float().cpu().tolist())
        if len(values) != len(passages) or not all(math.isfinite(x) for x in values):
            raise ValueError("Non-finite or incomplete evidence reranking scores.")
        return values

    def window(self, hit, radius=1):
        metadata = hit.get("metadata", {})
        if "title" not in metadata or "chunk_index" not in metadata:
            return hit["text"]
        source, title, position = metadata.get("source"), metadata["title"], metadata["chunk_index"]
        if (source,title) in self.source_passages:
            return self.source_passages[(source,title)]
        pieces = [self.locations[(source, title, i)] for i in range(max(0, position-radius), position+radius+1)
                  if (source, title, i) in self.locations]
        return merge_overlapping(pieces) if pieces else hit["text"]

    def retrieve(self, query, rewritten_query=None, candidate_k=24, keep=3):
        from src.retrieval.index import retrieve
        hits = retrieve(query, self.pipeline.embedder, self.pipeline.index,
                        self.pipeline.chunks, k=candidate_k)
        if rewritten_query and rewritten_query.strip().casefold() != query.strip().casefold():
            hits += retrieve(rewritten_query, self.pipeline.embedder, self.pipeline.index,
                             self.pipeline.chunks, k=candidate_k)
        passages = unique_texts(self.window(hit) for hit in hits)
        if not passages:
            return [], []
        scores = self.score(query, passages)
        ranked = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)[:keep]
        return [passages[i] for i in ranked], [scores[i] for i in ranked]
