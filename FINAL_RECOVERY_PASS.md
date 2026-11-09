# Accepted final recovery revision: runtime 4.2

The final deployment uses the accepted runtime-adapted BERT checkpoint, adjacent-chunk evidence windows, and recovery-only formatting of one unambiguous requested year or percentage. The complete-source-paragraph variant was rejected during validation and is not enabled.

The active checkpoint is `data/bert_failure_classifier_runtime_v4`. Its SHA-256 is `2a7c254ea237eeabd5fdf7761ee7f16c3d7a87d0f6da72d73b1df07a166ca72e`. It was adapted using 100 generated training states and the original training triples; validation and regression rows were excluded from optimization. Runtime route-validation macro F1 improved from 0.7133 to 0.9795.

The final 138-case regression run completed with zero execution errors. Across 120 reference-scored QA cases, exact match was 60/120 (50.00%) and token F1 was 51.63%. Clean cases matched 27/30 references; 60/63 answered QA cases matched exactly. All 18 designated missing-information challenges produced expected refusals. The 57 remaining QA refusals are recorded separately.

All failure routes share fresh local retrieval, adjacent-context restoration, deduplication, relevance reranking and concise generation. Retrieval failure additionally rewrites the question. A literal span in positively ranked evidence can be returned while remaining unverified by the unchanged signal audit. A supported initial answer is retained when a retry weakens it; unsupported retries otherwise lead to refusal.

The main inference/recovery code remains byte-identical to the accepted evaluated source. The deployment update connects that code to API startup, the original HTML frontend and the Colab notebook; it does not introduce a new inference algorithm or claim a fresh model evaluation. Saved evidence is included under `benchmark/final_results/`.
