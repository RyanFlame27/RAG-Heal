# Runtime data

Included: the final runtime-adapted BERT checkpoint, its tokenizer/training manifest/grouped split, and `source_documents.json` containing the 40 fixed source paragraphs used for the corpus extension.

The accepted BERT weight file is 437,967,876 bytes; SHA-256:
`2a7c254ea237eeabd5fdf7761ee7f16c3d7a87d0f6da72d73b1df07a166ca72e`.

The original supplied project had no saved FAISS binaries. Run `python -m src.prepare_deployment` on the Colab T4 to build or reuse:

- `faiss_index/`: 366,568 Wikipedia vectors plus matching `chunks.pkl`.
- `faiss_index_with_source_passages/`: 366,742 final vectors, matching `chunks.pkl` and `source_index_manifest.json`.

The notebook persists these caches back to the project's Drive folder. It does not retrain BERT. The builder checks the expected corpus size and source-document fingerprint before allowing deployment. Named pretrained generator/audit/embedding models download from Hugging Face on first use.
