"""Build/reuse the evaluated corpus without retraining; --check checks local assets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

EXPECTED_WEIGHTS = "2a7c254ea237eeabd5fdf7761ee7f16c3d7a87d0f6da72d73b1df07a166ca72e"
EXPECTED_BASE_VECTORS, EXPECTED_TOTAL_VECTORS = 366568, 366742


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_assets(require_index=True):
    checkpoint = config.BERT_CLASSIFIER_DIR
    required = [checkpoint/name for name in ["model.safetensors", "training_manifest.json",
                "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "split.json"]]
    required += [config.SOURCE_DOCUMENTS_PATH, config.FAILURE_DATASET_PATH]
    if require_index:
        required += [config.SOURCE_FAISS_INDEX_DIR/name for name in
                     ["index.faiss", "chunks.pkl", "source_index_manifest.json"]]
    missing = [str(p.relative_to(config.PROJECT_ROOT)) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing deployment assets: " + ", ".join(missing) +
                                ". Run python -m src.prepare_deployment for missing indices.")
    manifest = json.loads((checkpoint/"training_manifest.json").read_text(encoding="utf-8"))
    if sha256(checkpoint/"model.safetensors") != EXPECTED_WEIGHTS or manifest["checkpoint_sha256"] != EXPECTED_WEIGHTS:
        raise ValueError("Checkpoint is not the accepted runtime-adapted BERT v4 weights.")
    if require_index:
        manifest = json.loads((config.SOURCE_FAISS_INDEX_DIR/"source_index_manifest.json").read_text())
        if manifest.get("total_index_vectors") != EXPECTED_TOTAL_VECTORS:
            raise ValueError("Index size differs from the final evaluated corpus.")
        if manifest.get("source_documents_sha256") != sha256(config.SOURCE_DOCUMENTS_PATH):
            raise ValueError("Source passages do not match the prepared index.")
    return {"checkpoint_sha256": EXPECTED_WEIGHTS, "index_checked": require_index}


def prepare():
    check_assets(require_index=False)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Connect the Colab T4 GPU before preparing the 366K-chunk index.")
    from src.retrieval.embedder import Embedder
    from src.retrieval.index import build_index, load_index, save_index
    from src.retrieval.source_corpus import prepare_source_augmented_index
    from src.retrieval.chunker import chunk_documents
    embedder = Embedder()
    source_dir = config.SOURCE_FAISS_INDEX_DIR
    if all((source_dir/name).is_file() for name in ["index.faiss", "chunks.pkl", "source_index_manifest.json"]):
        check_assets()
        index, chunks = load_index(source_dir)
    else:
        base_dir = config.FAISS_INDEX_DIR
        if (base_dir/"index.faiss").is_file() and (base_dir/"chunks.pkl").is_file():
            index, chunks = load_index(base_dir)
        else:
            from datasets import load_dataset
            stream = load_dataset(config.WIKI_DATASET_NAME, config.WIKI_DATASET_CONFIG,
                                  split="train", streaming=True)
            articles = [{"title": row["title"], "text": row["text"]}
                        for row in stream.take(config.NUM_WIKI_ARTICLES)
                        if len(row.get("text", "").strip()) >= 100]
            chunks = chunk_documents(articles)
            index = build_index(chunks, embedder)
            if index.ntotal != EXPECTED_BASE_VECTORS:
                raise ValueError(f"Rebuilt corpus has {index.ntotal} chunks; expected {EXPECTED_BASE_VECTORS}. "
                                 "Check dataset and splitter versions before continuing.")
            save_index(index, chunks, base_dir)
        if index.ntotal != EXPECTED_BASE_VECTORS or len(chunks) != EXPECTED_BASE_VECTORS:
            raise ValueError("Existing base index differs from the final evaluated corpus.")
        data = json.loads(config.FAILURE_DATASET_PATH.read_text(encoding="utf-8"))
        split = json.loads((config.BERT_CLASSIFIER_DIR/"split.json").read_text())
        index, chunks, _ = prepare_source_augmented_index(data, split, embedder, base_dir,
                                  source_dir, config.SOURCE_DOCUMENTS_PATH)
    if index.ntotal != EXPECTED_TOTAL_VECTORS or len(chunks) != EXPECTED_TOTAL_VECTORS or index.d != 384:
        raise ValueError("Augmented index does not match evaluated corpus dimensions/count.")
    check_assets()
    print(f"Ready: {index.ntotal} local vectors and accepted BERT checkpoint. No training performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        if args.check:
            print(check_assets())
        else:
            prepare()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
