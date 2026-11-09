"""Build the static source-augmented index without running model evaluation."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def main():
    from src.retrieval.embedder import Embedder
    from src.retrieval.source_corpus import prepare_source_augmented_index
    data = json.loads(config.FAILURE_DATASET_PATH.read_text(encoding='utf-8'))
    split = json.loads((config.BERT_CLASSIFIER_DIR/'split.json').read_text(encoding='utf-8'))
    index, _, bundle = prepare_source_augmented_index(data, split, Embedder(),
        config.FAISS_INDEX_DIR, config.SOURCE_FAISS_INDEX_DIR, config.SOURCE_DOCUMENTS_PATH)
    print('Source index ready:', index.ntotal, 'vectors;',
          bundle['provenance']['unique_source_documents'], 'source documents')
    print('Pass this index to RAGPipeline, or use the default recovery v2 loader.')


if __name__ == '__main__':
    main()
