"""Build a static reference corpus from source passages matched by training questions."""

import hashlib
import json
from pathlib import Path
from src.classifier.bert_data import question_key

SOURCE_DATASET = 'rajpurkar/squad'


def collect_training_source_documents(source_rows, training_questions):
    """Select original passages, never answer fields or evaluation questions."""
    wanted = {question_key(query) for query in training_questions}
    documents, matched = {}, set()
    for row in source_rows:
        query = question_key(row['question'])
        if query not in wanted:
            continue
        text = row['context'].strip()
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        matched.add(query)
        documents.setdefault(digest, {'title': f"{row['title']} / passage {digest[:12]}",
                                     'text': text, 'sha256': digest,
                                     'source': SOURCE_DATASET})
    return list(documents.values()), sorted(matched)


def load_training_source_documents(data, split, max_source_rows=10000):
    from datasets import load_dataset
    questions = [data[i]['query'] for i in split['indices']['train']]
    source = load_dataset(SOURCE_DATASET, split='train', streaming=True)
    documents, matched = collect_training_source_documents(source.take(max_source_rows), questions)
    if not documents:
        raise RuntimeError('No original source passages matched training questions.')
    provenance = {'source_dataset': SOURCE_DATASET, 'source_split': 'train',
                  'source_rows_examined_at_most': max_source_rows,
                  'selection': 'Exact normalised question matches from classifier training IDs only',
                  'training_questions_matched': len(matched), 'unique_source_documents': len(documents),
                  'matched_training_question_keys': matched,
                  'indexed_fields': ['original source context text'],
                  'excluded_fields': ['questions', 'answer annotations', 'constructed class labels'],
                  'evaluation_questions_used_for_selection': False}
    return documents, provenance


def prepare_source_augmented_index(data, split, embedder, base_index_dir, output_dir, documents_path):
    """Extend a saved index with original passages selected using training IDs."""
    from src.classifier.bert_data import dataset_hash
    from src.retrieval.chunker import chunk_documents
    from src.retrieval.index import load_index, save_index
    documents_path, output_dir = Path(documents_path), Path(output_dir)
    training_signature = hashlib.sha256(json.dumps(sorted(split['indices']['train'])).encode()).hexdigest()
    if documents_path.exists():
        bundle = json.loads(documents_path.read_text(encoding='utf-8'))
        assert bundle['provenance']['dataset_sha256'] == dataset_hash(data)
        assert bundle['provenance']['training_ids_sha256'] == training_signature
    else:
        documents, provenance = load_training_source_documents(data, split)
        provenance.update(dataset_sha256=dataset_hash(data), training_ids_sha256=training_signature)
        bundle = {'documents': documents, 'provenance': provenance}
        documents_path.parent.mkdir(parents=True, exist_ok=True)
        documents_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest_path = output_dir/'source_index_manifest.json'
    document_signature = hashlib.sha256(documents_path.read_bytes()).hexdigest()
    if manifest_path.exists():
        index_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert index_manifest['source_documents_sha256'] == document_signature
        index, chunks = load_index(output_dir)
        return index, chunks, bundle
    index, chunks = load_index(base_index_dir)
    base_vectors = index.ntotal
    source_chunks = chunk_documents(bundle['documents'])
    for chunk in source_chunks:
        chunk['metadata']['source'] = 'squad_original_source_passage'
    vectors = embedder.embed_documents([chunk['text'] for chunk in source_chunks], show_progress=False)
    index.add(vectors)
    chunks = chunks + source_chunks
    save_index(index, chunks, output_dir)
    manifest = {'base_index_vectors': base_vectors, 'added_source_chunks': len(source_chunks),
                'total_index_vectors': index.ntotal, 'source_documents_sha256': document_signature}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return index, chunks, bundle
