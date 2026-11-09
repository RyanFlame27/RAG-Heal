"""Evidence-selection and answer-preservation checks without model downloads."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
evidence = load('evidence_under_test', 'src/retrieval/evidence.py')
recovery = load('recovery_under_test', 'src/recovery/evidence_recovery.py')
policy = load('policy_under_test', 'src/detection/response_check.py')


class EvidenceRecoveryTests(unittest.TestCase):
    def test_source_corpus_uses_training_matches_and_original_passages_only(self):
        collect_training_source_documents = load('source_corpus_under_test',
            'src/retrieval/source_corpus.py').collect_training_source_documents
        rows = [dict(question='Training question?', context='Original source paragraph.',
                     title='Article', answers={'text':['ANSWER_ANNOTATION_NOT_FOR_INDEX']}),
                dict(question='Different training question?', context='Original source paragraph.',
                     title='Article', answers={'text':['OTHER_ANNOTATION']}),
                dict(question='Test-only question?', context='TEST_ONLY_PARAGRAPH',
                     title='Other article', answers={'text':['SECRET']})]
        documents, matched = collect_training_source_documents(rows, ['Training question?', 'Different training question?'])
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]['text'], 'Original source paragraph.')
        self.assertEqual(len(matched), 2)
        self.assertNotIn('ANNOTATION', str(documents))
        self.assertNotIn('TEST_ONLY', str(documents))

    def test_windows_do_not_cross_articles_and_remove_exact_overlap(self):
        chunks = [dict(text='A nearby sentence about Paris.', metadata=dict(source='wiki', title='A', chunk_index=0)),
                  dict(text='about Paris. The city is in France.', metadata=dict(source='wiki', title='A', chunk_index=1)),
                  dict(text='Unrelated other article.', metadata=dict(source='wiki', title='B', chunk_index=0))]
        retriever = evidence.EvidenceRetriever(SimpleNamespace(chunks=chunks))
        window = retriever.window(chunks[1])
        self.assertEqual(window, 'A nearby sentence about Paris. The city is in France.')
        self.assertNotIn('Unrelated', window)

    def test_extractive_span_has_word_boundaries_and_normalises_punctuation(self):
        self.assertTrue(recovery.span_in_evidence('Max Scheler', ['His advisor was Max Scheler.']))
        self.assertTrue(recovery.span_in_evidence('$400m', ['The project cost $400m.']))
        self.assertFalse(recovery.span_in_evidence('Rome', ['Arometric unrelated text.']))
        self.assertFalse(recovery.span_in_evidence('Berlin', ['The city is Paris.']))

    def test_source_paragraph_is_restored_and_duplicate_windows_collapse(self):
        chunks=[dict(text=text,metadata=dict(source='squad_original_source_passage',title='A',chunk_index=i))
                for i,text in enumerate(['First sentence.','Second sentence.','Third sentence.','The answer is Paris.'])]
        retriever=evidence.EvidenceRetriever(SimpleNamespace(chunks=chunks),full_source_paragraphs=True)
        windows=[retriever.window(hit) for hit in chunks]
        self.assertIn('The answer is Paris.',windows[0])
        self.assertEqual(len(evidence.unique_texts(windows)),1)

    def test_output_parser_removes_added_tasks_and_keeps_raw_answer_separate(self):
        self.assertEqual(recovery.clean_generated_answer('39.1%\n\nQuestion: Another question?\nAnswer: invented'), '39.1%')
        self.assertEqual(recovery.clean_generated_answer('Answer: Max Scheler'), 'Max Scheler')
        self.assertEqual(recovery.clean_generated_answer(recovery.ABSTENTION+' (An invented explanation.)'), recovery.ABSTENTION)

    def test_requested_year_and_percentage_are_extracted_from_supported_sentences(self):
        self.assertEqual(recovery.requested_value('In what year was the college formed?',
            'The college was formed in 1920.',['The college opened in 1920.']),'1920')
        self.assertEqual(recovery.requested_value('What percentage took part?',
            '39.1% of students took part.',['The proportion was 39.1%.']),'39.1%')
        self.assertEqual(recovery.requested_value('What percentage took part?',
            'Over 80% took part.',['Over 80% participated.']),'Over 80%')

    def test_requested_value_does_not_choose_ambiguous_or_absent_numbers(self):
        text='The college opened in 1920 and changed in 1950.'
        self.assertEqual(recovery.requested_value('In what year did it open?',text,[text]),text)
        self.assertEqual(recovery.requested_value('In what year did it open?','It opened in 1920.',['It opened in 1950.']),'It opened in 1920.')
        self.assertEqual(recovery.requested_value('Who founded it?','A person in 1920.',['A person in 1920.']),'A person in 1920.')
        self.assertEqual(recovery.requested_value('In what year did it open?','Perhaps 1920.',['It opened in 1920.']),'Perhaps 1920.')
        self.assertEqual(recovery.requested_value('What percentage took part?','Not more than 80%.',['More than 80%.']),'Not more than 80%.')

    def test_deduplication_keeps_first_original_wording(self):
        self.assertEqual(evidence.unique_texts([' Paris is a city. ', 'PARIS  IS A CITY.', 'Berlin']),
                         ['Paris is a city.', 'Berlin'])

    def test_pipeline_keeps_original_mode_and_runs_concise_candidate_mode(self):
        calls = []
        def generate(prompt, **kwargs):
            calls.append((prompt, kwargs))
            return dict(text='Paris', token_logits=[])
        llm = SimpleNamespace(build_rag_prompt=lambda *args: 'Original prompt', generate=generate)
        with patch.dict(sys.modules, {
                'src.retrieval.embedder': SimpleNamespace(Embedder=object),
                'src.retrieval.index': SimpleNamespace(load_index=None, retrieve=None, retrieve_with_embeddings=None),
                'src.generation.llm': SimpleNamespace(LLMWrapper=object),
                'src.recovery': ModuleType('src.recovery'),
                'src.recovery.evidence_recovery': recovery}):
            module = load('pipeline_under_test', 'src/pipeline/rag.py')
            for version in (1, 2):
                pipeline = module.RAGPipeline(embedder=object(), llm=llm,
                    index=SimpleNamespace(ntotal=1), chunks=[], recovery_version=version)
                result = pipeline.run_rag('Capital of France?', custom_chunks=['The capital is Paris.'])
                self.assertEqual(result['answer'], 'Paris')
                self.assertEqual(result['context_texts'], ['The capital is Paris.'])
        self.assertEqual(calls[0], ('Original prompt', {'return_logits':True}))
        self.assertEqual(calls[1][1]['temperature'], 0.0)
        self.assertEqual(calls[1][1]['max_new_tokens'], 64)

    def execute(self, candidate, candidate_signals, initial_supported=False, route='grounding_failure'):
        good = dict(retrieval_similarity=.8, entailment_score=.9, mean_entropy=.2, coverage_score=.8)
        initial = dict(answer='Paris' if initial_supported else recovery.ABSTENTION,
                       context_texts=['Old unhelpful context.'], signals=good)
        retrieve_calls = []
        def retrieve(query, **kwargs):
            retrieve_calls.append(query)
            return ['The capital of France is Paris.'], [6.0]
        pipeline = SimpleNamespace(evidence_retriever=SimpleNamespace(retrieve=retrieve),
            llm=SimpleNamespace(generate=lambda *args, **kwargs: dict(text=candidate, token_logits=[])))
        matrix = SimpleNamespace(pipeline=pipeline, _audit=lambda *args: candidate_signals)
        detection = ModuleType('src.detection')
        detection.__path__ = []
        with patch.dict(sys.modules, {'src.detection': detection, 'src.detection.response_check': policy}):
            result = recovery.heal_with_evidence(matrix, 'Capital of France?', initial, route)
        self.assertEqual(retrieve_calls, ['Capital of France?'])
        return result

    def test_default_loader_uses_source_index_only_for_evidence_mode(self):
        import config
        calls = []
        def load_index(path=None):
            calls.append(path)
            return SimpleNamespace(ntotal=1), []
        with tempfile.TemporaryDirectory() as directory, patch.object(config, 'SOURCE_FAISS_INDEX_DIR', Path(directory)), patch.dict(sys.modules, {
                'src.retrieval.embedder': SimpleNamespace(Embedder=object),
                'src.retrieval.index': SimpleNamespace(load_index=load_index, retrieve=None, retrieve_with_embeddings=None),
                'src.generation.llm': SimpleNamespace(LLMWrapper=object)}):
            module = load('pipeline_loader_under_test', 'src/pipeline/rag.py')
            for name in ('index.faiss','chunks.pkl'):
                (Path(directory)/name).touch()
            for version in (1,2):
                module.RAGPipeline(embedder=object(),llm=object(),recovery_version=version)
            self.assertEqual(calls,[None,Path(directory)])

    def test_grounding_repair_retrieves_fresh_evidence_and_returns_supported_span(self):
        signals = dict(retrieval_similarity=.8, entailment_score=.9, mean_entropy=.2, coverage_score=.8)
        result = self.execute('Paris', signals)
        self.assertEqual(result['healed_answer'], 'Paris')
        self.assertEqual(result['post_context_texts'], ['The capital of France is Paris.'])
        self.assertTrue(result['recovery_trace']['candidate_used'])

    def test_unsupported_candidate_is_not_promoted_by_high_nli_alone(self):
        signals = dict(retrieval_similarity=.8, entailment_score=.99, mean_entropy=.2, coverage_score=.8)
        result = self.execute('Berlin', signals)
        self.assertEqual(result['assessment']['outcome'], 'refusal')
        self.assertFalse(result['recovery_trace']['candidate_used'])
        self.assertIsNone(result['post_signals']['mean_entropy'])
        self.assertEqual(result['post_context_texts'], ['Old unhelpful context.'])

    def test_coverage_route_checks_actual_evidence_before_abstaining(self):
        signals = dict(retrieval_similarity=.8, entailment_score=.9, mean_entropy=.2, coverage_score=.8)
        result = self.execute('Paris', signals, route='coverage_failure')
        self.assertEqual(result['healed_answer'], 'Paris')
        self.assertTrue(result['recovery_trace']['candidate_used'])

    def test_verbatim_answer_can_remain_unverified_instead_of_being_discarded(self):
        signals = dict(retrieval_similarity=.8, entailment_score=.01, mean_entropy=.2, coverage_score=.8)
        result = self.execute('Paris', signals)
        self.assertEqual(result['healed_answer'], 'Paris')
        self.assertTrue(result['recovery_trace']['candidate_used'])
        self.assertEqual(result['assessment']['outcome'], 'unverified')
        self.assertFalse(result['verified'])

    def test_weaker_retry_does_not_replace_initial_verified_answer(self):
        signals = dict(retrieval_similarity=.1, entailment_score=.9, mean_entropy=.2, coverage_score=.8)
        result = self.execute('Paris', signals, initial_supported=True)
        self.assertEqual(result['post_context_texts'], ['Old unhelpful context.'])
        self.assertFalse(result['recovery_trace']['candidate_used'])


if __name__ == '__main__':
    unittest.main()
