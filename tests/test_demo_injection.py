"""Controlled fault input, live-only output, and isolation from normal queries."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from src import api_server as api
from src.demo_cases import demo_cases
from test_api_contract import fixture_runtime


class InjectedDemoTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def runtime(self, outcome='supported', route='grounding_failure'):
        pipeline, classifier, matrix, initial, recovered = fixture_runtime(outcome)
        pipeline.run_rag_with_detection = Mock(side_effect=AssertionError('Demo must use the injected state'))
        matrix._audit = Mock(return_value=dict(initial['signals']))
        matrix.heal = Mock(return_value=recovered)
        classifier.predict = Mock(return_value=dict(predicted_label=route, classifier_backend='bert', probabilities={route:1.0}))
        return pipeline, classifier, matrix, recovered

    def test_catalog_contains_only_fixed_corrupted_states_not_reference_answers(self):
        data = self.client.get('/demo/cases').json()
        self.assertEqual(data['mode'], 'injected_demo')
        self.assertEqual(len(data['cases']), 3)
        self.assertEqual(data['cases'][0]['injected_answer'], 'the Pacific Ocean')
        for case in data['cases']:
            self.assertNotIn('label', case)
            self.assertNotIn('final_answer', case)
            self.assertNotIn('reference_answer', case)
            self.assertNotIn('expected_answer', case)

    def test_injected_state_is_audited_and_bert_selects_route_without_label_override(self):
        for route in ['grounding_failure', 'retrieval_failure', 'clean']:
            with self.subTest(route=route):
                pipeline, classifier, matrix, recovered = self.runtime(route=route)
                with patch.object(api, 'get_runtime', return_value=(pipeline,classifier,matrix)):
                    response = self.client.post('/demo/run', json={'case_id':'publication-year'})
                self.assertEqual(response.status_code, 200)
                data = response.json()
                case = demo_cases()[0]
                matrix._audit.assert_called_once_with(case['query'], case['injected_answer'], case['injected_contexts'], token_logits=None)
                classifier.predict.assert_called_once_with(data['pre_signals'], query=case['query'], contexts=case['injected_contexts'], answer=case['injected_answer'])
                self.assertEqual(matrix.heal.call_args.args[2], route)
                self.assertEqual(data['initial_answer'], 'the Pacific Ocean')
                self.assertEqual(data['initial_contexts'], case['injected_contexts'])
                self.assertIsNone(data['token_entropy'])
                self.assertIsNone(data['pre_signals']['mean_entropy'])
                self.assertFalse(data['demo']['benchmark_result'])
                self.assertEqual(data['healed_answer'], recovered['healed_answer'])
                pipeline.run_rag_with_detection.assert_not_called()

    def test_no_successful_final_answer_is_substituted_for_refusal_or_unverified(self):
        for outcome in ['refusal', 'unverified']:
            pipeline, classifier, matrix, recovered = self.runtime(outcome=outcome)
            with patch.object(api, 'get_runtime', return_value=(pipeline,classifier,matrix)):
                data = self.client.post('/demo/run', json={'case_id':'publication-year'}).json()
            self.assertEqual(data['response_outcome'], outcome)
            self.assertEqual(data['healed_answer'], recovered['healed_answer'])
            self.assertFalse(data['grounded_answer'])

    def test_no_client_supplied_question_or_final_answer_is_used(self):
        pipeline, classifier, matrix, recovered = self.runtime()
        with patch.object(api, 'get_runtime', return_value=(pipeline,classifier,matrix)):
            data = self.client.post('/demo/run', json={'case_id':'publication-year', 'query':'Override', 'answer':'Preset correct answer'}).json()
        self.assertEqual(data['query'], demo_cases()[0]['query'])
        self.assertEqual(data['healed_answer'], recovered['healed_answer'])

    def test_bad_case_ids_rejected_and_gpu_readiness_and_lock_are_shared(self):
        self.assertEqual(self.client.post('/demo/run', json={'case_id':'../data'}).status_code, 404)
        self.assertEqual(self.client.post('/demo/run', json={}).status_code, 422)
        with patch.object(api, 'pipeline_instance', None):
            self.assertEqual(self.client.post('/demo/run', json={'case_id':'publication-year'}).status_code, 503)
        api._query_lock.acquire()
        try:
            self.assertEqual(self.client.post('/demo/run', json={'case_id':'publication-year'}).status_code, 429)
        finally:
            api._query_lock.release()

    def test_live_queries_still_generate_their_own_initial_answer(self):
        pipeline, classifier, matrix, initial, recovered = fixture_runtime()
        pipeline.run_rag_with_detection = Mock(return_value=initial)
        with patch.object(api, 'get_runtime', return_value=(pipeline,classifier,matrix)):
            data = self.client.post('/query', json={'query':'Custom question'}).json()
        pipeline.run_rag_with_detection.assert_called_once_with('Custom question')
        self.assertEqual(data['execution_mode'], 'live_query')
        self.assertNotIn('demo', data)


if __name__ == '__main__':
    unittest.main()
