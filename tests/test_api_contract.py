"""HTTP integration checks with injected runtime fixtures; no model downloads."""
import json
import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from src import api_server as api


def fixture_runtime(outcome='supported'):
    signals = dict(retrieval_similarity=.8, entailment_score=.9, mean_entropy=.4,
                   coverage_score=.8, is_refusal=False)
    initial = dict(answer='Old answer', context_texts=['Evidence paragraph'], signals=signals,
                   retrieved_chunks=[dict(text='Evidence paragraph', score=.8, rank=0, metadata={'title':'Article'})])
    pipeline = SimpleNamespace(index=SimpleNamespace(ntotal=366742), run_rag_with_detection=lambda query: initial)
    prediction = dict(predicted_label='retrieval_failure', classifier_backend='bert',
        probabilities=dict(clean=.01,retrieval_failure=.96,grounding_failure=.01,confidence_failure=.01,coverage_failure=.01))
    classifier = SimpleNamespace(predict=lambda *args, **kwargs: prediction)
    recovered = dict(healed_answer='Final answer' if outcome != 'refusal' else "I don't have enough information to answer this question.",
        action_taken='Fresh local retrieval', verified=outcome != 'unverified',
        assessment=dict(outcome=outcome,accepted=outcome!='unverified',grounded_answer=outcome=='supported',refusal=outcome=='refusal'),
        post_signals=dict(signals,mean_entropy=None if outcome=='refusal' else .2),
        post_context_texts=['Final evidence'], recovery_trace=dict(version='4.2',route='retrieval_failure',candidate_used=True),
        verification_notes='Abstention issued; no factual-answer verification claimed.' if outcome=='refusal' else 'Signal assessment')
    matrix = SimpleNamespace(heal=lambda *args:recovered)
    return pipeline,classifier,matrix,initial,recovered


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(api.app)

    def test_not_ready_is_not_a_successful_query(self):
        with patch.object(api,'pipeline_instance',None):
            self.assertEqual(self.client.get('/health').status_code,503)
            self.assertEqual(self.client.post('/query',json={'query':'Question'}).status_code,503)

    def test_health_reports_real_readiness_and_index(self):
        pipeline,*_=fixture_runtime()
        with patch.object(api,'pipeline_instance',pipeline):
            response=self.client.get('/health')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['index_vectors'],366742)
        self.assertIsInstance(response.json()['gpu_available'],bool)

    def test_api_carries_real_before_after_probabilities_evidence_and_trace(self):
        pipeline,classifier,matrix,_,_=fixture_runtime()
        with patch.object(api,'get_runtime',return_value=(pipeline,classifier,matrix)):
            response=self.client.post('/query',json={'query':'Question'})
        self.assertEqual(response.status_code,200)
        data=response.json()
        self.assertEqual(data['class_probabilities']['retrieval_failure'],.96)
        self.assertEqual(data['recovery_trace']['version'],'4.2')
        self.assertEqual(data['final_contexts'],['Final evidence'])
        self.assertTrue(data['grounded_answer'])
        self.assertEqual(data['thresholds']['mean_entropy'],2)

    def test_refusal_is_separate_from_grounded_answer(self):
        pipeline,classifier,matrix,_,_=fixture_runtime('refusal')
        with patch.object(api,'get_runtime',return_value=(pipeline,classifier,matrix)):
            data=self.client.post('/query',json={'query':'Question'}).json()
        self.assertEqual(data['response_outcome'],'refusal')
        self.assertFalse(data['grounded_answer'])
        self.assertTrue(data['verified'])
        self.assertIsNone(data['post_signals']['mean_entropy'])

    def test_unverified_is_not_promoted_to_verified(self):
        pipeline,classifier,matrix,_,_=fixture_runtime('unverified')
        with patch.object(api,'get_runtime',return_value=(pipeline,classifier,matrix)):
            data=self.client.post('/query',json={'query':'Question'}).json()
        self.assertEqual(data['response_outcome'],'unverified')
        self.assertFalse(data['verified'])

    def test_json_nonfinite_values_are_null_in_all_nested_fields(self):
        pipeline,classifier,matrix,initial,recovered=fixture_runtime()
        initial['signals']['mean_entropy']=float('nan')
        recovered['recovery_trace']['scores']=[float('inf'),float('-inf')]
        with patch.object(api,'get_runtime',return_value=(pipeline,classifier,matrix)):
            response=self.client.post('/query',json={'query':'Question'})
        data=response.json()
        self.assertIsNone(data['token_entropy'])
        self.assertEqual(data['recovery_trace']['scores'],[None,None])
        json.dumps(data,allow_nan=False)

    def test_empty_and_long_queries_rejected(self):
        for query in ['', '   ', 'x'*2001]:
            self.assertEqual(self.client.post('/query',json={'query':query}).status_code,422)

    def test_gpu_concurrency_guard(self):
        api._query_lock.acquire()
        try:
            self.assertEqual(self.client.post('/query',json={'query':'Question'}).status_code,429)
        finally:
            api._query_lock.release()

    def test_failure_releases_query_lock(self):
        with patch.object(api,'get_runtime',side_effect=RuntimeError('fixture failure')):
            self.assertEqual(self.client.post('/query',json={'query':'Question'}).status_code,500)
        self.assertTrue(api._query_lock.acquire(blocking=False))
        api._query_lock.release()

    def test_notebook_injection_does_not_initialize_models_again(self):
        pipeline,classifier,*_=fixture_runtime()
        with patch.object(api,'pipeline_instance',pipeline),patch.object(api,'initialize_runtime') as initialize:
            with TestClient(api.app) as client:
                self.assertEqual(client.get('/health').status_code,200)
            initialize.assert_not_called()

    def test_dashboard_and_assets_are_served_without_node(self):
        self.assertEqual(self.client.get('/').status_code,200)
        for path in ['/static/app.js','/static/benchmark-data.js','/static/vendor/lucide.min.js']:
            self.assertEqual(self.client.get(path).status_code,200,path)

    def test_benchmark_is_recorded_and_separate_from_live_inference(self):
        data=self.client.get('/benchmark').json()
        self.assertEqual(data['regression_rows'],138)
        self.assertEqual(data['answered_accuracy']['correct'],60)
        self.assertEqual(len(data['examples']),5)
        self.assertEqual([r['name'] for r in data['comparison']],['Self-RAG adaptation','Our final system'])
        self.assertFalse(self.client.get('/metadata').json()['external_web_search'])


if __name__=='__main__':
    unittest.main()
