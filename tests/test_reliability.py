"""Regression checks for invalid entropy, abstention, and verification gates."""

import math
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import config
from src.generation.llm import LLMWrapper
from src.detection.token_entropy import detect_confidence_failure, is_confidence_failure
from src.detection.response_check import assess_response, is_refusal
from src.classifier.predict import FailureClassifier
from src.recovery.recovery_matrix import RecoveryMatrix


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.wrapper = LLMWrapper.__new__(LLMWrapper)
        self.wrapper.tokenizer = SimpleNamespace(decode=lambda ids: str(ids[0]))
        self.signals = dict(retrieval_similarity=0.8, entailment_score=0.9,
                            mean_entropy=0.4, coverage_score=0.8)

    def test_masked_and_uniform_entropy(self):
        result = self.wrapper._extract_token_logits((torch.tensor([[0., -float('inf')]]),))
        self.assertEqual(result[0]['entropy'], 0.0)
        uniform = self.wrapper._extract_token_logits((torch.zeros(1, 2),))
        self.assertAlmostEqual(uniform[0]['entropy'], math.log(2), places=6)

    def test_invalid_entropy_remains_unavailable(self):
        invalid = self.wrapper._extract_token_logits((torch.full((1, 2), float('nan')),))
        for tokens in [None, [], invalid, [{'entropy': 0.5}, {'entropy': float('nan')}]]:
            self.assertIsNone(detect_confidence_failure(tokens))
        self.assertTrue(is_confidence_failure(None))
        self.assertTrue(is_confidence_failure(config.ENTROPY_THRESHOLD))

    def test_missing_uncertainty_does_not_pass_as_clean(self):
        signals = dict(self.signals, mean_entropy=None)
        self.assertEqual(FailureClassifier(backend='rules').predict(signals)['predicted_label'], 'confidence_failure')
        self.assertFalse(assess_response('Paris is the capital.', signals)['accepted'])

    def test_refusal_is_not_grounding(self):
        result = assess_response("I don't have enough information to answer this question.", self.signals)
        self.assertTrue(result['refusal'])
        self.assertTrue(result['accepted'])
        self.assertFalse(result['grounded_answer'])
        self.assertFalse(assess_response("I cannot answer this question.", self.signals,
                                         allow_refusal=False)['accepted'])

    def test_refusal_prefix_does_not_hide_an_assertion(self):
        self.assertFalse(is_refusal("I don't have enough information. Rome fell in 2019."))
        self.assertFalse(is_refusal("I cannot answer, but the Moon is made of cheese."))

    def test_all_four_classifier_routes(self):
        examples = {
            'coverage_failure': dict(self.signals, coverage_score=0.1),
            'retrieval_failure': dict(self.signals, retrieval_similarity=0.1),
            'grounding_failure': dict(self.signals, entailment_score=0.1),
            'confidence_failure': dict(self.signals, mean_entropy=3.0),
        }
        for expected, signals in examples.items():
            self.assertEqual(FailureClassifier(backend='rules').predict(signals)['predicted_label'], expected)

    def test_recovery_requires_all_checks(self):
        matrix = RecoveryMatrix(SimpleNamespace())
        initial = {'answer': 'An unsupported claim.', 'context_texts': ['Initial context'],
                   'signals': self.signals}
        output = {'new_answer': 'A regenerated claim.', 'expanded_context_texts': ['New context'],
                  'token_logits': [{'entropy': 0.1}], 'action_taken': 'Expansion'}
        with patch('src.recovery.recovery_matrix.recover_confidence_failure', return_value=output):
            with patch.object(matrix, '_audit', return_value=dict(self.signals, entailment_score=0.01)):
                result = matrix.heal('Question', initial, 'confidence_failure')
                self.assertFalse(result['verified'])
            with patch.object(matrix, '_audit', return_value=self.signals):
                self.assertTrue(matrix.heal('Question', initial, 'confidence_failure')['verified'])

    def test_api_reports_refusal_and_missing_measurement(self):
        from src import api_server
        initial = {'answer': 'An unsupported claim.', 'context_texts': ['Context'],
                   'signals': dict(self.signals, coverage_score=0.1, mean_entropy=None)}
        stub = SimpleNamespace(run_rag_with_detection=lambda query: initial)
        with patch.object(api_server, 'pipeline_instance', stub), patch(
                'src.classifier.predict.load_classifier', return_value=FailureClassifier(backend='rules')):
            result = api_server.process_query_endpoint(api_server.QueryRequest(query='Question'))
        self.assertIsNone(result['token_entropy'])
        self.assertEqual(result['response_outcome'], 'refusal')
        self.assertIn('Abstention', result['verification_status'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
