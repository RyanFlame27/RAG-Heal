"""Actual BERT forward/backward and checkpoint round-trip, without downloads."""

import importlib.util
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


@unittest.skipUnless(AVAILABLE, "PyTorch and Transformers are required for the neural integration test")
class BertModelTests(unittest.TestCase):
    def test_neural_training_save_load_and_probabilities(self):
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
        from src.classifier.bert_data import LABELS, LABEL_MAP, INPUT_SCHEMA
        from src.classifier.predict import FailureClassifier
        import config
        from unittest.mock import patch
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                          "question", "answer", "context", ":", "paris", "france", "capital", "."]
            vocab = root / "vocab.txt"
            vocab.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(vocab))
            model = BertForSequenceClassification(BertConfig(
                vocab_size=len(vocabulary), hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32, num_labels=5,
                id2label=dict(enumerate(LABELS)), label2id=LABEL_MAP))
            inputs = tokenizer("question: capital france answer: paris context: paris .", return_tensors="pt")
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            initial_weight = model.classifier.weight.detach().clone()
            model.train()
            loss = model(**inputs, labels=torch.tensor([0])).loss
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(initial_weight, model.classifier.weight.detach()))
            model.save_pretrained(root, safe_serialization=True)
            tokenizer.save_pretrained(root)
            manifest = {"input_schema": INPUT_SCHEMA, "labels": list(LABELS),
                        "max_length": 32, "dataset_sha256": "toy-neural-integration-test",
                        "checkpoint_sha256": hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest()}
            (root / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(config, "BERT_CLASSIFIER_DEVICE", "cpu"):
                router = FailureClassifier(model_path=root, backend="bert")
                prediction = router.predict({}, query="capital france", contexts=["paris"], answer="paris")
            self.assertEqual(prediction["classifier_backend"], "bert")
            self.assertEqual(set(prediction["probabilities"]), set(LABELS))
            self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0, places=5)
            self.assertIn(prediction["predicted_label"], LABELS)


if __name__ == "__main__":
    unittest.main()
