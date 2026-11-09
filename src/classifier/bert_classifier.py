"""Five-class BERT failure routing from query, context and answer text."""

import json
import hashlib
from pathlib import Path

import config
from .bert_data import LABELS, INPUT_SCHEMA, classifier_text


class BertFailureClassifier:
    def __init__(self, model_path=None):
        self.path = Path(model_path or config.BERT_CLASSIFIER_DIR)
        manifest_path = self.path / "training_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No trained BERT failure classifier at {self.path}. "
                "Run python -m src.classifier.train_bert_classifier --dataset <dataset.json> "
                "before evaluation. Rules are available only through backend='rules'.")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("input_schema") != INPUT_SCHEMA:
            raise ValueError("BERT checkpoint uses an incompatible input schema.")
        if self.manifest.get("labels") != list(LABELS):
            raise ValueError("BERT checkpoint label order differs from the recovery routes.")
        weights_path = self.path / "model.safetensors"
        if not weights_path.is_file() or hashlib.sha256(weights_path.read_bytes()).hexdigest() != self.manifest.get("checkpoint_sha256"):
            raise ValueError("BERT weights do not match the completed training manifest.")
        import torch
        from transformers import BertForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = BertForSequenceClassification.from_pretrained(
            self.path, local_files_only=True).to(config.BERT_CLASSIFIER_DEVICE).eval()
        expected = {i: label for i, label in enumerate(LABELS)}
        actual = {int(i): label for i, label in self.model.config.id2label.items()}
        if actual != expected:
            raise ValueError("Saved BERT id2label mapping is incompatible.")
        self.description = {"backend": "bert", "checkpoint": str(self.path),
                            "training_dataset_sha256": self.manifest["dataset_sha256"],
                            "checkpoint_sha256": self.manifest["checkpoint_sha256"],
                            "input_schema": INPUT_SCHEMA}

    def predict(self, signals, *, query=None, contexts=None, answer=None):
        text = classifier_text(query, contexts, answer)
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=self.manifest["max_length"])
        encoded = encoded.to(next(self.model.parameters()).device)
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits[0].float()
            if not bool(self.torch.isfinite(logits).all()):
                raise ValueError("BERT returned non-finite logits.")
            probabilities = self.torch.softmax(logits, dim=-1).cpu().tolist()
        class_id = max(range(len(probabilities)), key=probabilities.__getitem__)
        label = LABELS[class_id]
        return {"predicted_label": label, "class_id": class_id,
                "probabilities": dict(zip(LABELS, probabilities)),
                "is_failure": label != "clean", "classifier_backend": "bert"}
