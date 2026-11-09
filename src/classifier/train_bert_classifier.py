"""Fine-tune BERT on injected examples with question-grouped held-out evaluation.

This learns labels of constructed examples. It does not establish natural-error
accuracy. No simulated entropy or coverage values are used as training inputs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

import config
from .bert_data import LABELS, LABEL_MAP, INPUT_SCHEMA, classifier_text, grouped_split


def train(dataset_path, output_dir=None, epochs=3, batch_size=8,
          learning_rate=2e-5, max_length=None, allow_cpu=False, seed=42):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from transformers import AutoTokenizer, BertForSequenceClassification, DataCollatorWithPadding

    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("BERT training needs a Colab GPU, or explicit --allow-cpu for a slower local run.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    output_dir = Path(output_dir or config.BERT_CLASSIFIER_DIR)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty output directory; keeping existing checkpoint at {output_dir}.")
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    split = grouped_split(dataset, seed)
    max_length = max_length or config.BERT_CLASSIFIER_MAX_LENGTH
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_CLASSIFIER_BASE)
    model = BertForSequenceClassification.from_pretrained(
        config.BERT_CLASSIFIER_BASE, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=LABEL_MAP).to(device)
    base_revision = getattr(model.config, "_commit_hash", None)
    collator = DataCollatorWithPadding(tokenizer)
    loaders = {}
    for name, indices in split["indices"].items():
        rows = []
        for i in indices:
            row = dataset[i]
            encoded = tokenizer(classifier_text(row["query"], row["context_chunks"], row["answer"]),
                                truncation=True, max_length=max_length)
            encoded["labels"] = LABEL_MAP[row["label"]]
            rows.append(encoded)
        loaders[name] = DataLoader(rows, batch_size=batch_size, shuffle=name == "train",
                                    collate_fn=collator,
                                    generator=torch.Generator().manual_seed(seed))

    def evaluate(active_model, loader):
        active_model.eval()
        truth, predicted = [], []
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                logits = active_model(**batch).logits
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Non-finite BERT evaluation logits.")
                truth.extend(batch["labels"].cpu().tolist())
                predicted.extend(logits.argmax(-1).cpu().tolist())
        return {"macro_f1": float(f1_score(truth, predicted, labels=list(range(5)),
                                           average="macro", zero_division=0)),
                "classification_report": classification_report(
                    truth, predicted, labels=list(range(5)), target_names=list(LABELS),
                    output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(5))).tolist()}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    history, best_score = [], -1.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            # FP32 training is predictable for this small dataset and avoids
            # introducing mixed-precision scaling changes across torch versions.
            loss = model(**batch).loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite BERT training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
        validation = evaluate(model, loaders["validation"])
        entry = {"epoch": epoch + 1, "mean_training_loss": total_loss / len(loaders["train"]),
                 "validation_macro_f1": validation["macro_f1"]}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if validation["macro_f1"] > best_score:
            best_score = validation["macro_f1"]
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Only validation selects the checkpoint. Test is evaluated once afterwards.
    model = BertForSequenceClassification.from_pretrained(output_dir).to(device)
    test = evaluate(model, loaders["test"])
    manifest = {
        "input_schema": INPUT_SCHEMA, "labels": list(LABELS),
        "base_model": config.BERT_CLASSIFIER_BASE,
        "base_model_revision": base_revision,
        "dataset_sha256": split["dataset_sha256"], "max_length": max_length,
        "seed": seed, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "best_validation_macro_f1": best_score,
        "history": history, "held_out_test": test,
        "torch_version": torch.__version__, "device": str(device),
        "training_inputs": "question, stored answer, constructed context; no labels or simulated signals in inputs",
        "scope": "Synthetic injection classification; not independent natural-error or end-to-end accuracy.",
    }
    manifest["checkpoint_sha256"] = hashlib.sha256((output_dir / "model.safetensors").read_bytes()).hexdigest()
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"held_out_test": test, "split_class_counts": split["class_counts"]}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.FAILURE_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=config.BERT_CLASSIFIER_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    train(args.dataset, args.output, args.epochs, args.batch_size, allow_cpu=args.allow_cpu)
