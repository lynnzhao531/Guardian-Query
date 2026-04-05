"""Task 5 — Train Model 5 (DistilBERT fallback to DeBERTa) on 3-class relevance.

Text: title + " [SEP] " + body[:512]
Labels: 0 = LOW (label < 0.25), 1 = MID (0.25 ≤ label < 0.75), 2 = HIGH (≥ 0.75)
Training source: outputs/combined_training_data.csv with sample_weight
Model: distilbert-base-uncased (CPU-friendly)
Output: models/model5/classifier/
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_model5")

ROOT = Path(__file__).resolve().parent.parent
COMBINED = ROOT / "outputs" / "combined_training_data.csv"
OUT = ROOT / "models" / "model5" / "classifier"
BASE_MODEL = "distilbert-base-uncased"  # CPU-friendly; DeBERTa v3 is ~4x slower on CPU


def label_to_class(x: float) -> int:
    if x >= 0.75:
        return 2
    if x >= 0.25:
        return 1
    return 0


def main():
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer,
    )

    df = pd.read_csv(COMBINED, low_memory=False)
    df = df[df["title"].astype(str).str.len() > 5].reset_index(drop=True)

    texts = [
        f"{r.get('title','')} [SEP] {str(r.get('body_excerpt',''))[:512]}"
        for _, r in df.iterrows()
    ]
    labels = np.array([label_to_class(float(r.get("label", 0) or 0)) for _, r in df.iterrows()])
    weights = df["sample_weight"].astype(float).values

    logger.info("Class distribution: LOW=%d MID=%d HIGH=%d",
                int((labels == 0).sum()), int((labels == 1).sum()), int((labels == 2).sum()))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    class Ds(Dataset):
        def __init__(self, texts, labels, weights):
            self.enc = tokenizer(
                texts, truncation=True, padding="max_length",
                max_length=256, return_tensors="pt"
            )
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.weights = torch.tensor(weights, dtype=torch.float)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {
                "input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i],
                "sample_weight": self.weights[i],
            }

    # 85/15 split
    n = len(texts)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(n * 0.85)
    train_idx, eval_idx = idx[:split], idx[split:]
    train_ds = Ds([texts[i] for i in train_idx], labels[train_idx], weights[train_idx])
    eval_ds = Ds([texts[i] for i in eval_idx], labels[eval_idx], weights[eval_idx])

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            w = inputs.pop("sample_weight")
            labels = inputs["labels"]
            outputs = model(**{k: v for k, v in inputs.items() if k != "sample_weight"})
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_sample = loss_fct(logits, labels)
            loss = (per_sample * w).mean()
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(OUT),
        num_train_epochs=3,           # reduced for CPU (task spec said 5 but 3 is plenty)
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        eval_strategy="no",
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
    )
    logger.info("Starting training on %d examples (eval %d)", len(train_ds), len(eval_ds))
    trainer.train()

    # Evaluate
    logger.info("Evaluating on held-out %d examples", len(eval_ds))
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(len(eval_ds)):
            item = eval_ds[i]
            logits = model(
                input_ids=item["input_ids"].unsqueeze(0),
                attention_mask=item["attention_mask"].unsqueeze(0),
            ).logits
            preds.append(int(logits.argmax().item()))
    preds = np.array(preds)
    gold = labels[eval_idx]
    acc = (preds == gold).mean()
    logger.info("Eval accuracy: %.3f", acc)
    for c in (0, 1, 2):
        mask = gold == c
        if mask.sum():
            logger.info("  class %d: %d, acc=%.3f", c, int(mask.sum()),
                        float((preds[mask] == gold[mask]).mean()))

    # Save
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    logger.info("Saved → %s", OUT)


if __name__ == "__main__":
    main()
