"""Model 5 — DistilBERT/DeBERTa 3-class relevance scorer.

Loads a fine-tuned transformer classifier that predicts LOW/MID/HIGH
article relevance from title + body snippet. Maps the 3-class output
to the pipeline's 7-vector format.

Mapping rule (per Task 5 spec):
  All 6 method dimensions + decision get the same score = p_high.
  Model 5 detects RELEVANCE, not method type, so we don't differentiate.
  Consensus pairs M5's relevance signal with M1/M3's method signals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model5" / "classifier"
MAX_LEN = 256

DIMENSIONS = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is not None:
        return
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    _model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    _model.eval()


def _p_high(title: str, body: str) -> float:
    import torch
    _load()
    text = f"{title or ''} [SEP] {(body or '')[:512]}"
    enc = _tokenizer(text, truncation=True, padding="max_length",
                     max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    # probs: [LOW, MID, HIGH]
    return float(probs[2])


# Layer 2: continuous pass-through — p_HIGH softmax output is the p1.
# Rollback: restore commented block.
# def _discrete(p: float) -> dict:
#     p = max(0.0, min(1.0, float(p)))
#     if p < 0.25:
#         return {"score": 0.0, "p0": 0.80, "p05": 0.15, "p1": 0.05}
#     if p < 0.75:
#         return {"score": 0.5, "p0": 0.20, "p05": 0.60, "p1": 0.20}
#     return     {"score": 1.0, "p0": 0.05, "p05": 0.15, "p1": 0.80}

def _discrete(p: float) -> dict:
    p = max(0.0, min(1.0, float(p)))
    remaining = 1.0 - p
    if p >= 0.5:
        p05 = remaining * 0.7
        p0 = remaining * 0.3
    else:
        p0 = remaining * 0.7
        p05 = remaining * 0.3
    return {"score": p, "p0": p0, "p05": p05, "p1": p}


def score_article(title: str, body: str) -> dict:
    p_high = _p_high(title, body)
    disc = _discrete(p_high)
    return {dim: dict(disc) for dim in DIMENSIONS}


if __name__ == "__main__":
    import json
    print(json.dumps(score_article(
        "Stop and breathe: police staff offered meditation lessons",
        "Meditation lessons will be made available to all 200,000 police staff "
        "after a trial across five forces found the practice improved wellbeing."
    ), indent=2))
