"""Model 3 v3 scorer — loads fine-tuned ST encoder + MLP regressors.

Reads from models/model3_v3/ (does NOT touch the running pipeline's models/model3/).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "model3_v3"
ST_DIR = MODEL_DIR / "st_finetuned"
MAX_BODY_CHARS = 800

DIMENSIONS = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]

_st_model = None
_centroids: Dict[str, np.ndarray] = {}
_regressors: Dict = {}


def _get_st():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(str(ST_DIR))
    return _st_model


def load_models():
    global _centroids, _regressors
    with open(MODEL_DIR / "centroids.pkl", "rb") as f:
        _centroids = pickle.load(f)
    _regressors = {}
    for dim in DIMENSIONS:
        p = MODEL_DIR / f"{dim}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                _regressors[dim] = pickle.load(f)


def _cos_sims(emb: np.ndarray) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity
    sims = []
    for dim in DIMENSIONS:
        if dim == "decision":
            continue
        if dim in _centroids:
            sims.append(float(cosine_similarity(
                emb.reshape(1, -1), _centroids[dim].reshape(1, -1))[0, 0]))
        else:
            sims.append(0.0)
    return np.array(sims)


def _discrete(pred: float) -> dict:
    p = float(np.clip(pred, 0.0, 1.0))
    if p < 0.25:
        return {"score": 0.0, "p0": 0.80, "p05": 0.15, "p1": 0.05}
    if p < 0.75:
        return {"score": 0.5, "p0": 0.20, "p05": 0.60, "p1": 0.20}
    return {"score": 1.0, "p0": 0.05, "p05": 0.15, "p1": 0.80}


def score_article(title: str, body_text: str) -> dict:
    if not _regressors:
        load_models()
    text = f"{title or ''} {(body_text or '')[:MAX_BODY_CHARS]}".strip()
    emb = _get_st().encode(text, show_progress_bar=False)
    features = _cos_sims(emb).reshape(1, -1)
    result = {}
    for dim in DIMENSIONS:
        if dim in _regressors:
            pred = float(np.clip(_regressors[dim].predict(features)[0], 0.0, 1.0))
        else:
            pred = 0.0
        result[dim] = _discrete(pred)
    return result


if __name__ == "__main__":
    print(score_article(
        "Stop and breathe: police staff offered meditation lessons",
        "Meditation lessons will be made available to all 200,000 police staff "
        "in England and Wales after a trial across five forces found the practice "
        "improved wellbeing and work performance."))
