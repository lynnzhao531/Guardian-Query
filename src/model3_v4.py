"""Model 3 v4 scorer — loads models/model3_v4/ artifacts.

Same public API as model3_embedding_classifier: score_article(title, body) → 7-vector.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "model3_v4"
ST_DIR = MODEL_DIR / "st_finetuned"

DIMENSIONS = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]
METHOD_DIMS = [d for d in DIMENSIONS if d != "decision"]

_st = None
_centroids = None
_regs: dict = {}


def _load():
    global _st, _centroids
    if _st is not None:
        return
    from sentence_transformers import SentenceTransformer
    _st = SentenceTransformer(str(ST_DIR))
    with open(MODEL_DIR / "centroids.pkl", "rb") as f:
        _centroids = pickle.load(f)
    for dim in DIMENSIONS:
        p = MODEL_DIR / f"{dim}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                _regs[dim] = pickle.load(f)


def _cos_sims(emb: np.ndarray) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity
    sims = []
    for dim in METHOD_DIMS:
        if dim in _centroids:
            sims.append(float(cosine_similarity(
                emb.reshape(1, -1), _centroids[dim].reshape(1, -1))[0, 0]))
        else:
            sims.append(0.0)
    return np.array(sims).reshape(1, -1)


def _probs(score: float) -> dict:
    s = max(0.0, min(1.0, float(score)))
    if s >= 0.75:
        return {"score": 1.0, "p0": 0.05, "p05": 0.15, "p1": 0.80}
    if s >= 0.25:
        return {"score": 0.5, "p0": 0.20, "p05": 0.60, "p1": 0.20}
    return {"score": 0.0, "p0": 0.80, "p05": 0.15, "p1": 0.05}


def score_article(title: str, body: str) -> dict:
    _load()
    text = f"{title or ''} {(body or '')[:800]}"
    emb = _st.encode([text], show_progress_bar=False)[0]
    feats = _cos_sims(emb)
    out = {}
    for dim in DIMENSIONS:
        reg = _regs.get(dim)
        val = float(reg.predict(feats)[0]) if reg is not None else 0.0
        out[dim] = _probs(val)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(score_article(
        "Stop and breathe: police staff offered meditation lessons",
        "Meditation lessons will be made available to all 200,000 police staff after a trial across five forces found the practice improved wellbeing.",
    ), indent=2))
