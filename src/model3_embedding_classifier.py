from __future__ import annotations

"""Model 3: Embedding Classifier — cosine-similarity-only pipeline.

Engine: sentence-transformers all-MiniLM-L6-v2 (local, free).
Features: 6 cosine similarities to method-specific prototype centroids.
NO K* hypothesis features — keeps M3 independent from M4.
7 MLP regressors (one per 7-vector dimension).
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPRegressor

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "model3"
BODY_COLS = ["body", "article_body", "body_text", "bodyText"]
MAX_BODY_CHARS = 800

DIMENSIONS = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]

# Gold files used to build one centroid each (all-high exemplars).
GOLD_FILES = {
    "method_rct": "rct.csv",
    "method_prepost": "prepost.csv",
    "method_case_study": "casestudy.csv",
    "method_expert_qual": "expert_qual.csv",
    "method_expert_secondary": "expert_secondary_quant.csv",
    "method_gut": "gut_decision.csv",
}

# Regressor training data: dimension -> (csv_files, label_source)
REGRESSOR_DEFS = {
    "method_rct":              (["rct 2.csv", "rct.csv"], "method"),
    "method_prepost":          (["prepost 2.csv", "prepost.csv"], "method"),
    "method_case_study":       (["case studies.csv", "casestudy.csv"], "method"),
    "method_expert_secondary": (["quantitative.csv", "expert_secondary_quant.csv"], "method"),
    "method_expert_qual":      (["expert_qual.csv"], "gold"),
    "method_gut":              (["gut.csv", "gut_decision.csv"], "method"),
    "decision":                ("ALL", "decision"),
}

# Module-level caches
_st_model = None
_centroids: Dict[str, np.ndarray] = {}
_regressors: Dict[str, MLPRegressor] = {}


# ── Embedding helpers ───────────────────────────────────────────────

def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def _detect_body(df: pd.DataFrame) -> Optional[str]:
    for c in BODY_COLS:
        if c in df.columns:
            return c
    return None


def _article_text(title: str, body: str) -> str:
    return f"{title or ''} {(body or '')[:MAX_BODY_CHARS]}".strip()


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string -> 1-D numpy array."""
    return _get_st_model().encode(text, show_progress_bar=False)


def _embed_batch(texts: List[str]) -> np.ndarray:
    return _get_st_model().encode(texts, show_progress_bar=False, batch_size=64)


def _cos_sims(emb: np.ndarray) -> np.ndarray:
    """Return 6 cosine similarities from one embedding to all 6 centroids."""
    emb_2d = emb.reshape(1, -1)
    sims = []
    for method in GOLD_FILES:
        if method in _centroids:
            sims.append(float(
                cosine_similarity(emb_2d, _centroids[method].reshape(1, -1))[0, 0]
            ))
        else:
            sims.append(0.0)
    return np.array(sims, dtype=np.float64)


# ── Centroids ───────────────────────────────────────────────────────

def build_centroids() -> Dict[str, np.ndarray]:
    """Build 6 method-specific prototype centroids from gold files."""
    centroids: Dict[str, np.ndarray] = {}
    for method, fname in GOLD_FILES.items():
        path = DATA_DIR / fname
        if not path.exists():
            logger.warning("Gold file not found: %s", path)
            continue
        df = pd.read_csv(path)
        body_col = _detect_body(df)
        texts = [
            _article_text(str(r.get("title", "")),
                          str(r.get(body_col, "")) if body_col else "")
            for _, r in df.iterrows()
        ]
        if not texts:
            logger.warning("No articles in %s", fname)
            continue
        centroids[method] = _embed_batch(texts).mean(axis=0)
        logger.info("Centroid %s: %d articles from %s", method, len(texts), fname)
    return centroids


def compute_features(title: str, body_text: str) -> np.ndarray:
    """Return feature vector: 6 cosine similarities (no K* features)."""
    if not _centroids:
        load_models()
    return _cos_sims(embed_text(_article_text(title, body_text)))


# ── Continuous mapping (Layer 2: was _continuous_to_discrete) ──────
# The old discretization snapped predictions to {0.05, 0.20, 0.80} p1 values,
# which destroyed information and coupled every downstream decision to the
# 0.25/0.75 knees. The continuous version below passes the raw prediction
# through as p1 and distributes the rest over p0/p05 smoothly.
#
# Rollback: uncomment the _continuous_to_discrete body below and remove
# the new implementation.
#
# def _continuous_to_discrete(pred: float) -> dict:
#     pred = float(np.clip(pred, 0.0, 1.0))
#     if pred < 0.25:
#         return {"score": 0.0, "p0": 0.8,  "p05": 0.15, "p1": 0.05}
#     if pred < 0.75:
#         return {"score": 0.5, "p0": 0.2,  "p05": 0.6,  "p1": 0.2}
#     return     {"score": 1.0, "p0": 0.05, "p05": 0.15, "p1": 0.8}

def _continuous_to_discrete(pred: float) -> dict:
    """Continuous pass-through: the raw prediction IS the probability of HIGH."""
    pred = float(np.clip(pred, 0.0, 1.0))
    p1 = pred
    # Remaining mass split symmetrically around 0.5
    remaining = 1.0 - p1
    if pred >= 0.5:
        p05 = remaining * 0.7
        p0 = remaining * 0.3
    else:
        p0 = remaining * 0.7
        p05 = remaining * 0.3
    return {"score": pred, "p0": p0, "p05": p05, "p1": p1}


# ── Training data ───────────────────────────────────────────────────

def _load_csv_features(filename: str) -> List[dict]:
    """Load CSV, embed articles, return list of {features, method, decision}."""
    path = DATA_DIR / filename
    if not path.exists():
        logger.warning("File not found: %s", path)
        return []
    df = pd.read_csv(path)
    body_col = _detect_body(df)
    has_m = "Method" in df.columns
    has_d = "Decision" in df.columns
    texts: List[str] = []
    metas: List[dict] = []
    for _, r in df.iterrows():
        title = str(r.get("title", ""))
        body = str(r.get(body_col, "")) if body_col else ""
        texts.append(_article_text(title, body))
        try:
            mv = min(max(round(float(r["Method"]) * 2) / 2, 0), 1) \
                if has_m and pd.notna(r.get("Method")) else 1.0
        except (ValueError, TypeError):
            mv = 1.0
        try:
            dv = min(max(round(float(r["Decision"]) * 2) / 2, 0), 1) \
                if has_d and pd.notna(r.get("Decision")) else 1.0
        except (ValueError, TypeError):
            dv = 1.0
        metas.append({"method": mv, "decision": dv})
    if not texts:
        return []
    embeddings = _embed_batch(texts)
    for i, meta in enumerate(metas):
        meta["features"] = _cos_sims(embeddings[i])
    return metas


def _gather_training_data(csv_files, label_source: str):
    """Build X (n x 6), y (n,), sample_weights (n,) arrays for one regressor."""
    if csv_files == "ALL":
        csv_files = [f.name for f in sorted(DATA_DIR.glob("*.csv"))
                     if f.name != "Training_cases.csv"]
    rows: List[dict] = []
    for fname in csv_files:
        rows.extend(_load_csv_features(fname))
    if not rows:
        return np.zeros((0, 6)), np.zeros(0), np.zeros(0)

    # For gold-only files, add negative examples from other CSVs
    if label_source == "gold":
        for r in rows:
            r["_label"] = 1.0
        neg_rows = []
        for f in sorted(DATA_DIR.glob("*.csv")):
            if f.name in csv_files or f.name == "Training_cases.csv":
                continue
            neg_rows.extend(_load_csv_features(f.name))
        if neg_rows:
            import random
            random.seed(42)
            n_neg = min(len(neg_rows), len(rows) * 3)
            sampled = random.sample(neg_rows, n_neg)
            for r in sampled:
                r["_label"] = 0.0
            rows.extend(sampled)

    X = np.array([r["features"] for r in rows])
    if label_source == "gold":
        y = np.array([r["_label"] for r in rows])
    elif label_source == "decision":
        y = np.array([r["decision"] for r in rows])
    else:
        y = np.array([r["method"] for r in rows])

    # Class weights: 3x HIGH (y>=0.75), 2x MID (0.25<=y<0.75), 1x LOW (y<0.25)
    weights = np.ones(len(y))
    weights[y >= 0.75] = 3.0
    weights[(y >= 0.25) & (y < 0.75)] = 2.0
    return X, y, weights


# ── Train / Save / Load ────────────────────────────────────────────

def train(articles_df: pd.DataFrame | None = None) -> None:
    """Train centroids + 7 MLP regressors on 6 cosine-sim features only."""
    global _centroids
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _centroids.update(build_centroids())
    with open(MODEL_DIR / "centroids.pkl", "wb") as f:
        pickle.dump(_centroids, f)
    for dim, (files, label_src) in REGRESSOR_DEFS.items():
        X, y, sample_weights = _gather_training_data(files, label_src)
        if len(X) < 2:
            logger.warning("Insufficient data for %s -- skipping", dim)
            continue
        reg = MLPRegressor(
            hidden_layer_sizes=(32, 16), max_iter=500,
            random_state=42, early_stopping=True, validation_fraction=0.15,
        )
        # Duplicate high-weight samples to simulate sample_weight
        # (MLPRegressor doesn't natively support sample_weight in all sklearn versions)
        X_weighted, y_weighted = [], []
        for xi, yi, wi in zip(X, y, sample_weights):
            repeat = int(wi)
            for _ in range(repeat):
                X_weighted.append(xi)
                y_weighted.append(yi)
        X_w = np.array(X_weighted)
        y_w = np.array(y_weighted)
        reg.fit(X_w, y_w)
        _regressors[dim] = reg
        with open(MODEL_DIR / f"{dim}.pkl", "wb") as f:
            pickle.dump(reg, f)
        logger.info("Trained %s on %d samples (6 cosine-sim features)", dim, len(X))
    print(f"Model 3 training complete. {len(_centroids)} centroids, "
          f"{len(_regressors)} regressors saved to {MODEL_DIR}")


def save_models() -> None:
    """Persist current centroids and regressors to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "centroids.pkl", "wb") as f:
        pickle.dump(_centroids, f)
    for dim, reg in _regressors.items():
        with open(MODEL_DIR / f"{dim}.pkl", "wb") as f:
            pickle.dump(reg, f)


def load_models() -> None:
    """Load centroids and regressors from disk."""
    global _centroids
    cent_path = MODEL_DIR / "centroids.pkl"
    if cent_path.exists():
        with open(cent_path, "rb") as f:
            _centroids.update(pickle.load(f))
    else:
        logger.warning("No centroids file at %s", cent_path)
    _regressors.clear()
    for dim in DIMENSIONS:
        path = MODEL_DIR / f"{dim}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                _regressors[dim] = pickle.load(f)
    logger.info("Model 3: loaded %d centroids, %d regressors",
                len(_centroids), len(_regressors))


# ── Scoring ─────────────────────────────────────────────────────────

def score_article(title: str, body_text: str) -> dict:
    """Score article on 7-vector using only cosine-sim features.

    Returns nested dict: {dim: {"score": X, "p0": Y, "p05": Z, "p1": W}, ...}
    """
    if not _regressors:
        load_models()
    features = compute_features(title, body_text).reshape(1, -1)
    result: dict = {}
    for dim in DIMENSIONS:
        if dim in _regressors:
            pred = float(np.clip(_regressors[dim].predict(features)[0], 0.0, 1.0))
        else:
            pred = 0.0
        result[dim] = _continuous_to_discrete(pred)
    return result


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        print("Usage: python src/model3_embedding_classifier.py train")
        print("Then use score_article(title, body) from other modules.")
