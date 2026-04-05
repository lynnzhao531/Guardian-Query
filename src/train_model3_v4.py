"""Experiment C — Model 3 v4: all-hard-negative contrastive + MNRL loss.

Changes vs v3:
  • Pairs are ALL hard negatives — no easy (HIGH vs LIKELY_LOW) pairs.
    POSITIVE: two HIGH articles (label ≥ 0.75), any method
    HARD NEG: HIGH + LOW (label < 0.25) from the SAME Guardian section
    Target: 800 positive + 1200 hard-neg → 2000 triplets (anchor, pos, hard_neg)
  • Loss: MultipleNegativesRankingLoss (better for small datasets; uses in-batch
    negatives plus the explicit hard negative).
  • 5 epochs (not 3).

Outputs: models/model3_v4/   (st_finetuned/, centroids.pkl, <dim>.pkl × 7)
Never overwrites models/model3/ or models/model3_v3/.
"""
from __future__ import annotations

import logging
import pickle
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_model3_v4")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODEL_DIR = ROOT / "models" / "model3_v4"
ST_OUT = MODEL_DIR / "st_finetuned"
COMBINED = OUTPUTS / "combined_training_data.csv"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

METHOD_DIMS = [
    "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]
METHOD_KEYS = [d.replace("method_", "") for d in METHOD_DIMS]

GOLD_CSVS = {
    "rct": "rct.csv",
    "prepost": "prepost.csv",
    "case_study": "casestudy.csv",
    "expert_qual": "expert_qual.csv",
    "expert_secondary": "expert_secondary_quant.csv",
    "gut": "gut_decision.csv",
}


def _section(url: str) -> str:
    m = re.search(r"theguardian\.com/([^/]+)/", str(url))
    return m.group(1) if m else "_none_"


def _text(row) -> str:
    return f"{row.get('title','')} {str(row.get('body_excerpt',''))[:600]}".strip()


# ── Step 1: Build triplets ───────────────────────────────────────────────────

def build_triplets(n_pos: int = 800, n_hard: int = 1200):
    from sentence_transformers import InputExample

    df = pd.read_csv(COMBINED, low_memory=False)
    df = df[df["title"].astype(str).str.len() > 10].reset_index(drop=True)
    df["section"] = df["url_canon"].apply(_section)

    high = df[df["label"] >= 0.75].reset_index(drop=True)
    low = df[df["label"] < 0.25].reset_index(drop=True)
    logger.info("HIGH=%d LOW=%d", len(high), len(low))

    # Group HIGH and LOW by section for hard-neg pairing
    low_by_section: dict[str, list[int]] = {}
    for i, r in low.iterrows():
        low_by_section.setdefault(r["section"], []).append(i)

    examples: list = []
    rng = random.Random(SEED)

    # Positive pairs (anchor = HIGH, positive = another HIGH). MNRL only needs
    # (anchor, positive) — in-batch examples serve as negatives.
    # We still emit them as (anchor, positive) pairs (2-tuple).
    for _ in range(n_pos):
        a, b = rng.sample(range(len(high)), 2) if len(high) >= 2 else (0, 0)
        examples.append(InputExample(texts=[_text(high.iloc[a]), _text(high.iloc[b])]))

    # Hard negatives: (anchor=HIGH, positive=HIGH_same_method_if_possible,
    # hard_negative=LOW same section). MNRL accepts 3-tuples (anchor, pos, neg).
    sections_with_low = [s for s, ids in low_by_section.items() if ids]
    high_by_section: dict[str, list[int]] = {}
    for i, r in high.iterrows():
        high_by_section.setdefault(r["section"], []).append(i)

    built = 0
    attempts = 0
    while built < n_hard and attempts < n_hard * 10:
        attempts += 1
        sec = rng.choice(sections_with_low)
        if sec not in high_by_section:
            # anchor from any section, positive = another HIGH any section, hard_neg from sec
            a_idx = rng.randrange(len(high))
            p_idx = rng.randrange(len(high))
        else:
            a_idx = rng.choice(high_by_section[sec])
            p_idx = rng.choice(high_by_section[sec]) if len(high_by_section[sec]) > 1 else rng.randrange(len(high))
        n_idx = rng.choice(low_by_section[sec])
        a_txt = _text(high.iloc[a_idx])
        p_txt = _text(high.iloc[p_idx])
        n_txt = _text(low.iloc[n_idx])
        if a_txt == p_txt:
            continue
        examples.append(InputExample(texts=[a_txt, p_txt, n_txt]))
        built += 1

    rng.shuffle(examples)
    logger.info("Built %d total examples (%d positive pairs, %d hard-neg triplets)",
                len(examples), n_pos, built)
    return examples


# ── Step 2: Fine-tune ────────────────────────────────────────────────────────

def finetune(examples, epochs: int = 5, batch_size: int = 16):
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer("all-MiniLM-L6-v2")
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    warmup = int(len(loader) * epochs * 0.1)
    logger.info("Fine-tune MNRL: %d examples × %d epochs (bs=%d, warmup=%d)",
                len(examples), epochs, batch_size, warmup)

    ST_OUT.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup,
        output_path=str(ST_OUT),
        show_progress_bar=False,
    )
    logger.info("ST saved → %s", ST_OUT)
    return model


# ── Step 3: centroids ────────────────────────────────────────────────────────

def _detect_body(df):
    for c in ["body", "article_body", "body_text", "bodyText"]:
        if c in df.columns:
            return c
    return None


def build_centroids(model):
    centroids = {}
    for key, fname in GOLD_CSVS.items():
        p = DATA / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        body_col = _detect_body(df)
        texts = [
            f"{str(r.get('title',''))} {str(r.get(body_col,''))[:800] if body_col else ''}".strip()
            for _, r in df.iterrows()
        ]
        if not texts:
            continue
        emb = model.encode(texts, show_progress_bar=False, batch_size=64)
        centroids[f"method_{key}"] = emb.mean(axis=0)
        logger.info("Centroid %s from %d articles", key, len(texts))
    return centroids


# ── Step 4: regressors ──────────────────────────────────────────────────────

def _cos_sims(emb: np.ndarray, centroids: dict) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity
    sims = []
    for dim in METHOD_DIMS:
        if dim in centroids:
            sims.append(float(cosine_similarity(
                emb.reshape(1, -1), centroids[dim].reshape(1, -1))[0, 0]))
        else:
            sims.append(0.0)
    return np.array(sims)


def train_regressors(model, centroids):
    from sklearn.neural_network import MLPRegressor

    df = pd.read_csv(COMBINED, low_memory=False)
    df = df[df["title"].astype(str).str.len() > 5].reset_index(drop=True)

    texts = [
        f"{str(r.get('title',''))} {str(r.get('body_excerpt',''))[:800]}"
        for _, r in df.iterrows()
    ]
    logger.info("Embedding %d articles...", len(texts))
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
    features = np.array([_cos_sims(e, centroids) for e in embeddings])

    y_method = {}
    for dim in METHOD_DIMS:
        key = dim.replace("method_", "")
        y = [float(r["label"]) if str(r["method_dimension"]) == key else 0.0
             for _, r in df.iterrows()]
        y_method[dim] = np.array(y)
    y_decision = df["label"].astype(float).values
    base_w = df["sample_weight"].astype(float).values

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for dim in METHOD_DIMS + ["decision"]:
        y = y_decision if dim == "decision" else y_method[dim]
        w = base_w * np.where(y >= 0.75, 3.0, np.where(y >= 0.25, 2.0, 1.0))
        rep = np.clip(np.round(w / w.min() if w.min() > 0 else w).astype(int), 1, 5)
        Xb, yb = [], []
        for xi, yi, ri in zip(features, y, rep):
            for _ in range(int(ri)):
                Xb.append(xi); yb.append(yi)
        Xb, yb = np.array(Xb), np.array(yb)
        reg = MLPRegressor(
            hidden_layer_sizes=(32, 16), max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.15,
        )
        reg.fit(Xb, yb)
        with open(MODEL_DIR / f"{dim}.pkl", "wb") as f:
            pickle.dump(reg, f)
        logger.info("Trained %s on %d rows", dim, len(Xb))

    with open(MODEL_DIR / "centroids.pkl", "wb") as f:
        pickle.dump(centroids, f)
    logger.info("Saved centroids + regressors → %s", MODEL_DIR)


def main():
    logger.info("=== Model 3 v4 training (hard-neg + MNRL) ===")
    examples = build_triplets(n_pos=800, n_hard=1200)
    model = finetune(examples, epochs=5, batch_size=16)
    centroids = build_centroids(model)
    train_regressors(model, centroids)
    logger.info("=== DONE → %s ===", MODEL_DIR)


if __name__ == "__main__":
    main()
