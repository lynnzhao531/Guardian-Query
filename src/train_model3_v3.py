"""Task 3 — Train Model 3 v3 with contrastive fine-tuning.

Pipeline:
  1. Build ~2000 contrastive pairs from combined_training_data.csv
       positive (same method, both HIGH) → target cos sim 0.8
       hard negative (different method, both HIGH) → target 0.1
       easy negative (HIGH vs LIKELY_LOW) → target 0.0
  2. Fine-tune sentence-transformers/all-MiniLM-L6-v2 3 epochs
       → models/model3_v3/st_finetuned/
  3. Recompute 6 method centroids with fine-tuned encoder
  4. Train 7 MLP regressors (6 cos-sim features → score) with
     sample_weight from combined_training_data.csv + class weights.
     → models/model3_v3/<dim>.pkl + centroids.pkl

Keep-or-discard gate (§Task 3):
  - Accuracy on Training_cases.csv ≥ old model3 accuracy
  - M3-M4 agreement < 80% (check via error correlation matrix in Task 7)
This script TRAINS and SAVES only; evaluation happens in Task 7.
"""
from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_model3_v3")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODEL_DIR = ROOT / "models" / "model3_v3"
ST_OUT = MODEL_DIR / "st_finetuned"
COMBINED = OUTPUTS / "combined_training_data.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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


# ── Step 1: Build contrastive pairs ──────────────────────────────────────────

def _collect_text_bank() -> dict[str, list[tuple[str, str, float]]]:
    """Return {method_key: [(title, body, label_weight), ...]}."""
    df = pd.read_csv(COMBINED, low_memory=False)
    bank: dict[str, list[tuple[str, str, float]]] = {k: [] for k in METHOD_KEYS}
    low_bank: list[tuple[str, str]] = []
    df = df[df["title"].astype(str).str.len() > 10]
    for _, r in df.iterrows():
        m = str(r["method_dimension"])
        text = f"{r.get('title','')} {str(r.get('body_excerpt',''))[:600]}"
        label = float(r.get("label", 0) or 0)
        if m in bank and label >= 0.5:
            bank[m].append((str(r.get("title", "")), text, label))
        elif label < 0.25:
            low_bank.append((str(r.get("title", "")), text))
    bank["__low__"] = low_bank  # type: ignore
    for k, items in bank.items():
        logger.info("bank[%s]: %d", k, len(items))
    return bank  # type: ignore


def build_pairs(bank, n_pairs_per_method: int = 350):
    from sentence_transformers import InputExample
    examples = []
    low = bank["__low__"]
    methods = [k for k in METHOD_KEYS if len(bank[k]) >= 2]

    for m in methods:
        items = bank[m]
        other_methods = [o for o in methods if o != m]
        # Positives (same method HIGH vs HIGH): 60%
        n_pos = int(n_pairs_per_method * 0.5)
        n_hard = int(n_pairs_per_method * 0.25)
        n_easy = n_pairs_per_method - n_pos - n_hard
        for _ in range(n_pos):
            a, b = random.sample(items, 2) if len(items) >= 2 else (items[0], items[0])
            examples.append(InputExample(texts=[a[1], b[1]], label=0.8))
        for _ in range(n_hard):
            a = random.choice(items)
            om = random.choice(other_methods)
            b = random.choice(bank[om])
            examples.append(InputExample(texts=[a[1], b[1]], label=0.1))
        for _ in range(n_easy):
            a = random.choice(items)
            if not low:
                break
            b = random.choice(low)
            examples.append(InputExample(texts=[a[1], b[1]], label=0.0))
    random.shuffle(examples)
    logger.info("Built %d contrastive pairs", len(examples))
    return examples


# ── Step 2: Fine-tune sentence-transformer ──────────────────────────────────

def finetune_st(examples, epochs: int = 3, batch_size: int = 16):
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer("all-MiniLM-L6-v2")
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.CosineSimilarityLoss(model=model)

    warmup = int(len(loader) * epochs * 0.1)
    logger.info("Fine-tuning: %d pairs, %d epochs, batch=%d, warmup=%d",
                len(examples), epochs, batch_size, warmup)

    ST_OUT.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup,
        output_path=str(ST_OUT),
        show_progress_bar=False,
    )
    logger.info("Fine-tuned ST saved → %s", ST_OUT)
    return model


# ── Step 3: Build centroids using fine-tuned encoder ────────────────────────

def _detect_body(df):
    for c in ["body", "article_body", "body_text", "bodyText"]:
        if c in df.columns:
            return c
    return None


def build_centroids(model):
    centroids = {}
    for method_key, fname in GOLD_CSVS.items():
        path = DATA / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        body_col = _detect_body(df)
        texts = [
            f"{str(r.get('title',''))} {str(r.get(body_col,''))[:800] if body_col else ''}".strip()
            for _, r in df.iterrows()
        ]
        if not texts:
            continue
        emb = model.encode(texts, show_progress_bar=False, batch_size=64)
        centroids[f"method_{method_key}"] = emb.mean(axis=0)
        logger.info("Centroid %s built from %d articles", method_key, len(texts))
    return centroids


# ── Step 4: Train MLP regressors ─────────────────────────────────────────────

def _cos_sims(emb: np.ndarray, centroids: dict) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity
    sims = []
    for dim in METHOD_DIMS:
        if dim in centroids:
            sims.append(float(cosine_similarity(emb.reshape(1, -1),
                                               centroids[dim].reshape(1, -1))[0, 0]))
        else:
            sims.append(0.0)
    return np.array(sims)


def train_regressors(model, centroids):
    from sklearn.neural_network import MLPRegressor

    df = pd.read_csv(COMBINED, low_memory=False)
    df = df[df["title"].astype(str).str.len() > 5]

    texts = [
        f"{str(r.get('title',''))} {str(r.get('body_excerpt',''))[:800]}"
        for _, r in df.iterrows()
    ]
    logger.info("Embedding %d training articles...", len(texts))
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
    features = np.array([_cos_sims(e, centroids) for e in embeddings])

    # Build per-dimension labels from method_dimension + label column
    y_method = {}
    for dim in METHOD_DIMS:
        key = dim.replace("method_", "")
        y = []
        for _, r in df.iterrows():
            if str(r["method_dimension"]) == key:
                y.append(float(r["label"]))
            else:
                y.append(0.0)
        y_method[dim] = np.array(y)
    y_decision = df["label"].astype(float).values

    base_weights = df["sample_weight"].astype(float).values

    regressors = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for dim in METHOD_DIMS + ["decision"]:
        y = y_decision if dim == "decision" else y_method[dim]
        w = base_weights.copy()
        # class weights: HIGH×3, MID×2, LOW×1
        w = w * np.where(y >= 0.75, 3.0, np.where(y >= 0.25, 2.0, 1.0))

        # Duplicate rows by integer weight
        rep = np.clip(np.round(w / w.min() if w.min() > 0 else w).astype(int), 1, 5)
        Xb, yb = [], []
        for xi, yi, ri in zip(features, y, rep):
            for _ in range(int(ri)):
                Xb.append(xi)
                yb.append(yi)
        Xb = np.array(Xb)
        yb = np.array(yb)

        reg = MLPRegressor(
            hidden_layer_sizes=(32, 16), max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.15,
        )
        reg.fit(Xb, yb)
        regressors[dim] = reg
        with open(MODEL_DIR / f"{dim}.pkl", "wb") as f:
            pickle.dump(reg, f)
        logger.info("Trained %s on %d rows (expanded from %d)", dim, len(Xb), len(features))

    with open(MODEL_DIR / "centroids.pkl", "wb") as f:
        pickle.dump(centroids, f)
    logger.info("Saved centroids + %d regressors to %s", len(regressors), MODEL_DIR)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== Model 3 v3 training ===")
    bank = _collect_text_bank()
    examples = build_pairs(bank, n_pairs_per_method=350)
    model = finetune_st(examples, epochs=3, batch_size=16)
    centroids = build_centroids(model)
    train_regressors(model, centroids)
    logger.info("=== DONE → %s ===", MODEL_DIR)


if __name__ == "__main__":
    main()
