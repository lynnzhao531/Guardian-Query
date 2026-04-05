"""Model 4 v3 — K*-hypothesis bottleneck with CONTINUOUS features (0-10 / 10).

Changes vs v2:
  - Haiku prompt asks 0-10 strength per K* hypothesis, not binary
  - Features stored as [0,1] continuous, giving the Ridge regressor more signal
  - Training data from outputs/combined_training_data.csv with sample_weight
  - Saves to models/model4_v3/ (does NOT overwrite models/model4/)
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import re
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
KSTAR_PATH = ROOT / "knowledge_base" / "K_star.json"
MODEL_DIR = ROOT / "models" / "model4_v3"
COMBINED = ROOT / "outputs" / "combined_training_data.csv"

with open(KSTAR_PATH) as f:
    K_STAR = json.load(f)["hypotheses"]

# Reuse the same keyword set as v2 so v3 is a drop-in alternative
ALL_KEYWORD_COLS: List[str] = sorted({
    "case study", "case studies", "case example", "policy", "program",
    "scheme", "service", "government", "public sector",
    "randomized controlled trial", "randomised controlled trial",
    "randomized trial", "randomised trial", "field experiment",
    "A/B test", "AB test", "clinical trial", "placebo", "control group",
    "public health", "education", "transport", "policing", "council",
    "city", "before and after", "before-and-after", "pre and post",
    "pre-post", "baseline and follow-up", "post-implementation",
    "post implementation", "evaluation", "study", "analysis",
    "assessment", "has decided to", "the minister decided",
    "the mayor decided", "the council decided",
    "the government decided to", "measure", "regulation",
    "analysis of administrative data", "administrative data analysis",
    "observational study", "observational analysis", "real-world evidence",
    "retrospective analysis", "regression analysis",
    "econometric analysis", "population",
})

FEATURE_DIM = len(K_STAR) + len(ALL_KEYWORD_COLS)

DIMENSIONS = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]

_api_stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
_regressors: Dict[str, Ridge] = {}


# ── Haiku call ───────────────────────────────────────────────────────────────

def _get_client():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    import anthropic
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0)


def _call_haiku(prompt: str, max_retries: int = 3) -> str:
    client = _get_client()
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=384,
                messages=[{"role": "user", "content": prompt}],
            )
            _api_stats["calls"] += 1
            _api_stats["input_tokens"] += resp.usage.input_tokens
            _api_stats["output_tokens"] += resp.usage.output_tokens
            return resp.content[0].text.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def _build_prompt(title: str, body: str) -> str:
    principles = "\n".join(f"{i+1}. {h}" for i, h in enumerate(K_STAR))
    body_trunc = (body or "")[:6000]
    return (
        f"Article title: {title}\n\nArticle text:\n{body_trunc}\n\n"
        f"For each principle below, rate how strongly this article exhibits it "
        f"on a 0-10 integer scale (0=not at all, 10=central to the article).\n"
        f"Principles:\n{principles}\n\n"
        f"Answer as a comma-separated list of {len(K_STAR)} integers 0-10."
    )


def _parse_response(text: str) -> np.ndarray:
    nums = re.findall(r"\b(10|[0-9])\b", text)
    arr = np.zeros(len(K_STAR), dtype=np.float64)
    for i, v in enumerate(nums[:len(K_STAR)]):
        arr[i] = float(v) / 10.0  # normalise 0-10 → [0,1]
    return arr


def _keyword_features(title: str, body: str) -> np.ndarray:
    combined = ((title or "") + " " + (body or "")).lower()
    return np.array(
        [1.0 if kw.lower() in combined else 0.0 for kw in ALL_KEYWORD_COLS],
        dtype=np.float64,
    )


# ── Feature extraction ──────────────────────────────────────────────────────

def extract_features(title: str, body: str) -> np.ndarray:
    raw = _call_haiku(_build_prompt(title, body))
    kstar = _parse_response(raw)
    kw = _keyword_features(title, body)
    return np.concatenate([kstar, kw])


def extract_features_offline(title: str, body: str) -> np.ndarray:
    """Training-time: no API. K* portion set to zeros, Ridge learns on keywords."""
    return np.concatenate([np.zeros(len(K_STAR)), _keyword_features(title, body)])


# Layer 2: continuous pass-through replaces the 3-bucket discretization.
# Rollback: restore the commented block below.
# def _discrete(pred: float) -> dict:
#     pred = float(np.clip(pred, 0.0, 1.0))
#     if pred < 0.25:
#         return {"score": 0.0, "p0": 0.80, "p05": 0.15, "p1": 0.05}
#     if pred < 0.75:
#         return {"score": 0.5, "p0": 0.20, "p05": 0.60, "p1": 0.20}
#     return     {"score": 1.0, "p0": 0.05, "p05": 0.15, "p1": 0.80}

def _discrete(pred: float) -> dict:
    """Continuous pass-through: Ridge prediction IS p1."""
    pred = float(np.clip(pred, 0.0, 1.0))
    remaining = 1.0 - pred
    if pred >= 0.5:
        p05 = remaining * 0.7
        p0 = remaining * 0.3
    else:
        p0 = remaining * 0.7
        p05 = remaining * 0.3
    return {"score": pred, "p0": p0, "p05": p05, "p1": pred}


# ── Training ────────────────────────────────────────────────────────────────

def train() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(COMBINED, low_memory=False)
    df = df[df["title"].astype(str).str.len() > 5].reset_index(drop=True)

    logger.info("Building offline features for %d rows...", len(df))
    X = np.array([
        extract_features_offline(str(r.get("title", "")), str(r.get("body_excerpt", "")))
        for _, r in df.iterrows()
    ])

    base_weights = df["sample_weight"].astype(float).values

    for dim in DIMENSIONS:
        if dim == "decision":
            y = df["label"].astype(float).values
        else:
            key = dim.replace("method_", "")
            y = np.array([
                float(r["label"]) if str(r["method_dimension"]) == key else 0.0
                for _, r in df.iterrows()
            ])

        # Class weights on top of base sample_weight
        class_w = np.where(y >= 0.75, 3.0, np.where(y >= 0.25, 2.0, 1.0))
        w = base_weights * class_w

        reg = Ridge(alpha=1.0)
        reg.fit(X, y, sample_weight=w)
        _regressors[dim] = reg
        with open(MODEL_DIR / f"{dim}.pkl", "wb") as f:
            pickle.dump(reg, f)
        logger.info("Trained %s on %d rows, nonzero y=%d",
                    dim, len(X), int((y > 0).sum()))

    with open(MODEL_DIR / "feature_meta.json", "w") as f:
        json.dump({"k_star_count": len(K_STAR),
                   "keyword_cols": ALL_KEYWORD_COLS,
                   "feature_dim": FEATURE_DIM,
                   "version": "v3_continuous"}, f, indent=2)
    logger.info("Model 4 v3 trained → %s", MODEL_DIR)


def load_models() -> None:
    _regressors.clear()
    for dim in DIMENSIONS:
        p = MODEL_DIR / f"{dim}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                _regressors[dim] = pickle.load(f)


def score_article(title: str, body: str) -> dict:
    if not _regressors:
        load_models()
    features = extract_features(title, body)
    out = {}
    for dim in DIMENSIONS:
        if dim in _regressors:
            pred = float(np.clip(_regressors[dim].predict(features.reshape(1, -1))[0], 0.0, 1.0))
        else:
            pred = 0.0
        out[dim] = _discrete(pred)
    return out


def get_api_stats() -> dict:
    cost = (_api_stats["input_tokens"] * 0.25 + _api_stats["output_tokens"] * 1.25) / 1e6
    return {**_api_stats, "estimated_cost_usd": round(cost, 4)}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        print("Usage: python src/model4_v3.py train")
