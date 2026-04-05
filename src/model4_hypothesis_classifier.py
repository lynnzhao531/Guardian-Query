from __future__ import annotations
"""Model 4: K*-Hypothesis Bottleneck Classifier (MASTER_PLAN_v3.md Section 5.4).
Single Haiku call -> binary K* vector + keyword indicators -> 7 Ridge regressors."""
from typing import Optional, List, Dict

import json, os, time, logging, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
KSTAR_PATH, MODEL_DIR, DATA_DIR = (
    ROOT / "knowledge_base" / "K_star.json", ROOT / "models" / "model4", ROOT / "data")

with open(KSTAR_PATH) as f:
    K_STAR = json.load(f)["hypotheses"]

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
# Regressor config: dimension -> (csv_files, dim_key)
REGRESSOR_DEFS = {
    "method_rct":              (["rct 2.csv", "rct.csv"], "method_rct"),
    "method_prepost":          (["prepost 2.csv", "prepost.csv"], "method_prepost"),
    "method_case_study":       (["case studies.csv", "casestudy.csv"], "method_case_study"),
    "method_expert_secondary": (["quantitative.csv", "expert_secondary_quant.csv"],
                                "method_expert_secondary"),
    "method_expert_qual":      (["expert_qual.csv"], "method_expert_qual"),
    "method_gut":              (["gut.csv", "gut_decision.csv"], "method_gut"),
    "decision":                ("ALL", "decision"),
}

def _continuous_to_discrete(pred: float) -> dict:
    """Map Ridge output to discrete score + probabilities."""
    if pred < 0.25:
        return {"score": 0.0, "p0": 0.8, "p05": 0.15, "p1": 0.05}
    elif pred < 0.75:
        return {"score": 0.5, "p0": 0.2, "p05": 0.6, "p1": 0.2}
    else:
        return {"score": 1.0, "p0": 0.05, "p05": 0.15, "p1": 0.8}


_api_stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

def _get_client():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    import anthropic
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0)


def _call_haiku(prompt: str, max_retries: int = 3) -> str:
    """Call Claude Haiku with retry + exponential backoff."""
    client = _get_client()
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            _api_stats["calls"] += 1
            _api_stats["input_tokens"] += resp.usage.input_tokens
            _api_stats["output_tokens"] += resp.usage.output_tokens
            return resp.content[0].text.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error("Haiku call failed after %d retries: %s", max_retries, e)
                raise
            wait = 2 ** attempt
            logger.warning("Haiku retry %d/%d in %ds: %s", attempt + 1, max_retries, wait, e)
            time.sleep(wait)


def _build_kstar_prompt(title: str, body_text: str) -> str:
    """Single prompt to classify all K* hypotheses."""
    principles = "\n".join(f"{i+1}. {h}" for i, h in enumerate(K_STAR))
    # Truncate body to ~6000 chars to stay within Haiku context cheaply
    body_trunc = (body_text or "")[:6000]
    return (
        f"Article title: {title}\n\nArticle text:\n{body_trunc}\n\n"
        f"For each principle below, answer 1 if this article clearly exhibits it, "
        f"0 if not.\nPrinciples:\n{principles}\n\n"
        f"Answer as comma-separated 0s and 1s only."
    )


def _parse_kstar_response(text: str) -> np.ndarray:
    """Parse comma-separated 0/1 response into array."""
    import re
    nums = re.findall(r"[01]", text)
    arr = np.zeros(len(K_STAR), dtype=np.float64)
    for i, v in enumerate(nums[:len(K_STAR)]):
        arr[i] = float(v)
    return arr


def _keyword_features(title: str, body_text: str) -> np.ndarray:
    """Check keyword presence in title+body -> binary vector aligned to ALL_KEYWORD_COLS."""
    combined = ((title or "") + " " + (body_text or "")).lower()
    return np.array([1.0 if kw.lower() in combined else 0.0 for kw in ALL_KEYWORD_COLS],
                    dtype=np.float64)


def extract_features(title: str, body_text: str) -> np.ndarray:
    """K* hypothesis features (via Haiku) + keyword indicator features -> 1-D array."""
    prompt = _build_kstar_prompt(title, body_text)
    raw = _call_haiku(prompt)
    kstar_feats = _parse_kstar_response(raw)
    kw_feats = _keyword_features(title, body_text)
    return np.concatenate([kstar_feats, kw_feats])


def extract_features_offline(title: str, body_text: str,
                             kstar_vector: np.ndarray | None = None) -> np.ndarray:
    """Feature extraction without API call (uses pre-computed K* vector or zeros)."""
    kstar_feats = kstar_vector if kstar_vector is not None else np.zeros(len(K_STAR))
    kw_feats = _keyword_features(title, body_text)
    return np.concatenate([kstar_feats, kw_feats])


BODY_COLS = ["body", "article_body", "body_text", "bodyText"]


def _detect_body(df: pd.DataFrame) -> Optional[str]:
    for c in BODY_COLS:
        if c in df.columns:
            return c
    return None


def _load_csv_rows(filename: str) -> List[dict]:
    """Load a CSV and return list of {title, body, Method, Decision, kw_features}."""
    path = DATA_DIR / filename
    if not path.exists():
        logger.warning("File not found: %s", path)
        return []
    df = pd.read_csv(path)
    body_col = _detect_body(df)
    rows = []
    for _, r in df.iterrows():
        title = str(r.get("title", ""))
        body = str(r.get(body_col, "")) if body_col else ""
        method_raw = r.get("Method") if "Method" in df.columns else 1.0
        decision_raw = r.get("Decision") if "Decision" in df.columns else 1.0
        method = float(method_raw) if pd.notna(method_raw) else 0.0
        decision = float(decision_raw) if pd.notna(decision_raw) else 0.0
        # Keyword features from CSV boolean columns directly if available
        combined_text = (title + " " + body).lower()
        kw_from_csv = np.array(
            [float(r[kw]) if (kw in df.columns and pd.notna(r.get(kw))) else
             (1.0 if kw.lower() in combined_text else 0.0)
             for kw in ALL_KEYWORD_COLS],
            dtype=np.float64,
        )
        kw_from_csv = np.nan_to_num(kw_from_csv, 0.0)
        rows.append({
            "title": title, "body": body,
            "method": min(max(round(method * 2) / 2, 0), 1),
            "decision": min(max(round(decision * 2) / 2, 0), 1),
            "kw_features": kw_from_csv,
        })
    return rows


def _gather_training_data(csv_files: List[str] | str, dim_key: str):
    """Build X, y arrays for one regressor. Uses keyword features only (no API)."""
    if csv_files == "ALL":
        csv_files = [f.name for f in sorted(DATA_DIR.glob("*.csv"))
                     if f.name != "Training_cases.csv"]
    all_rows = []
    for fname in csv_files:
        all_rows.extend(_load_csv_rows(fname))
    if not all_rows:
        return np.zeros((0, FEATURE_DIM)), np.zeros(0)
    # For gold-only files (expert_qual), all labels=1; add random low samples (y=0)
    # from other scored files to give the regressor negative examples.
    label_col = "method" if dim_key != "decision" else "decision"
    if all(row[label_col] >= 1.0 for row in all_rows):
        neg_rows = []
        for f in sorted(DATA_DIR.glob("*.csv")):
            if f.name in csv_files or f.name == "Training_cases.csv":
                continue
            neg_rows.extend(_load_csv_rows(f.name))
        if neg_rows:
            import random
            random.seed(42)
            n_neg = min(len(neg_rows), len(all_rows) * 3)
            sampled = random.sample(neg_rows, n_neg)
            for row in sampled:
                row["method"] = 0.0  # force label=0 for negatives
            all_rows.extend(sampled)
    # Build feature matrix (K* features are zeros during training — no API calls)
    X = np.array([
        np.concatenate([np.zeros(len(K_STAR)), row["kw_features"]])
        for row in all_rows
    ])
    y = np.array([row[label_col] for row in all_rows])
    return X, y

_regressors: Dict[str, Ridge] = {}

def train(data_loader_output: pd.DataFrame | None = None) -> None:
    """Train all 7 Ridge regressors and save to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for dim, (files, dim_key) in REGRESSOR_DEFS.items():
        X, y = _gather_training_data(files, dim_key)
        if len(X) == 0:
            logger.warning("No training data for %s — skipping", dim)
            continue
        reg = Ridge(alpha=1.0)
        reg.fit(X, y)
        _regressors[dim] = reg
        out_path = MODEL_DIR / f"{dim}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(reg, f)
        logger.info("Trained %s on %d samples, saved to %s", dim, len(X), out_path)
    # Save feature metadata
    meta = {"k_star_count": len(K_STAR), "keyword_cols": ALL_KEYWORD_COLS}
    with open(MODEL_DIR / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Model 4 training complete. {len(_regressors)} regressors saved.")


def load_models() -> None:
    """Load saved Ridge regressors from disk."""
    _regressors.clear()
    for dim in REGRESSOR_DEFS:
        path = MODEL_DIR / f"{dim}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                _regressors[dim] = pickle.load(f)
            logger.info("Loaded %s", path)
        else:
            logger.warning("Model file not found: %s", path)
    print(f"Model 4: loaded {len(_regressors)} regressors.")


SEVEN_VECTOR_KEYS = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]


def score_article(title: str, body_text: str) -> dict:
    """Score a single article. Returns 7-vector dict matching Model 1 format."""
    if not _regressors:
        load_models()

    features = extract_features(title, body_text)
    result = {}
    for dim in SEVEN_VECTOR_KEYS:
        if dim in _regressors:
            pred = float(_regressors[dim].predict(features.reshape(1, -1))[0])
            pred = np.clip(pred, 0.0, 1.0)
        else:
            pred = 0.0
        disc = _continuous_to_discrete(pred)
        result[dim] = {"score": disc["score"], "p0": disc["p0"],
                       "p05": disc["p05"], "p1": disc["p1"]}
    return result


def get_api_stats() -> dict:
    """Return cumulative API call statistics."""
    cost = (_api_stats["input_tokens"] * 0.25 + _api_stats["output_tokens"] * 1.25) / 1e6
    return {**_api_stats, "estimated_cost_usd": round(cost, 4)}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        print("Usage: python src/model4_hypothesis_classifier.py train")
        print("Then use score_article(title, body) from other modules.")
