"""Data Loader — auto-detect and unify CSV files in data/ into a 7-vector DataFrame.
Implements detection rules from MASTER_PLAN_v3.md Section 3."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SEVEN_VECTOR = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]

BODY_CANDIDATES = ["body", "article_body", "body_text", "bodyText"]
# Column-content patterns that map to method types (applied to non-score columns)
_METHOD_COL_PATTERNS = {
    "method_rct": ["randomized controlled trial", "randomised trial",
                   "randomized trial", "randomised controlled trial",
                   "field experiment", "a/b test", "ab test", "clinical trial"],
    "method_prepost": ["before and after", "pre-post", "pre and post",
                       "before-and-after", "baseline and follow-up"],
    "method_case_study": ["case study", "case studies", "case example"],
    "method_expert_secondary": ["administrative data", "observational study",
                                "observational analysis", "regression analysis",
                                "econometric analysis"],
    "method_gut": ["has decided to", "minister decided", "mayor decided",
                   "council decided", "government decided"],
}
# Filename-based fallback mapping
_FILENAME_METHOD = {
    "rct": "method_rct",
    "prepost": "method_prepost",
    "casestudy": "method_case_study",
    "case studies": "method_case_study",
    "expert_qual": "method_expert_qual",
    "expert_secondary_quant": "method_expert_secondary",
    "quantitative": "method_expert_secondary",
    "gut_decision": "method_gut",
    "gut": "method_gut",
}
# Training_cases method_category to 7-vector key
_CATEGORY_METHOD = {
    "RCT_Field_AB": "method_rct",
    "PrePost_Eval": "method_prepost",
    "Case_Study": "method_case_study",
    "Expert_Qual": "method_expert_qual",
    "Expert_Secondary_Quant": "method_expert_secondary",
    "Gut_Decision": "method_gut",
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def normalize_url(url: str) -> str:
    """Lowercase, strip trailing slash for dedup."""
    if pd.isna(url) or not url:
        return ""
    return str(url).strip().rstrip("/").lower()


def detect_body_col(df: pd.DataFrame) -> Optional[str]:
    """Return the first matching body column name, or None."""
    for col in BODY_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _is_scored(df: pd.DataFrame) -> bool:
    """True if df has numeric Method AND Decision columns."""
    if "Method" not in df.columns or "Decision" not in df.columns:
        return False
    try:
        pd.to_numeric(df["Method"], errors="raise")
        pd.to_numeric(df["Decision"], errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def get_method_type(filename: str, df: pd.DataFrame) -> str:
    """Detect method type from column names, then fall back to filename."""
    stem = Path(filename).stem.lower().strip()
    # Non-score columns (exclude standard fields)
    skip = {"id", "title", "date", "url", "section", "body", "article_body",
            "body_text", "bodytext", "method", "decision"}
    extra_cols = [c for c in df.columns if c.lower() not in skip]
    col_text = " ".join(c.lower() for c in extra_cols)

    for method, patterns in _METHOD_COL_PATTERNS.items():
        for pat in patterns:
            if pat in col_text:
                return method

    # Filename fallback
    for key, method in _FILENAME_METHOD.items():
        if key == stem or stem.startswith(key):
            return method

    return "method_expert_qual"  # safe default


def _empty_scores() -> dict:
    return {col: -1.0 for col in SEVEN_VECTOR}


def _make_row(url, title, body, source_file, method_type, scores) -> dict:
    return {
        "url": url, "title": title, "body_text": body,
        "source_file": source_file, "method_type": method_type,
        **scores,
    }


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

def get_gold_articles() -> pd.DataFrame:
    """Load all ALL-HIGH files. Every article: method=1, decision=1, rest=-1."""
    rows = []
    for fpath in sorted(DATA_DIR.glob("*.csv")):
        if fpath.name == "Training_cases.csv":
            continue
        df = pd.read_csv(fpath)
        body_col = detect_body_col(df)
        if body_col is None or _is_scored(df):
            continue
        method = get_method_type(fpath.name, df)
        for _, r in df.iterrows():
            scores = _empty_scores()
            scores["decision"] = 1.0
            scores[method] = 1.0
            rows.append(_make_row(
                r.get("url", ""), r.get("title", ""), r.get(body_col, ""),
                fpath.name, method, scores,
            ))
    return pd.DataFrame(rows)


def get_scored_articles() -> pd.DataFrame:
    """Load SCORED files. Map Method->detected method, Decision->decision."""
    rows = []
    for fpath in sorted(DATA_DIR.glob("*.csv")):
        if fpath.name == "Training_cases.csv":
            continue
        df = pd.read_csv(fpath)
        body_col = detect_body_col(df)
        if body_col is None or not _is_scored(df):
            continue
        method = get_method_type(fpath.name, df)
        for _, r in df.iterrows():
            scores = _empty_scores()
            m_val = r["Method"]
            d_val = r["Decision"]
            if pd.notna(m_val):
                m_val = float(m_val)
                scores[method] = min(max(round(m_val * 2) / 2, 0), 1)
            if pd.notna(d_val):
                d_val = float(d_val)
                scores["decision"] = min(max(round(d_val * 2) / 2, 0), 1)
            rows.append(_make_row(
                r.get("url", ""), r.get("title", ""), r.get(body_col, ""),
                fpath.name, method, scores,
            ))
    return pd.DataFrame(rows)


def get_training_articles() -> pd.DataFrame:
    """Load Training_cases.csv. Map rubric_score_0to5 to 7-vector."""
    path = DATA_DIR / "Training_cases.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    body_col = detect_body_col(df)
    if body_col is None:
        return pd.DataFrame()

    rows = []
    for _, r in df.iterrows():
        rubric = float(r.get("rubric_score_0to5", 0))
        cat = str(r.get("method_category", ""))
        method = _CATEGORY_METHOD.get(cat, "method_expert_qual")
        scores = _empty_scores()

        if rubric >= 3:
            scores["decision"] = 1.0
            scores[method] = 1.0
        elif rubric == 2:
            scores["decision"] = 0.5
            scores[method] = 0.5
        else:
            scores = {k: 0.0 for k in SEVEN_VECTOR}

        rows.append(_make_row(
            r.get("url", ""), r.get("title", ""), r.get(body_col, ""),
            "Training_cases.csv", method, scores,
        ))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def load_all_data() -> pd.DataFrame:
    """Load, merge, and deduplicate all data sources. Returns unified DataFrame."""
    parts = []
    for loader, label in [
        (get_gold_articles, "gold"),
        (get_scored_articles, "scored"),
        (get_training_articles, "training"),
    ]:
        chunk = loader()
        if not chunk.empty:
            print(f"  {label}: {len(chunk)} rows")
            parts.append(chunk)

    if not parts:
        return pd.DataFrame(columns=["url", "title", "body_text", "source_file",
                                      "method_type"] + SEVEN_VECTOR)

    combined = pd.concat(parts, ignore_index=True)

    # Normalize URLs for dedup
    combined["_norm_url"] = combined["url"].apply(normalize_url)
    # Keep first occurrence (gold > scored > training, per concat order)
    before = len(combined)
    combined = combined[combined["_norm_url"] != ""]  # drop missing URLs
    combined = combined.drop_duplicates(subset="_norm_url", keep="first")
    combined = combined.drop(columns="_norm_url").reset_index(drop=True)
    print(f"  Dedup: {before} -> {len(combined)} articles")

    return combined


# ------------------------------------------------------------------
if __name__ == "__main__":
    df = load_all_data()
    print(f"\nTotal articles: {len(df)}")
    print(f"Method types: {df['method_type'].value_counts().to_dict()}")
    print(f"Source files: {df['source_file'].nunique()}")
    for col in SEVEN_VECTOR:
        counts = df[col].value_counts().to_dict()
        print(f"  {col}: {counts}")
