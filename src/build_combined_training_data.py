"""Task 1 — Build combined training dataset with GOLD/SILVER/BRONZE/UNCERTAIN/LIKELY_LOW tiers.

Per REVISED_ARCHITECTURE.md §12:
  GOLD (1.0): expert CSVs + Training_cases.csv (all labelled positive)
  SILVER (0.8): human-scored CSVs with Method/Decision columns
  BRONZE (0.5): pipeline Tier A articles
  UNCERTAIN (0.2): pipeline Tier B with 2+ models
  LIKELY_LOW (0.3): scored by pipeline, not in any tier

Outputs:
  outputs/all_scored_articles.csv  — merged pipeline scored_results_full
  outputs/combined_training_data.csv  — final training file with sample_weight
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
ROUNDS = OUTPUTS / "rounds"

# Columns of final combined file
OUT_COLS = [
    "url_canon", "title", "body_excerpt", "label", "sample_weight",
    "source", "method_dimension", "decision_score", "method_score",
]


def canon_url(u: str) -> str:
    if not isinstance(u, str) or not u:
        return ""
    try:
        p = urlparse(u.strip().lower())
        path = re.sub(r"/+$", "", p.path)
        return urlunparse((p.scheme, p.netloc, path, "", "", ""))
    except Exception:
        return u.strip().lower()


def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _body_excerpt(text: str, n: int = 2000) -> str:
    if not isinstance(text, str):
        return ""
    return text[:n].replace("\r", " ").replace("\n", " ").strip()


# ── GOLD / SILVER from data/ CSVs ────────────────────────────────────────────

# Mapping: file → (method_dimension, tier, source_label)
DATA_FILES = {
    "rct.csv":                  ("rct",                "GOLD",   "expert_rct"),
    "rct 2.csv":                ("rct",                "SILVER", "human_rct"),
    "prepost.csv":              ("prepost",            "GOLD",   "expert_prepost"),
    "prepost 2.csv":            ("prepost",            "SILVER", "human_prepost"),
    "casestudy.csv":            ("case_study",         "GOLD",   "expert_casestudy"),
    "case studies.csv":         ("case_study",         "SILVER", "human_casestudy"),
    "expert_qual.csv":          ("expert_qual",        "GOLD",   "expert_qual"),
    "expert_secondary_quant.csv":("expert_secondary",  "GOLD",   "expert_secondary"),
    "quantitative.csv":         ("expert_secondary",   "SILVER", "human_quant"),
    "gut.csv":                  ("gut",                "GOLD",   "expert_gut"),
    "gut_decision.csv":         ("gut",                "SILVER", "human_gut"),
    "Training_cases.csv":       ("mixed",              "GOLD",   "training_cases"),
}

TIER_WEIGHTS = {
    "GOLD": 1.0,
    "SILVER": 0.8,
    "BRONZE": 0.5,
    "UNCERTAIN": 0.2,
    "LIKELY_LOW": 0.3,
}


def load_expert_silver() -> pd.DataFrame:
    rows = []
    for fname, (method, tier, source) in DATA_FILES.items():
        path = DATA / fname
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            continue
        if df.empty:
            continue

        url_col = _first_col(df, ["url", "URL"])
        title_col = _first_col(df, ["title", "Title", "headline"])
        body_col = _first_col(df, ["body", "bodyText", "body_text", "article_body"])

        if fname == "Training_cases.csv":
            # Has method_category + prototype_score_0to5 + rubric_score_0to5 + bodyText
            method_col = "method_category" if "method_category" in df.columns else None
            score_col = "rubric_score_0to5" if "rubric_score_0to5" in df.columns else "prototype_score_0to5"
            for _, r in df.iterrows():
                raw_method = str(r.get(method_col, "")).strip().lower() if method_col else ""
                mm = {
                    "rct": "rct", "prepost": "prepost", "pre_post": "prepost",
                    "pre-post": "prepost", "case_study": "case_study",
                    "casestudy": "case_study", "expert_qual": "expert_qual",
                    "expert_qualitative": "expert_qual",
                    "expert_secondary": "expert_secondary",
                    "expert_secondary_quant": "expert_secondary",
                    "quantitative": "expert_secondary", "gut": "gut",
                }.get(raw_method, raw_method or "mixed")
                try:
                    s = float(r.get(score_col, 0) or 0) / 5.0
                except Exception:
                    s = 0.5
                label = 1.0 if s >= 0.6 else (0.0 if s < 0.2 else 0.5)
                rows.append({
                    "url_canon": canon_url(str(r.get(url_col, ""))) if url_col else "",
                    "title": str(r.get(title_col, "") or ""),
                    "body_excerpt": _body_excerpt(str(r.get(body_col, "") or "")),
                    "label": label,
                    "sample_weight": TIER_WEIGHTS[tier],
                    "source": source,
                    "method_dimension": mm,
                    "decision_score": label,
                    "method_score": label,
                })
            continue

        # For the other files: GOLD = all positive; SILVER = use Method/Decision cols if present
        has_method_col = "Method" in df.columns
        has_decision_col = "Decision" in df.columns
        for _, r in df.iterrows():
            if has_method_col and has_decision_col:
                try:
                    mscore = float(r.get("Method", 0) or 0)
                except Exception:
                    mscore = 0.0
                try:
                    dscore = float(r.get("Decision", 0) or 0)
                except Exception:
                    dscore = 0.0
                # Normalize to 0-1 (CSVs use 0/0.5/1)
                mscore = max(0.0, min(1.0, mscore))
                dscore = max(0.0, min(1.0, dscore))
                label = min(mscore, dscore) if (mscore > 0 and dscore > 0) else max(mscore, dscore) / 2
            else:
                # All-positive expert file
                mscore = 1.0
                dscore = 1.0
                label = 1.0

            rows.append({
                "url_canon": canon_url(str(r.get(url_col, ""))) if url_col else "",
                "title": str(r.get(title_col, "") or ""),
                "body_excerpt": _body_excerpt(str(r.get(body_col, "") or "")) if body_col else "",
                "label": label,
                "sample_weight": TIER_WEIGHTS[tier],
                "source": source,
                "method_dimension": method,
                "decision_score": dscore,
                "method_score": mscore,
            })

    df = pd.DataFrame(rows, columns=OUT_COLS)
    print(f"  expert+human rows: {len(df)}")
    return df


# ── Pipeline scored_results_full (BRONZE / UNCERTAIN / LIKELY_LOW) ───────────

def load_pipeline_scored() -> pd.DataFrame:
    frames = []
    for rd in sorted(ROUNDS.glob("round_*/scored_results_full.csv")):
        try:
            df = pd.read_csv(rd, low_memory=False)
            df["_source_round"] = rd.parent.name
            frames.append(df)
        except Exception as e:
            print(f"  SKIP {rd}: {e}")
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    # Save merged snapshot
    merged.to_csv(OUTPUTS / "all_scored_articles.csv", index=False)
    print(f"  merged {len(frames)} scored files → {len(merged)} rows → outputs/all_scored_articles.csv")
    return merged


def pipeline_to_training(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(columns=OUT_COLS)
    rows = []
    for _, r in scored.iterrows():
        tier = str(r.get("tier", "") or "").upper()
        models_high = int(r.get("models_agreeing_high", 0) or 0)
        if tier == "A":
            data_tier = "BRONZE"
        elif tier == "B" and models_high >= 2:
            data_tier = "UNCERTAIN"
        else:
            data_tier = "LIKELY_LOW"

        method_dim = str(r.get("classified_method", "") or "") or "unknown"
        # Strip "method_" prefix that pipeline uses
        if method_dim.startswith("method_"):
            method_dim = method_dim[len("method_"):]
        rel = float(r.get("article_relevance_score", 0) or 0)
        # derive decision / method scores from m1 (primary) if present
        d_score = float(r.get("m1_decision_p1", 0) or 0)
        # max method p1 across m1
        m_cols = [c for c in r.index if c.startswith("m1_method_") and c.endswith("_p1")]
        m_score = max((float(r.get(c, 0) or 0) for c in m_cols), default=0.0)

        rows.append({
            "url_canon": canon_url(str(r.get("url", "") or "")),
            "title": str(r.get("title", "") or ""),
            "body_excerpt": "",  # not stored in scored_results_full
            "label": rel if rel > 0 else (1.0 if tier == "A" else 0.5 if tier == "B" else 0.0),
            "sample_weight": TIER_WEIGHTS[data_tier],
            "source": f"pipeline_{data_tier.lower()}_{r.get('_source_round', '')}",
            "method_dimension": method_dim,
            "decision_score": d_score,
            "method_score": m_score,
        })
    df = pd.DataFrame(rows, columns=OUT_COLS)
    print(f"  pipeline rows: {len(df)}")
    return df


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("== Loading expert + human-scored CSVs ==")
    expert_df = load_expert_silver()
    print("== Loading pipeline scored_results_full ==")
    pipe_raw = load_pipeline_scored()
    pipe_df = pipeline_to_training(pipe_raw) if not pipe_raw.empty else pd.DataFrame(columns=OUT_COLS)

    combined = pd.concat([expert_df, pipe_df], ignore_index=True)
    # Drop empty urls + dedup (keep highest sample_weight)
    combined = combined[combined["url_canon"].astype(str).str.len() > 0].copy()
    combined = combined.sort_values("sample_weight", ascending=False)
    combined = combined.drop_duplicates(subset=["url_canon"], keep="first")

    out = OUTPUTS / "combined_training_data.csv"
    combined.to_csv(out, index=False)
    print(f"\n== Wrote {len(combined)} rows → {out} ==")
    print("\nBreakdown by source tier weight:")
    print(combined.groupby("sample_weight").size())
    print("\nBy method_dimension:")
    print(combined.groupby("method_dimension").size())
    print("\nLabel distribution:")
    print(combined["label"].describe())


if __name__ == "__main__":
    main()
