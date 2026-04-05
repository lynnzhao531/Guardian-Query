"""Build a 100-article stratified test set, excluding the prior 30 articles.

Stratification:
  30 GOLD HIGH  (5 per method × 6 methods; fewer if a gold file is small)
  10 MID        (2 per scored SILVER file × 5 files)
  20 LOW        (4 per scored SILVER file × 5 files)
  20 Tier B     (real pipeline borderline cases)
  20 Pipeline LOW (real pipeline negatives)
  ───────────
 100 total

Output: outputs/test_set_100.csv
Columns: bucket, method_hint, url, title, body_excerpt, source
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
ROUNDS = OUTPUTS / "rounds"

RNG_SEED = 2026

GOLD_FILES = {
    "rct":              ("rct.csv",                      "body"),
    "prepost":          ("prepost.csv",                  "article_body"),
    "case_study":       ("casestudy.csv",                "article_body"),
    "expert_qual":      ("expert_qual.csv",              "body_text"),
    "expert_secondary": ("expert_secondary_quant.csv",   "body_text"),
    "gut":              ("gut_decision.csv",             "body_text"),
}

SILVER_FILES = [
    "rct 2.csv",
    "prepost 2.csv",
    "case studies.csv",
    "quantitative.csv",
    "gut.csv",
]


def _prior_urls() -> set[str]:
    """Reproduce the prior 30-article test set and return their URLs."""
    df = pd.read_csv(OUTPUTS / "combined_training_data.csv", low_memory=False)
    df = df[df["title"].astype(str).str.len() > 10]
    df = df[df["body_excerpt"].astype(str).str.len() > 100]
    gh = df[(df["sample_weight"] >= 0.8) & (df["label"] >= 0.9)]
    mid = df[(df["label"] >= 0.4) & (df["label"] <= 0.6)]
    low = df[df["label"] < 0.2]
    g = gh.sample(n=min(15, len(gh)), random_state=42)
    m = mid.sample(n=min(5, len(mid)), random_state=42) if len(mid) > 0 else mid
    l = low.sample(n=min(10, len(low)), random_state=42)
    prior = pd.concat([g, m, l])
    return set(prior["url_canon"].dropna().astype(str).tolist())


def _pick_gold(prior_urls: set[str]) -> list[dict]:
    rows = []
    for method, (fname, body_col) in GOLD_FILES.items():
        path = DATA / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[~df["url"].astype(str).isin(prior_urls)]
        df = df[df[body_col].astype(str).str.len() > 100] if body_col in df.columns else df
        n = min(5, len(df))
        if n == 0:
            continue
        picked = df.sample(n=n, random_state=RNG_SEED)
        for _, r in picked.iterrows():
            rows.append({
                "bucket": "GOLD_HIGH",
                "method_hint": method,
                "url": str(r["url"]),
                "title": str(r.get("title", "")),
                "body_excerpt": str(r.get(body_col, ""))[:1200],
                "source": f"data/{fname}",
            })
    return rows


def _pick_silver(prior_urls: set[str], n_mid_each: int, n_low_each: int,
                 taken: set[str]) -> tuple[list[dict], list[dict]]:
    mids, lows = [], []
    for fname in SILVER_FILES:
        path = DATA / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        if "Method" not in df.columns or "Decision" not in df.columns:
            continue
        df = df[~df["url"].astype(str).isin(prior_urls | taken)]
        df = df[df["body"].astype(str).str.len() > 100] if "body" in df.columns else df

        mid_df = df[(df["Method"].astype(float) == 0.5) | (df["Decision"].astype(float) == 0.5)]
        low_df = df[(df["Method"].astype(float) == 0.0) & (df["Decision"].astype(float) == 0.0)]

        mid_pick = mid_df.sample(n=min(n_mid_each, len(mid_df)), random_state=RNG_SEED)
        low_pick = low_df.sample(n=min(n_low_each, len(low_df)), random_state=RNG_SEED)

        for _, r in mid_pick.iterrows():
            mids.append({
                "bucket": "SILVER_MID",
                "method_hint": "",
                "url": str(r["url"]),
                "title": str(r.get("title", "")),
                "body_excerpt": str(r.get("body", ""))[:1200],
                "source": f"data/{fname}",
            })
            taken.add(str(r["url"]))
        for _, r in low_pick.iterrows():
            lows.append({
                "bucket": "SILVER_LOW",
                "method_hint": "",
                "url": str(r["url"]),
                "title": str(r.get("title", "")),
                "body_excerpt": str(r.get("body", ""))[:1200],
                "source": f"data/{fname}",
            })
            taken.add(str(r["url"]))
    return mids, lows


def _pick_pipeline(prior_urls: set[str], taken: set[str]) -> tuple[list[dict], list[dict]]:
    # Body text comes from combined_training_data.csv (indexed by url_canon)
    comb = pd.read_csv(OUTPUTS / "combined_training_data.csv", low_memory=False)
    body_by_url: dict[str, tuple[str, str]] = {}
    for _, r in comb.iterrows():
        body = str(r.get("body_excerpt", ""))
        if len(body) < 100:  # skip rows with missing body
            continue
        body_by_url[str(r["url_canon"])] = (str(r.get("title", "")), body)

    # Tier B urls from pipeline
    tb_urls: list[str] = []
    tb_seen: set[str] = set()
    for f in sorted(glob.glob(str(ROUNDS / "round_*/round_*_tier_b_papers.csv"))):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "url" not in d.columns:
            continue
        for u in d["url"].dropna().astype(str).tolist():
            if u in tb_seen or u in prior_urls or u in taken:
                continue
            if u not in body_by_url:
                continue
            tb_seen.add(u)
            tb_urls.append(u)

    # Pipeline LOW urls: all model p1 < 0.25 in scored_results_full
    low_urls: list[str] = []
    low_seen: set[str] = set()
    for f in sorted(glob.glob(str(ROUNDS / "round_*/scored_results_full.csv"))):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        mcols = [c for c in d.columns if c.endswith("_decision_p1")]
        if not mcols:
            continue
        maxp1 = d[mcols].astype(float).max(axis=1)
        cand = d[maxp1 < 0.25]
        for _, r in cand.iterrows():
            u = str(r.get("url", ""))
            if u in low_seen or u in prior_urls or u in taken or u in tb_seen:
                continue
            if u not in body_by_url:
                continue
            low_seen.add(u)
            low_urls.append(u)

    rng_tb = pd.Series(tb_urls).sample(n=min(20, len(tb_urls)), random_state=RNG_SEED).tolist()
    rng_low = pd.Series(low_urls).sample(n=min(20, len(low_urls)), random_state=RNG_SEED).tolist()

    tb_rows = [{
        "bucket": "PIPELINE_TIER_B",
        "method_hint": "",
        "url": u,
        "title": body_by_url[u][0],
        "body_excerpt": body_by_url[u][1][:1200],
        "source": "pipeline:tier_b",
    } for u in rng_tb]
    low_rows = [{
        "bucket": "PIPELINE_LOW",
        "method_hint": "",
        "url": u,
        "title": body_by_url[u][0],
        "body_excerpt": body_by_url[u][1][:1200],
        "source": "pipeline:low",
    } for u in rng_low]
    return tb_rows, low_rows


def main():
    prior = _prior_urls()
    print(f"Prior 30 URLs to exclude: {len(prior)}")

    taken: set[str] = set()

    gold = _pick_gold(prior)
    for r in gold:
        taken.add(r["url"])
    print(f"GOLD_HIGH picked: {len(gold)}")

    mids, lows = _pick_silver(prior, n_mid_each=2, n_low_each=4, taken=taken)
    print(f"SILVER_MID: {len(mids)}, SILVER_LOW: {len(lows)}")

    tb, plow = _pick_pipeline(prior, taken)
    print(f"PIPELINE_TIER_B: {len(tb)}, PIPELINE_LOW: {len(plow)}")

    all_rows = gold + mids + lows + tb + plow
    df = pd.DataFrame(all_rows)
    df = df[df["body_excerpt"].astype(str).str.len() > 50].reset_index(drop=True)
    df.to_csv(OUTPUTS / "test_set_100.csv", index=False)
    print(f"TOTAL: {len(df)} → outputs/test_set_100.csv")
    print(df["bucket"].value_counts().to_dict())


if __name__ == "__main__":
    main()
