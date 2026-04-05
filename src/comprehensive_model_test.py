"""Task 7 — Comprehensive test of all model versions on 30 articles.

Selects:
  - 15 GOLD HIGH   (expert-verified relevant, label = 1.0, weight = 1.0)
  -  5 MID         (label ~ 0.5, Training_cases.csv rubric 2-3)
  - 10 LOW         (label < 0.25, LIKELY_LOW or from low-scoring files)

Scores each article with every available model version:
  M1, M1_v3, M2-old, M2-new, M3, M3_v3, M4, M4_v3, M5

Outputs:
  outputs/comprehensive_test_results.csv   per-article scores
  outputs/comprehensive_test_summary.csv   per-model accuracy + top-method hit
  outputs/comprehensive_test_correlation.csv  error correlation matrix
  outputs/comprehensive_test_report.md     recommendations

This script is SAFE to run while the pipeline runs: it only reads model files.
It does NOT touch any file the pipeline is writing to.
"""
from __future__ import annotations

import importlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("test")

ROOT = Path(__file__).resolve().parent.parent
COMBINED = ROOT / "outputs" / "combined_training_data.csv"
OUT_DIR = ROOT / "outputs"

# Model modules to test. (key, module, callable_name, label)
MODELS_TO_TEST = [
    ("m1_old",    "model1_llm_judge",              "score_article", "M1 (v2)"),
    ("m1_v3",     "model1_llm_judge_v3",           "score_article", "M1 v3 (decomposed)"),
    ("m2old",     "model2_old",                    "score_article", "M2-old"),
    ("m2new",     "model2_new",                    "score_article", "M2-new"),
    ("m3_old",    "model3_embedding_classifier",   "score_article", "M3 (v2)"),
    ("m3_v3",     "model3_v3",                     "score_article", "M3 v3 (contrastive)"),
    ("m4_old",    "model4_hypothesis_classifier",  "score_article", "M4 (v2)"),
    ("m4_v3",     "model4_v3",                     "score_article", "M4 v3 (continuous)"),
    ("m5",        "model5_deberta",                "score_article", "M5 (DistilBERT)"),
]

DIMENSIONS = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]


def _load_module(name):
    try:
        return importlib.import_module(name)
    except Exception as e:
        logger.warning("Could not import %s: %s", name, e)
        return None


def pick_test_set() -> pd.DataFrame:
    df = pd.read_csv(COMBINED, low_memory=False)
    df = df[df["title"].astype(str).str.len() > 10]
    df = df[df["body_excerpt"].astype(str).str.len() > 100]

    gold_high = df[(df["sample_weight"] >= 0.8) & (df["label"] >= 0.9)]
    mid = df[(df["label"] >= 0.4) & (df["label"] <= 0.6)]
    low = df[df["label"] < 0.2]

    rng = np.random.default_rng(42)
    g = gold_high.sample(n=min(15, len(gold_high)), random_state=42)
    m = mid.sample(n=min(5, len(mid)), random_state=42) if len(mid) > 0 else mid
    l = low.sample(n=min(10, len(low)), random_state=42)

    test = pd.concat([
        g.assign(gold_bucket="HIGH"),
        m.assign(gold_bucket="MID"),
        l.assign(gold_bucket="LOW"),
    ], ignore_index=True)
    logger.info("Test set: %d HIGH, %d MID, %d LOW", len(g), len(m), len(l))
    return test


def score_one(mod, fn, title: str, body: str) -> dict | None:
    try:
        return getattr(mod, fn)(title, body)
    except Exception as e:
        logger.warning("score failed: %s", e)
        return None


def main():
    test = pick_test_set()
    rows = []
    cached = {}

    for key, mod_name, fn_name, label in MODELS_TO_TEST:
        mod = _load_module(mod_name)
        if mod is None:
            continue
        cached[key] = (mod, fn_name, label)

    for idx, r in test.iterrows():
        title = str(r.get("title", ""))
        body = str(r.get("body_excerpt", ""))
        gold_bucket = r.get("gold_bucket")
        gold_method = str(r.get("method_dimension", ""))

        row = {
            "idx": idx,
            "title": title[:80],
            "gold_bucket": gold_bucket,
            "gold_method": gold_method,
            "gold_label": float(r.get("label", 0)),
        }
        for key, (mod, fn_name, label) in cached.items():
            logger.info("[%s] scoring article %d (%s)", key, idx, gold_bucket)
            t0 = time.time()
            result = score_one(mod, fn_name, title, body)
            dur = time.time() - t0
            if result is None:
                row[f"{key}_decision"] = None
                row[f"{key}_max_method"] = None
                row[f"{key}_top_method"] = None
                row[f"{key}_high"] = 0
                continue
            d_p1 = float(result.get("decision", {}).get("p1", 0))
            m_p1 = {k: float(result.get(k, {}).get("p1", 0))
                    for k in DIMENSIONS if k != "decision"}
            max_method = max(m_p1.values()) if m_p1 else 0.0
            top_method = max(m_p1, key=m_p1.get) if m_p1 else None
            high = int(d_p1 >= 0.75 and max_method >= 0.75)
            row[f"{key}_decision"] = d_p1
            row[f"{key}_max_method"] = max_method
            row[f"{key}_top_method"] = top_method
            row[f"{key}_high"] = high
            row[f"{key}_time_s"] = round(dur, 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "comprehensive_test_results.csv", index=False)
    logger.info("Per-article results → outputs/comprehensive_test_results.csv")

    # Summary: per-model HIGH accuracy
    summary_rows = []
    for key, (_, _, label) in cached.items():
        high_col = f"{key}_high"
        tm_col = f"{key}_top_method"
        if high_col not in df.columns:
            continue
        # True positives: gold HIGH and predicted HIGH
        gh = df["gold_bucket"] == "HIGH"
        tp = int(((df[high_col] == 1) & gh).sum())
        fn_ = int(((df[high_col] == 0) & gh).sum())
        # True negatives: gold LOW and predicted not HIGH
        gl = df["gold_bucket"] == "LOW"
        tn = int(((df[high_col] == 0) & gl).sum())
        fp = int(((df[high_col] == 1) & gl).sum())
        # Top-method hit on gold HIGH with known method
        method_hits = 0
        method_totals = 0
        for _, r in df[gh].iterrows():
            if not r["gold_method"] or r["gold_method"] == "mixed":
                continue
            method_totals += 1
            if r[tm_col] == f"method_{r['gold_method']}":
                method_hits += 1
        summary_rows.append({
            "model": label,
            "key": key,
            "HIGH_recall": f"{tp}/{int(gh.sum())} ({tp/max(1,int(gh.sum())):.0%})",
            "LOW_specificity": f"{tn}/{int(gl.sum())} ({tn/max(1,int(gl.sum())):.0%})",
            "HIGH_precision": f"{tp}/{tp+fp} ({tp/max(1,tp+fp):.0%})",
            "top_method_hit": f"{method_hits}/{method_totals}" if method_totals else "N/A",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "comprehensive_test_summary.csv", index=False)
    logger.info("\n%s", summary_df.to_string(index=False))

    # Error correlation matrix
    model_keys = [k for k, _, _, _ in MODELS_TO_TEST if f"{k}_high" in df.columns]
    err_matrix = pd.DataFrame(index=model_keys, columns=model_keys, dtype=float)
    for a in model_keys:
        for b in model_keys:
            ea = df[f"{a}_high"].astype(float)
            eb = df[f"{b}_high"].astype(float)
            if ea.std() == 0 or eb.std() == 0:
                err_matrix.loc[a, b] = np.nan
            else:
                err_matrix.loc[a, b] = round(float(ea.corr(eb)), 3)
    err_matrix.to_csv(OUT_DIR / "comprehensive_test_correlation.csv")
    logger.info("Correlation matrix:\n%s", err_matrix)

    # Recommendations
    report_lines = ["# Comprehensive model test — recommendations", ""]
    report_lines.append(f"Test set: 15 HIGH, 5 MID, 10 LOW from combined_training_data.csv")
    report_lines.append("")
    report_lines.append("## Per-model accuracy")
    report_lines.append(summary_df.to_markdown(index=False))
    report_lines.append("")
    report_lines.append("## Error correlation (HIGH-flag co-occurrence)")
    report_lines.append(err_matrix.to_markdown())
    report_lines.append("")
    report_lines.append("## Recommendations")

    def rec(old, new, label_old, label_new):
        if old not in df.columns or new not in df.columns:
            return f"- **{label_new}**: missing, cannot compare."
        old_recall = ((df[f"{old}_high"] == 1) & (df["gold_bucket"] == "HIGH")).sum()
        new_recall = ((df[f"{new}_high"] == 1) & (df["gold_bucket"] == "HIGH")).sum()
        old_fp = ((df[f"{old}_high"] == 1) & (df["gold_bucket"] == "LOW")).sum()
        new_fp = ((df[f"{new}_high"] == 1) & (df["gold_bucket"] == "LOW")).sum()
        if new_recall >= old_recall and new_fp <= old_fp:
            return f"- **{label_new}** SWAP: recall {new_recall}≥{old_recall}, FP {new_fp}≤{old_fp}."
        return f"- **{label_new}** KEEP OLD: recall {new_recall} vs {old_recall}, FP {new_fp} vs {old_fp}."

    report_lines.append(rec("m1_old_high", "m1_v3_high", "M1 old", "M1 v3"))
    report_lines.append(rec("m3_old_high", "m3_v3_high", "M3 old", "M3 v3"))
    report_lines.append(rec("m4_old_high", "m4_v3_high", "M4 old", "M4 v3"))
    report_lines.append("- **M5**: new model, evaluate error correlation < 0.5 with M1/M3/M4 before including.")

    (OUT_DIR / "comprehensive_test_report.md").write_text("\n".join(report_lines))
    logger.info("Report → outputs/comprehensive_test_report.md")


if __name__ == "__main__":
    main()
