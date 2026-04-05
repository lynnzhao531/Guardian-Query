"""Experiment A/B/C test harness — evaluates M1-v4, M1-v5, M3-v4 on the
same 30 articles used by comprehensive_model_test.py.

Loads OLD-model scores from outputs/comprehensive_test_results.csv when
available so we don't re-pay for them; only the 3 new modules are scored.
"""
from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("exp")

ROOT = Path(__file__).resolve().parent.parent
COMBINED = ROOT / "outputs" / "combined_training_data.csv"
OLD_RESULTS = ROOT / "outputs" / "comprehensive_test_results.csv"
OUT = ROOT / "outputs" / "experiment_test_results.csv"

DIMENSIONS = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]

NEW_MODELS = [
    ("m1_v4", "model1_llm_judge_v4", "M1 v4 (continuous 0-10)"),
    ("m1_v5", "model1_llm_judge_v5", "M1 v5 (holistic + 3-shot)"),
    ("m3_v4", "model3_v4",           "M3 v4 (hard-neg + MNRL)"),
]


def pick_test_set() -> pd.DataFrame:
    """Identical logic to comprehensive_model_test.pick_test_set."""
    df = pd.read_csv(COMBINED, low_memory=False)
    df = df[df["title"].astype(str).str.len() > 10]
    df = df[df["body_excerpt"].astype(str).str.len() > 100]
    gold_high = df[(df["sample_weight"] >= 0.8) & (df["label"] >= 0.9)]
    mid = df[(df["label"] >= 0.4) & (df["label"] <= 0.6)]
    low = df[df["label"] < 0.2]
    g = gold_high.sample(n=min(15, len(gold_high)), random_state=42)
    m = mid.sample(n=min(5, len(mid)), random_state=42) if len(mid) > 0 else mid
    l = low.sample(n=min(10, len(low)), random_state=42)
    return pd.concat([
        g.assign(gold_bucket="HIGH"),
        m.assign(gold_bucket="MID"),
        l.assign(gold_bucket="LOW"),
    ], ignore_index=True)


def score_one(mod, title, body):
    try:
        return mod.score_article(title, body)
    except Exception as e:
        logger.warning("score failed: %s", e)
        return None


def flatten(key: str, result: dict | None, row: dict):
    if result is None:
        row[f"{key}_decision"] = None
        row[f"{key}_max_method"] = None
        row[f"{key}_top_method"] = None
        row[f"{key}_high"] = 0
        return
    d_p1 = float(result.get("decision", {}).get("p1", 0))
    m_p1 = {k: float(result.get(k, {}).get("p1", 0))
            for k in DIMENSIONS if k != "decision"}
    max_m = max(m_p1.values()) if m_p1 else 0.0
    top_m = max(m_p1, key=m_p1.get) if m_p1 else None
    row[f"{key}_decision"] = d_p1
    row[f"{key}_max_method"] = max_m
    row[f"{key}_top_method"] = top_m
    row[f"{key}_high"] = int(d_p1 >= 0.75 and max_m >= 0.75)


def summarize(df: pd.DataFrame, key: str) -> tuple[int, int, int, int]:
    """Return (HIGH_tp, HIGH_total, LOW_tn, LOW_total)."""
    gh = df["gold_bucket"] == "HIGH"
    gl = df["gold_bucket"] == "LOW"
    tp = int(((df[f"{key}_high"] == 1) & gh).sum())
    tn = int(((df[f"{key}_high"] == 0) & gl).sum())
    return tp, int(gh.sum()), tn, int(gl.sum())


def main():
    test = pick_test_set()
    logger.info("Test set: %d HIGH, %d MID, %d LOW",
                (test["gold_bucket"] == "HIGH").sum(),
                (test["gold_bucket"] == "MID").sum(),
                (test["gold_bucket"] == "LOW").sum())

    # Load OLD scores
    old = pd.read_csv(OLD_RESULTS) if OLD_RESULTS.exists() else pd.DataFrame()
    old_by_idx = {int(r["idx"]): r for _, r in old.iterrows()} if len(old) else {}

    # Load new modules
    loaded = {}
    for key, mod_name, label in NEW_MODELS:
        try:
            loaded[key] = (importlib.import_module(mod_name), label)
            logger.info("Loaded %s", mod_name)
        except Exception as e:
            logger.error("FAILED to import %s: %s", mod_name, e)

    rows = []
    for i, r in test.iterrows():
        idx = int(r.name) if hasattr(r, "name") else i
        # test rows carry original df index because we didn't reset_index
        # Use the index from pick_test_set source frame:
        try:
            orig_idx = int(test.index[i])
        except Exception:
            orig_idx = i
        title = str(r.get("title", ""))
        body = str(r.get("body_excerpt", ""))
        gold_bucket = r.get("gold_bucket")
        gold_method = str(r.get("method_dimension", ""))
        row = {
            "idx": orig_idx,
            "title": title[:80],
            "gold_bucket": gold_bucket,
            "gold_method": gold_method,
            "gold_label": float(r.get("label", 0)),
        }
        # Carry forward old scores if present
        if orig_idx in old_by_idx:
            o = old_by_idx[orig_idx]
            for col in o.index:
                if col in ("idx", "title", "gold_bucket", "gold_method", "gold_label"):
                    continue
                row[col] = o[col]

        for key, (mod, label) in loaded.items():
            logger.info("[%s] scoring %d (%s)", key, orig_idx, gold_bucket)
            t0 = time.time()
            result = score_one(mod, title, body)
            flatten(key, result, row)
            row[f"{key}_time_s"] = round(time.time() - t0, 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    logger.info("Results → %s", OUT)

    # Print summary lines per the spec
    lines = []

    # ── Experiment A: M1-v4 ──
    if "m1_old_high" in df.columns:
        tp_old, n_h, tn_old, n_l = summarize(df, "m1_old")
        lines.append(f"M1-old: HIGH={tp_old}/{n_h} ({tp_old/n_h:.0%}), LOW={tn_old}/{n_l} ({tn_old/n_l:.0%})")
    lines.append("M1-v3:  HIGH=3/15 (rejected)")
    if "m1_v4_high" in df.columns:
        tp, n_h, tn, n_l = summarize(df, "m1_v4")
        lines.append(f"M1-v4:  HIGH={tp}/{n_h} ({tp/n_h:.0%}), LOW={tn}/{n_l} ({tn/n_l:.0%})")

    # ── Experiment B: M1-v5 ──
    if "m1_v5_high" in df.columns:
        tp, n_h, tn, n_l = summarize(df, "m1_v5")
        lines.append(f"M1-v5:  HIGH={tp}/{n_h} ({tp/n_h:.0%}), LOW={tn}/{n_l} ({tn/n_l:.0%})")

    # ── Experiment C: M3-v4 ──
    def agreement(a: str, b: str) -> float:
        if f"{a}_high" not in df.columns or f"{b}_high" not in df.columns:
            return float("nan")
        return float((df[f"{a}_high"] == df[f"{b}_high"]).mean())

    if "m3_old_high" in df.columns:
        tp, n_h, _, _ = summarize(df, "m3_old")
        ag = agreement("m3_old", "m4_old") * 100
        lines.append(f"M3-old: HIGH={tp}/{n_h} ({tp/n_h:.0%}), M3-M4 agreement={ag:.0f}%")
    lines.append("M3-v3:  HIGH=1/15 (rejected)")
    if "m3_v4_high" in df.columns:
        tp, n_h, tn, n_l = summarize(df, "m3_v4")
        # Compare v4 against best M4 (m4_v3 recommended) OR m4_old
        m4key = "m4_v3" if "m4_v3_high" in df.columns else "m4_old"
        ag = agreement("m3_v4", m4key) * 100
        lines.append(f"M3-v4:  HIGH={tp}/{n_h} ({tp/n_h:.0%}), LOW={tn}/{n_l} ({tn/n_l:.0%}), M3-M4 agreement={ag:.0f}%")

    print("\n=== Experiment results ===")
    for line in lines:
        print(line)

    # Summary table
    def _pct(a, b): return f"{a/b:.0%}" if b else "-"
    summary = []

    def row(name, key, verdict_fn=None):
        if f"{key}_high" not in df.columns:
            return
        tp, nh, tn, nl = summarize(df, key)
        summary.append({
            "Model": name,
            "HIGH recall": f"{tp}/{nh} ({_pct(tp,nh)})",
            "LOW spec": f"{tn}/{nl} ({_pct(tn,nl)})",
            "tp": tp, "nh": nh, "tn": tn, "nl": nl,
        })

    row("M1-old", "m1_old")
    row("M1-v4",  "m1_v4")
    row("M1-v5",  "m1_v5")
    row("M3-old", "m3_old")
    row("M3-v4",  "m3_v4")

    print("\n=== Summary table ===")
    print(f"| {'Model':<8} | {'HIGH recall':<14} | {'LOW spec':<14} | {'Verdict':<20} |")
    print(f"| {'-'*8} | {'-'*14} | {'-'*14} | {'-'*20} |")

    # Baselines
    m1_old = next((s for s in summary if s["Model"] == "M1-old"), None)
    m3_old = next((s for s in summary if s["Model"] == "M3-old"), None)

    def verdict_m1(s):
        if not s or not m1_old:
            return "n/a"
        if s["Model"] == "M1-old":
            return "baseline"
        # Keep v4 ONLY if HIGH recall ≥ M1-old (60%) AND LOW spec ≥ 40%
        recall_ok = (s["tp"] / max(1, s["nh"])) >= (m1_old["tp"] / max(1, m1_old["nh"]))
        low_ok = (s["tn"] / max(1, s["nl"])) >= 0.40
        better = recall_ok and low_ok
        return "KEEP" if better else "REJECT"

    def verdict_m3(s):
        if not s or not m3_old:
            return "n/a"
        if s["Model"] == "M3-old":
            return "baseline"
        recall_ok = (s["tp"] / max(1, s["nh"])) > (m3_old["tp"] / max(1, m3_old["nh"]))
        # Agreement gate comes from summary above; approximate keep/reject here
        return "KEEP" if recall_ok else "REJECT"

    m1v4 = next((s for s in summary if s["Model"] == "M1-v4"), None)
    m1v5 = next((s for s in summary if s["Model"] == "M1-v5"), None)
    m3v4 = next((s for s in summary if s["Model"] == "M3-v4"), None)

    for s in summary:
        if s["Model"].startswith("M1"):
            v = verdict_m1(s)
        else:
            v = verdict_m3(s)
        print(f"| {s['Model']:<8} | {s['HIGH recall']:<14} | {s['LOW spec']:<14} | {v:<20} |")

    # Final recommendation
    keep_m1v4 = verdict_m1(m1v4) == "KEEP" if m1v4 else False
    keep_m1v5 = verdict_m1(m1v5) == "KEEP" if m1v5 else False
    keep_m3v4 = verdict_m3(m3v4) == "KEEP" if m3v4 else False

    winners = []
    if keep_m1v4:
        winners.append("M1-v4")
    if keep_m1v5:
        winners.append("M1-v5")
    if keep_m3v4:
        winners.append("M3-v4")

    if not winners:
        print("\nFINAL RECOMMENDED CONFIG: No improvement. Keep M1-old and M3-old.")
    else:
        print(f"\nFINAL RECOMMENDED CONFIG: {', '.join(winners)}  (plus M4-v3 + M5 from prior round)")


if __name__ == "__main__":
    main()
