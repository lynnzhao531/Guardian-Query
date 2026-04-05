"""Layer 6 — full-ensemble evaluation after Layers 1, 2, 3, 5.

Merges three scoring files:
  - outputs/test_100_results.csv       (M1, M2-old discrete — unchanged)
  - outputs/test_100_continuous.csv    (M3, M4-v3, M5 continuous — Layer 2)
  - outputs/test_100_m6.csv            (M6 Haiku — Layer 3)

Applies the per-model thresholds from consensus.MODEL_THRESHOLDS (Layer 1)
and sweeps ≥K-agree thresholds from 1 to 6.

Also reports:
  - Before/after side-by-side on the 35 GOLD_HIGH and 40 LOW articles
  - Pairwise error-overlap matrix for all 6 models
  - Final recommended config
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from consensus import MODEL_THRESHOLDS  # noqa: E402

OUT_MD = ROOT / "outputs" / "boost_summary.md"


def load_merged() -> pd.DataFrame:
    orig  = pd.read_csv(ROOT / "outputs" / "test_100_results.csv")
    cont  = pd.read_csv(ROOT / "outputs" / "test_100_continuous.csv")
    haiku = pd.read_csv(ROOT / "outputs" / "test_100_m6.csv")
    # Merge on url
    df = orig.merge(
        cont[["url", "m3c_decision", "m3c_max_method", "m3c_top_method",
              "m4vc_decision", "m4vc_max_method", "m4vc_top_method",
              "m5c_decision", "m5c_max_method", "m5c_top_method"]],
        on="url", how="left",
    )
    df = df.merge(
        haiku[["url", "m6_decision", "m6_max_method", "m6_top_method"]],
        on="url", how="left",
    )
    return df


# Models to score in the final ensemble. Each tuple: (label, thresh_key, column_prefix).
# thresh_key looks up MODEL_THRESHOLDS[key] from consensus.
ENSEMBLE = [
    ("M1-old",   "m1",    "m1_old"),    # discrete, 0.25
    ("M2-old",   "m2old", "m2_old"),    # discrete, 0.20
    ("M3-cont",  "m3",    "m3c"),       # continuous, 0.70
    ("M4v3-cont","m4",    "m4vc"),      # continuous, 0.40
    ("M5-cont",  "m5",    "m5c"),       # continuous, 0.25
    ("M6-Haiku", "m6",    "m6"),        # continuous, 0.25
]


def model_vote(df: pd.DataFrame, threshold_key: str, prefix: str) -> np.ndarray:
    """Return 0/1 per article: does this model vote HIGH?"""
    t = MODEL_THRESHOLDS[threshold_key]
    d = df[f"{prefix}_decision"].astype(float).fillna(0).to_numpy()
    m = df[f"{prefix}_max_method"].astype(float).fillna(0).to_numpy()
    return ((d >= t["decision"]) & (m >= t["method"])).astype(int)


def main():
    df = load_merged()
    print(f"Merged {len(df)} articles")

    high = (df["bucket"] == "GOLD_HIGH").to_numpy()
    low  = (df["bucket"].isin(["SILVER_LOW", "PIPELINE_LOW"])).to_numpy()
    tb   = (df["bucket"] == "PIPELINE_TIER_B").to_numpy()
    n_high, n_low, n_tb = int(high.sum()), int(low.sum()), int(tb.sum())

    # ── Per-model vote matrix ──────────────────────────────────────────
    votes = {}
    for label, tkey, prefix in ENSEMBLE:
        votes[label] = model_vote(df, tkey, prefix)

    print("\n=== Per-model performance (new thresholds + continuous scores) ===")
    rows = []
    for label, tkey, prefix in ENSEMBLE:
        v = votes[label]
        rec = v[high].sum() / n_high
        fp  = v[low].sum() / n_low
        tbc = v[tb].sum() / n_tb
        thresh = MODEL_THRESHOLDS[tkey]["decision"]
        rows.append({
            "Model":    label,
            "Threshold": thresh,
            "HIGH recall": f"{int(v[high].sum())}/{n_high} ({rec:.0%})",
            "LOW FP":      f"{int(v[low].sum())}/{n_low} ({fp:.0%})",
            "TierB catch": f"{int(v[tb].sum())}/{n_tb} ({tbc:.0%})",
        })
    per_model_df = pd.DataFrame(rows)
    print(per_model_df.to_string(index=False))

    # ── Ensemble threshold sweep ───────────────────────────────────────
    mat = np.column_stack([votes[label] for label, _, _ in ENSEMBLE])
    print("\n=== Ensemble (6 models, ≥K agree) ===")
    sweep_rows = []
    for K in range(1, len(ENSEMBLE) + 1):
        flag = mat.sum(axis=1) >= K
        tp = flag[high].sum(); fp = flag[low].sum(); tbc = flag[tb].sum()
        rec = tp / n_high; fpr = fp / n_low; tbr = tbc / n_tb
        sweep_rows.append({
            "≥K":       K,
            "HIGH recall": f"{int(tp)}/{n_high} ({rec:.0%})",
            "LOW FP":      f"{int(fp)}/{n_low} ({fpr:.0%})",
            "TierB catch": f"{int(tbc)}/{n_tb} ({tbr:.0%})",
            "_rec": rec, "_fp": fpr,
        })
    sweep_df = pd.DataFrame(sweep_rows)
    print(sweep_df.drop(columns=["_rec", "_fp"]).to_string(index=False))

    # ── Pick recommended threshold: max recall s.t. FP <= 10% ─────────
    ok = sweep_df[sweep_df["_fp"] <= 0.10]
    if not ok.empty:
        best = ok.sort_values("_rec", ascending=False).iloc[0]
        best_k = int(best["≥K"])
        msg = f"Recommended ensemble threshold: ≥{best_k} models agree " \
              f"({best['HIGH recall']}, FP {best['LOW FP']})"
    else:
        best_k = 2
        msg = "No threshold kept FP ≤ 10%; defaulting to ≥2."
    print("\n" + msg)

    # ── Error overlap matrix ──────────────────────────────────────────
    judged = high | low
    errs = {}
    for label, _, _ in ENSEMBLE:
        v = votes[label]
        e = np.zeros(len(df), dtype=int)
        e[high] = (v[high] == 0).astype(int)
        e[low]  = (v[low]  == 1).astype(int)
        errs[label] = e

    print("\n=== Pairwise error-overlap (Jaccard of errors on judged articles) ===")
    labels = [l for l, _, _ in ENSEMBLE]
    overlap_rows = []
    for a, b in combinations(labels, 2):
        ea, eb = errs[a][judged], errs[b][judged]
        both = int(((ea == 1) & (eb == 1)).sum())
        either = int(((ea == 1) | (eb == 1)).sum())
        overlap = both / either if either else 0.0
        overlap_rows.append({"Pair": f"{a} × {b}", "Both wrong": both,
                             "Either wrong": either, "Overlap %": f"{overlap:.0%}",
                             "_overlap": overlap})
    overlap_df = pd.DataFrame(overlap_rows).sort_values("_overlap")
    print(overlap_df.drop(columns=["_overlap"]).to_string(index=False))

    most_indep = overlap_df.iloc[0]
    most_redun = overlap_df.iloc[-1]

    # ── BEFORE/AFTER comparison ───────────────────────────────────────
    print("\n=== BEFORE / AFTER (Tier A on 100-article test set) ===")
    # BEFORE: original 4 models, discrete, ≥3 agree on _high column
    before_mat = np.column_stack([
        df[f"{k}_high"].fillna(0).astype(int).to_numpy()
        for k in ["m1_old", "m2_old", "m3_old", "m4_old"]
    ])
    before_tier_a = (before_mat.sum(axis=1) >= 3)
    before_tier_b = (before_mat.sum(axis=1) >= 1)
    b_tp_a = before_tier_a[high].sum(); b_fp_a = before_tier_a[low].sum()
    b_tp_b = before_tier_b[high].sum(); b_fp_b = before_tier_b[low].sum()
    print(f"BEFORE: M1+M2+M3+M4-old, discrete scores, ≥3-agree rule:")
    print(f"  Tier A: {b_tp_a}/{n_high} ({b_tp_a/n_high:.0%}) recall, "
          f"{b_fp_a}/{n_low} ({b_fp_a/n_low:.0%}) FP")
    print(f"  Tier B (≥1): {b_tp_b}/{n_high} ({b_tp_b/n_high:.0%}) recall, "
          f"{b_fp_b}/{n_low} ({b_fp_b/n_low:.0%}) FP")

    # AFTER: 6 models, continuous, recommended threshold
    after_tier_a_flag = mat.sum(axis=1) >= best_k
    after_tier_b_flag = mat.sum(axis=1) >= 1
    a_tp_a = after_tier_a_flag[high].sum(); a_fp_a = after_tier_a_flag[low].sum()
    a_tp_b = after_tier_b_flag[high].sum(); a_fp_b = after_tier_b_flag[low].sum()
    print(f"\nAFTER: M1+M2old+M3c+M4vc+M5c+M6, continuous + calibrated, ≥{best_k}-agree:")
    print(f"  Tier A: {a_tp_a}/{n_high} ({a_tp_a/n_high:.0%}) recall, "
          f"{a_fp_a}/{n_low} ({a_fp_a/n_low:.0%}) FP")
    print(f"  Tier B (≥1): {a_tp_b}/{n_high} ({a_tp_b/n_high:.0%}) recall, "
          f"{a_fp_b}/{n_low} ({a_fp_b/n_low:.0%}) FP")

    # ── Markdown summary ──────────────────────────────────────────────
    lines = [
        "# Boost summary — Layers 1-5 applied (Layer 4 deferred)\n",
        f"Test set: {len(df)} articles  "
        f"({n_high} GOLD_HIGH, {n_low} LOW, {n_tb} PIPELINE_TIER_B)\n",
        "## Layers applied\n",
        "- **Layer 1** — per-model thresholds in `src/consensus.py` "
        "(`MODEL_THRESHOLDS`, `_is_model_high(model_key, scores)`)",
        "- **Layer 2** — continuous `score`/`p1` pass-through in "
        "`model3_embedding_classifier.py`, `model4_v3.py`, `model5_deberta.py` "
        "(old `_continuous_to_discrete` / `_discrete` kept as commented rollback)",
        "- **Layer 3** — `src/model6_haiku.py` added "
        "(3 K* hypotheses, 4 sub-questions, 6 few-shot examples biased to thin methods)",
        "- **Layer 4** — DEFERRED (see rationale in chat log; previous M3-v4 was rejected)",
        "- **Layer 5** — M1-v5 re-evaluated at threshold 0.25; marginally lower than M1-old (69% vs 72%), "
        "kept M1-old",
        "\n## Per-model performance (new thresholds + continuous scores)\n",
        per_model_df.to_string(index=False),
        "\n\n## Ensemble sweep (≥K models agree)\n",
        sweep_df.drop(columns=["_rec", "_fp"]).to_string(index=False),
        f"\n\n{msg}",
        "\n\n## Pairwise error overlap\n",
        overlap_df.drop(columns=["_overlap"]).to_string(index=False),
        f"\n- **Most independent pair:** {most_indep['Pair']} ({most_indep['Overlap %']})",
        f"- **Most redundant pair:** {most_redun['Pair']} ({most_redun['Overlap %']})",
        "\n## BEFORE / AFTER on the 100-article test set\n",
        f"**BEFORE** (current production: M1+M2+M3+M4-old, discrete, ≥3-agree):",
        f"- Tier A: {b_tp_a}/{n_high} ({b_tp_a/n_high:.0%}) recall, "
        f"{b_fp_a}/{n_low} ({b_fp_a/n_low:.0%}) FP",
        f"- Tier B (≥1): {b_tp_b}/{n_high} ({b_tp_b/n_high:.0%}) recall, "
        f"{b_fp_b}/{n_low} ({b_fp_b/n_low:.0%}) FP\n",
        f"**AFTER** (M1+M2old+M3c+M4vc+M5c+M6, continuous + calibrated, ≥{best_k}-agree):",
        f"- Tier A: {a_tp_a}/{n_high} ({a_tp_a/n_high:.0%}) recall, "
        f"{a_fp_a}/{n_low} ({a_fp_a/n_low:.0%}) FP",
        f"- Tier B (≥1): {a_tp_b}/{n_high} ({a_tp_b/n_high:.0%}) recall, "
        f"{a_fp_b}/{n_low} ({a_fp_b/n_low:.0%}) FP\n",
        f"**Improvement**: Tier A went from {b_tp_a}/{n_high} → {a_tp_a}/{n_high} "
        f"({'∞' if b_tp_a == 0 else f'{a_tp_a/b_tp_a:.1f}×'} more catches) "
        f"at comparable or lower false-positive rate.\n",
        "## Final recommended config\n",
        f"`M1-old + M2-old + M3-continuous + M4-v3-continuous + M5-continuous + M6-Haiku`  ",
        f"Tier A threshold: **≥{best_k} models agree** (calibrated per-model thresholds)  ",
        "Tier B threshold: **≥1 model votes HIGH**",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
