"""Combined-model performance on outputs/test_100_results.csv.

Runs four analyses:
  1. Consensus simulation across five model combinations
  2. Per-model uniquely-caught GOLD_HIGH articles
  3. Threshold sweep (1..N models agree)
  4. Pairwise error-independence check

Does not touch any model file — only reads test_100_results.csv.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "outputs" / "test_100_results.csv"
OUT_MD = ROOT / "outputs" / "test_100_combined_summary.md"


# ─────────────────────────────────────────────────────────────────────────────
def load() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    return df


def hits(df: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Return a (n_articles × n_models) 0/1 matrix of HIGH votes."""
    arr = np.column_stack(
        [df[f"{k}_high"].fillna(0).astype(int).to_numpy() for k in keys]
    )
    return arr


def tier_counts(mat: np.ndarray, k: int) -> np.ndarray:
    """Per-article: True if ≥k models said HIGH."""
    return mat.sum(axis=1) >= k


def analysis_1(df: pd.DataFrame) -> pd.DataFrame:
    """Consensus simulation across configs."""
    high_mask = (df["bucket"] == "GOLD_HIGH").to_numpy()
    low_mask = ((df["bucket"] == "GOLD_LOW") | (df["bucket"] == "SILVER_LOW")
                | (df["bucket"] == "PIPELINE_LOW")).to_numpy()
    tb_mask = (df["bucket"] == "PIPELINE_TIER_B").to_numpy()
    n_high = int(high_mask.sum())
    n_low = int(low_mask.sum())
    n_tb = int(tb_mask.sum())

    configs = [
        ("A", "M1-old + M2-old + M3-old + M4-old",
         ["m1_old", "m2_old", "m3_old", "m4_old"]),
        ("B", "M1-old + M2-old + M3-old + M4-v3",
         ["m1_old", "m2_old", "m3_old", "m4_v3"]),
        ("C", "M1-old + M2-old + M3-old + M4-v3 + M5",
         ["m1_old", "m2_old", "m3_old", "m4_v3", "m5"]),
        ("D", "M1-old + M2-old + M4-v3 + M5",
         ["m1_old", "m2_old", "m4_v3", "m5"]),
        ("E", "M1-old + M4-v3 + M5",
         ["m1_old", "m4_v3", "m5"]),
    ]

    rows = []
    for tag, label, keys in configs:
        mat = hits(df, keys)
        tier_a = tier_counts(mat, 3) if len(keys) >= 3 else tier_counts(mat, len(keys))
        tier_b = tier_counts(mat, 1)
        ta_recall = tier_a[high_mask].mean()
        ta_fp     = tier_a[low_mask].mean()
        tb_recall = tier_b[high_mask].mean()
        tb_fp     = tier_b[low_mask].mean()
        tb_catch_a = tier_a[tb_mask].sum()
        tb_catch_b = tier_b[tb_mask].sum()
        rows.append({
            "Config": tag,
            "Models": label,
            "TierA recall": f"{int(tier_a[high_mask].sum())}/{n_high} ({ta_recall:.0%})",
            "TierA FP":     f"{int(tier_a[low_mask].sum())}/{n_low} ({ta_fp:.0%})",
            "TierB recall": f"{int(tier_b[high_mask].sum())}/{n_high} ({tb_recall:.0%})",
            "TierB FP":     f"{int(tier_b[low_mask].sum())}/{n_low} ({tb_fp:.0%})",
            "TierB-set → A/B": f"{tb_catch_a}/{tb_catch_b} of {n_tb}",
        })
    return pd.DataFrame(rows)


def analysis_2(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Which GOLD_HIGH articles are uniquely caught by each model?"""
    high_df = df[df["bucket"] == "GOLD_HIGH"].reset_index(drop=True)
    keys = ["m1_old", "m2_old", "m3_old", "m4_v3", "m5"]
    mat = hits(high_df, keys)  # (n_high × 5)

    # Unique: hit by exactly this model and no other (among these 5)
    unique_counts: dict[str, int] = {}
    for i, k in enumerate(keys):
        mask = (mat[:, i] == 1) & (mat.sum(axis=1) == 1)
        unique_counts[k] = int(mask.sum())

    n_consensus = int((mat.sum(axis=1) >= 3).sum())
    n_zero = int((mat.sum(axis=1) == 0).sum())

    summary = {
        "total_GOLD_HIGH": len(high_df),
        "unique_per_model": unique_counts,
        "caught_by_3plus": n_consensus,
        "caught_by_zero": n_zero,
    }

    missed = high_df[mat.sum(axis=1) == 0][
        ["title", "method_hint", "url"]
    ].reset_index(drop=True)
    return summary, missed


def analysis_3(df: pd.DataFrame) -> pd.DataFrame:
    """Threshold sweep on the recommended 5-model set."""
    keys = ["m1_old", "m2_old", "m3_old", "m4_v3", "m5"]
    mat = hits(df, keys)
    high_mask = (df["bucket"] == "GOLD_HIGH").to_numpy()
    low_mask = ((df["bucket"] == "GOLD_LOW") | (df["bucket"] == "SILVER_LOW")
                | (df["bucket"] == "PIPELINE_LOW")).to_numpy()
    rows = []
    for k in range(1, len(keys) + 1):
        flag = tier_counts(mat, k)
        recall = flag[high_mask].mean()
        fp     = flag[low_mask].mean()
        rows.append({
            "threshold ≥": k,
            "HIGH recall":  f"{int(flag[high_mask].sum())}/{int(high_mask.sum())} ({recall:.0%})",
            "LOW false-pos": f"{int(flag[low_mask].sum())}/{int(low_mask.sum())} ({fp:.0%})",
            "recall_num":    recall,
            "fp_num":        fp,
        })
    return pd.DataFrame(rows)


def analysis_4(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple, tuple]:
    """Pairwise error-overlap on the recommended 4-model set.

    Error on article = (HIGH article missed) OR (LOW article false-alarmed).
    """
    keys = ["m1_old", "m2_old", "m4_v3", "m5"]
    labels = {"m1_old": "M1-old", "m2_old": "M2-old",
              "m4_v3": "M4-v3", "m5": "M5"}
    high_mask = (df["bucket"] == "GOLD_HIGH").to_numpy()
    low_mask = ((df["bucket"] == "GOLD_LOW") | (df["bucket"] == "SILVER_LOW")
                | (df["bucket"] == "PIPELINE_LOW")).to_numpy()

    # error[i] = 1 iff model got article i wrong (for any bucket we can judge)
    def err(k: str) -> np.ndarray:
        h = df[f"{k}_high"].fillna(0).astype(int).to_numpy()
        e = np.zeros(len(df), dtype=int)
        e[high_mask] = (h[high_mask] == 0).astype(int)   # miss
        e[low_mask]  = (h[low_mask]  == 1).astype(int)   # false alarm
        return e

    errs = {k: err(k) for k in keys}
    judged_mask = high_mask | low_mask
    rows = []
    overlaps: list[tuple[str, str, float]] = []
    for a, b in combinations(keys, 2):
        ea, eb = errs[a][judged_mask], errs[b][judged_mask]
        both = int(((ea == 1) & (eb == 1)).sum())
        either = int(((ea == 1) | (eb == 1)).sum())
        only_a = int(((ea == 1) & (eb == 0)).sum())
        only_b = int(((ea == 0) & (eb == 1)).sum())
        overlap_pct = (both / either) if either else 0.0
        overlaps.append((labels[a], labels[b], overlap_pct))
        rows.append({
            "Pair":        f"{labels[a]} × {labels[b]}",
            "Both wrong":  both,
            "Only A wrong": only_a,
            "Only B wrong": only_b,
            "Overlap %":   f"{overlap_pct:.0%}",
        })
    tbl = pd.DataFrame(rows)
    most_indep = min(overlaps, key=lambda t: t[2])
    most_redun = max(overlaps, key=lambda t: t[2])
    return tbl, most_indep, most_redun


def main():
    df = load()
    lines: list[str] = []
    lines.append("# Combined model performance — test_100\n")
    lines.append(
        f"Test set: {len(df)} articles "
        f"({(df['bucket']=='GOLD_HIGH').sum()} GOLD_HIGH, "
        f"{(df['bucket']=='SILVER_LOW').sum()} SILVER_LOW, "
        f"{(df['bucket']=='PIPELINE_TIER_B').sum()} Tier B, "
        f"{(df['bucket']=='PIPELINE_LOW').sum()} pipeline LOW, "
        f"{(df['bucket']=='SILVER_MID').sum()} SILVER_MID)\n"
    )
    lines.append("HIGH vote = `_high == 1` (decision_p1 ≥ 0.75 AND max_method_p1 ≥ 0.75).\n")
    lines.append("LOW pool for FP calc = SILVER_LOW ∪ PIPELINE_LOW. SILVER_MID is ignored "
                 "because its label is intentionally ambiguous.\n")

    # ── ANALYSIS 1 ────────────────────────────────────────────────────
    lines.append("\n## Analysis 1 — consensus simulation\n")
    tbl1 = analysis_1(df)
    print("\n=== Analysis 1: consensus simulation ===")
    print(tbl1.to_string(index=False))
    lines.append(tbl1.to_string(index=False))
    lines.append("")

    # ── ANALYSIS 2 ────────────────────────────────────────────────────
    summary, missed = analysis_2(df)
    lines.append("\n## Analysis 2 — which articles does each model uniquely catch?\n")
    print("\n=== Analysis 2: per-model unique catches (on GOLD_HIGH) ===")
    for k, v in summary["unique_per_model"].items():
        msg = f"  {k:<10} uniquely catches {v}"
        print(msg)
        lines.append(f"- `{k}` uniquely catches **{v}** of {summary['total_GOLD_HIGH']}")
    print(f"  caught by ≥3 models (consensus): {summary['caught_by_3plus']}")
    print(f"  caught by 0 models (all miss):  {summary['caught_by_zero']}")
    lines.append(f"- caught by **≥3 models** (consensus): {summary['caught_by_3plus']}")
    lines.append(f"- caught by **0 models** (all miss):  {summary['caught_by_zero']}")

    print(f"\nArticles caught by 0 models: {len(missed)} — these are:")
    lines.append(f"\n### Articles caught by 0 models ({len(missed)})\n")
    for _, r in missed.iterrows():
        msg = f"  • [{r['method_hint']}] {r['title']}"
        print(msg)
        lines.append(f"- **[{r['method_hint']}]** {r['title']}  \n  <{r['url']}>")

    # ── ANALYSIS 3 ────────────────────────────────────────────────────
    lines.append("\n## Analysis 3 — threshold sweep (5-model set)\n")
    tbl3 = analysis_3(df)
    print("\n=== Analysis 3: threshold sweep ===")
    print(tbl3[["threshold ≥", "HIGH recall", "LOW false-pos"]].to_string(index=False))
    lines.append(tbl3[["threshold ≥", "HIGH recall", "LOW false-pos"]].to_string(index=False))

    # Best threshold: max recall s.t. FP < 0.10
    ok = tbl3[tbl3["fp_num"] < 0.10]
    if not ok.empty:
        best = ok.sort_values("recall_num", ascending=False).iloc[0]
        best_k = int(best["threshold ≥"])
        msg = (f"\nThreshold that maximises HIGH recall subject to FP < 10%: "
               f"≥{best_k} (recall={best['recall_num']:.0%}, FP={best['fp_num']:.0%})")
    else:
        best_k = 3
        msg = "\nNo threshold kept FP < 10%; defaulting to ≥3."
    print(msg)
    lines.append(msg)

    # ── ANALYSIS 4 ────────────────────────────────────────────────────
    lines.append("\n## Analysis 4 — pairwise error-overlap (M1, M2-old, M4-v3, M5)\n")
    tbl4, indep, redun = analysis_4(df)
    print("\n=== Analysis 4: pairwise error overlap ===")
    print(tbl4.to_string(index=False))
    lines.append(tbl4.to_string(index=False))

    print(f"\nMost independent pair: {indep[0]} × {indep[1]} ({indep[2]:.0%} error overlap)")
    print(f"Most redundant pair:   {redun[0]} × {redun[1]} ({redun[2]:.0%} error overlap)")
    lines.append(f"\n- **Most independent pair:** {indep[0]} × {indep[1]} ({indep[2]:.0%} overlap)")
    lines.append(f"- **Most redundant pair:**   {redun[0]} × {redun[1]} ({redun[2]:.0%} overlap)")

    if redun[2] > 0.70:
        warn = (f"WARNING: {redun[0]} × {redun[1]} are largely redundant "
                f"({redun[2]:.0%} error overlap). Consider dropping one "
                f"and lowering the consensus threshold.")
        print(warn)
        lines.append(f"\n> {warn}")

    # ── Final recommendation ─────────────────────────────────────────
    print(f"\nFINAL RECOMMENDED CONFIG: M1-old + M2-old + M4-v3 + M5 with threshold ≥{best_k}")
    lines.append(f"\n## Final recommendation\n")
    lines.append(f"**Config:** M1-old + M2-old + M4-v3 + M5  \n"
                 f"**Tier A threshold:** ≥{best_k} models agree  \n"
                 f"**Tier B threshold:** ≥1 model says HIGH")

    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote → {OUT_MD}")


if __name__ == "__main__":
    main()
