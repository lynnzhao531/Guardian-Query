"""Analyze outputs/test_100_results.csv:
  • HIGH recall & LOW specificity with 95% bootstrap CIs
  • Tier B catch rate, Pipeline LOW reject rate
  • McNemar's test old-vs-new for M1 and M3
  • Per-method HIGH recall stratification
  • FINAL verdicts

Writes: outputs/test_100_summary.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "outputs" / "test_100_results.csv"
OUT_MD = ROOT / "outputs" / "test_100_summary.md"

MODELS = [
    ("m1_old", "M1-old"),
    ("m1_v4",  "M1-v4"),
    ("m1_v5",  "M1-v5"),
    ("m2_old", "M2-old"),
    ("m3_old", "M3-old"),
    ("m3_v4",  "M3-v4"),
    ("m4_old", "M4-old"),
    ("m4_v3",  "M4-v3"),
    ("m5",     "M5"),
]

METHODS = ["rct", "prepost", "case_study", "expert_qual",
           "expert_secondary", "gut"]


def bootstrap_ci(hits: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Return (mean, lo_2.5%, hi_97.5%) of the proportion of hits, via bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(hits)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(hits, size=n, replace=True)
        means[i] = sample.mean()
    return float(hits.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar(a_hits: np.ndarray, b_hits: np.ndarray) -> float:
    """Two-sided McNemar exact p-value (binomial on discordant pairs)."""
    # a and b are paired 0/1 arrays
    b_only = int(((a_hits == 0) & (b_hits == 1)).sum())  # new correct, old wrong
    a_only = int(((a_hits == 1) & (b_hits == 0)).sum())  # old correct, new wrong
    n = b_only + a_only
    if n == 0:
        return 1.0
    # Exact two-sided binomial with p=0.5
    from math import comb
    k = min(b_only, a_only)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return p


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    if np.isnan(mean):
        return "n/a"
    return f"{mean:.0%} [{lo:.0%}–{hi:.0%}]"


def main():
    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} test rows")

    high_mask = df["bucket"] == "GOLD_HIGH"
    low_mask = df["bucket"] == "SILVER_LOW"
    tb_mask = df["bucket"] == "PIPELINE_TIER_B"
    plow_mask = df["bucket"] == "PIPELINE_LOW"

    n_high = int(high_mask.sum())
    n_low = int(low_mask.sum())
    n_tb = int(tb_mask.sum())
    n_plow = int(plow_mask.sum())
    print(f"Buckets: HIGH={n_high} LOW={n_low} TierB={n_tb} PipelineLOW={n_plow}")

    lines: list[str] = []
    lines.append("# Test 100 — stratified evaluation\n")
    lines.append(
        f"Test set: {len(df)} articles "
        f"(GOLD_HIGH={n_high}, SILVER_MID={(df['bucket']=='SILVER_MID').sum()}, "
        f"SILVER_LOW={n_low}, PIPELINE_TIER_B={n_tb}, PIPELINE_LOW={n_plow})\n"
    )
    lines.append("Scoring threshold: `high = decision_p1 ≥ 0.75 AND max_method_p1 ≥ 0.75`\n")

    # ── Main metrics table ────────────────────────────────────────────
    rows = []
    cache = {}  # for later McNemar / verdicts
    for key, label in MODELS:
        col = f"{key}_high"
        if col not in df.columns:
            continue
        arr = df[col].fillna(0).astype(int).to_numpy()
        cache[key] = arr
        high_hits = arr[high_mask.to_numpy()]
        low_miss = 1 - arr[low_mask.to_numpy()]  # specificity
        tb_flag = arr[tb_mask.to_numpy()]
        plow_rej = 1 - arr[plow_mask.to_numpy()]

        hr = bootstrap_ci(high_hits)
        ls = bootstrap_ci(low_miss)
        tb_rate = tb_flag.mean() if len(tb_flag) else float("nan")
        plow_rate = plow_rej.mean() if len(plow_rej) else float("nan")

        rows.append({
            "Model": label,
            "HIGH recall [95% CI]": fmt_ci(*hr),
            "LOW spec [95% CI]": fmt_ci(*ls),
            "Tier B flag rate": f"{int(tb_flag.sum())}/{len(tb_flag)} ({tb_rate:.0%})",
            "Pipeline LOW reject": f"{int(plow_rej.sum())}/{len(plow_rej)} ({plow_rate:.0%})",
        })

    metrics_df = pd.DataFrame(rows)
    lines.append("## Main metrics\n")
    lines.append(metrics_df.to_string(index=False))
    lines.append("\n")

    print("\n=== Main metrics ===")
    print(metrics_df.to_string(index=False))

    # ── McNemar tests ─────────────────────────────────────────────────
    # Use paired HIGH hits on GOLD_HIGH + LOW miss on SILVER_LOW combined as "correct"
    # Actually McNemar needs paired binary — use: correct = (HIGH_hit on HIGH rows) OR (LOW_reject on LOW rows)
    # Simpler and closest to spec: McNemar on GOLD_HIGH recall.
    hm = high_mask.to_numpy()
    lm = low_mask.to_numpy()

    comparisons = [
        ("m1_old", "m1_v4"),
        ("m1_old", "m1_v5"),
        ("m3_old", "m3_v4"),
    ]
    mcn_results = {}
    lines.append("\n## McNemar's tests (paired, on GOLD_HIGH articles)\n")
    print("\n=== McNemar (on GOLD_HIGH) ===")
    for a, b in comparisons:
        if a not in cache or b not in cache:
            continue
        p = mcnemar(cache[a][hm], cache[b][hm])
        mcn_results[(a, b)] = p
        sig = "significant" if p < 0.05 else "not significant"
        msg = f"{a} vs {b}: p={p:.3f} — {sig}"
        lines.append(f"- {msg}")
        print(msg)

    # ── Stratified HIGH recall by method ──────────────────────────────
    gold_df = df[high_mask].reset_index(drop=True)
    strat_rows = []
    for key, label in MODELS:
        if f"{key}_high" not in df.columns:
            continue
        row = {"Model": label}
        for m in METHODS:
            sub = gold_df[gold_df["method_hint"] == m]
            if len(sub) == 0:
                row[m] = "—"
            else:
                hits = sub[f"{key}_high"].fillna(0).astype(int).sum()
                row[m] = f"{int(hits)}/{len(sub)}"
        strat_rows.append(row)
    strat_df = pd.DataFrame(strat_rows)
    lines.append("\n## Stratified HIGH recall by method\n")
    lines.append(strat_df.to_string(index=False))
    print("\n=== Stratified HIGH recall by method ===")
    print(strat_df.to_string(index=False))

    # ── Tier B catch-rate row ─────────────────────────────────────────
    lines.append("\n## Tier B catch rate (most informative: real borderline cases)\n")
    tb_row = {label: f"{int(cache[key][tb_mask.to_numpy()].sum())}/{n_tb}"
              for key, label in MODELS if key in cache}
    lines.append(str(tb_row))
    print("\n=== Tier B catch rate ===")
    print(tb_row)

    # ── Verdicts ─────────────────────────────────────────────────────
    lines.append("\n## Final verdicts\n")
    print("\n=== Final verdicts ===")

    def verdict(old: str, new: str) -> str:
        if old not in cache or new not in cache:
            return "skip (missing data)"
        p = mcn_results.get((old, new), float("nan"))
        old_hr = cache[old][hm].mean()
        new_hr = cache[new][hm].mean()
        old_ls = (1 - cache[old][lm]).mean()  # spec-"miss" — recompute as specificity
        old_ls = 1 - (cache[old][lm].mean())
        new_ls = 1 - (cache[new][lm].mean())
        old_tb = cache[old][tb_mask.to_numpy()].mean()
        new_tb = cache[new][tb_mask.to_numpy()].mean()

        # (a) significantly better
        if not np.isnan(p) and p < 0.05 and new_hr > old_hr and new_ls >= old_ls:
            return f"APPROVE (p={p:.3f}, HIGH {old_hr:.0%}→{new_hr:.0%})"
        # (b) significantly worse
        if not np.isnan(p) and p < 0.05 and new_hr < old_hr:
            return f"REJECT (p={p:.3f}, HIGH {old_hr:.0%}→{new_hr:.0%})"
        # (d) special: worse on HIGH but better on Tier B
        if new_hr < old_hr and new_tb > old_tb:
            return (f"FLAG (p={p:.3f}, HIGH {old_hr:.0%}→{new_hr:.0%} "
                    f"but Tier B {old_tb:.0%}→{new_tb:.0%})")
        # (c) tie
        return f"TIE — keep old (p={p:.3f}, HIGH {old_hr:.0%} vs {new_hr:.0%})"

    for old, new in comparisons:
        v = verdict(old, new)
        msg = f"FINAL: {old} vs {new}: {v}"
        lines.append(f"- {msg}")
        print(msg)

    # Write markdown
    OUT_MD.write_text("\n".join(lines))
    print(f"\nSummary → {OUT_MD}")


if __name__ == "__main__":
    main()
