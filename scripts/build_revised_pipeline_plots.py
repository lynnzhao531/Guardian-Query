"""Build 5 PNGs for the 'Revised Pipeline (After Meeting)' page.

Plot 1 — plot_density_decision.png
    4 cols × 2 rows (7 models + 1 legend cell). Decision_p1 density
    split by high-by-design (A+B) vs low-by-design (C+D). Concern 3
    top-level.

Plot 2 — plot_scatter_decision_method.png  [filename preserved]
    2 cols × 3 rows (6 methods). Per-method calibration: green dots =
    articles labeled with method M; grey violin = articles NOT labeled
    with method M. Concern 4 top-level.

Plot 3 — plot_ranking_agreement.png
    Single 4×4 Spearman heatmap on decision_p1 rankings. Concern 5
    top-level.

Plot 4 — plot_per_method.png
    6 rows × 4 cols (methods × continuous models). Per-method density
    split by labeled vs not. Concern 3 fold.

Plot 4b — plot_per_method_ranking.png
    3 rows × 2 cols (6 methods). Per-method 4×4 Spearman heatmaps.
    Concern 5 fold.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
BENCH = ROOT / "outputs" / "benchmark" / "v1"
PLOTS = BENCH / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

MODELS = ["M1", "M2_old", "M2_new", "M3", "M4_v3", "M5", "M6"]
CONTINUOUS = ["M3", "M4_v3", "M5", "M6"]
CONT_SET = set(CONTINUOUS)
MODEL_LABEL = {
    "M1": "M1 Sonnet",
    "M2_old": "M2-old",
    "M2_new": "M2-new",
    "M3": "M3 Embeddings",
    "M4_v3": "M4-v3 Ridge",
    "M5": "M5 DistilBERT",
    "M6": "M6 Haiku",
}
METHODS = ["rct", "prepost", "case_study",
           "expert_qual", "expert_secondary", "gut"]

COLOR_HIGH = "#2ca02c"  # green (Target / high-by-design)
COLOR_LOW = "#7f7f7f"   # grey (Unrelated / low-by-design)


def load_data() -> pd.DataFrame:
    raw = pd.read_csv(BENCH / "raw_scores.csv")
    ans = pd.read_csv(ROOT / "synthetic_benchmark_answers.csv")
    return raw.merge(
        ans[["article_id", "category", "method_if_positive"]],
        on="article_id",
    )


# ── PLOT 1 ────────────────────────────────────────────────────────────────

def plot_decision_density(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    bins_disc = np.arange(-0.025, 1.05, 0.05)
    handles_store = [None, None]

    for i, m in enumerate(MODELS):
        ax = axes[i // 4, i % 4]
        hi = df[df["category"].isin(["A", "B"])][f"{m}_decision_p1"].dropna().values
        lo = df[df["category"].isin(["C", "D"])][f"{m}_decision_p1"].dropna().values

        if m in CONT_SET:
            if len(hi) > 1:
                sns.kdeplot(hi, ax=ax, fill=True, alpha=0.45,
                             color=COLOR_HIGH, clip=(0, 1), cut=0,
                             bw_adjust=0.8, label="High-decision (A+B)")
            if len(lo) > 1:
                sns.kdeplot(lo, ax=ax, fill=True, alpha=0.45,
                             color=COLOR_LOW, clip=(0, 1), cut=0,
                             bw_adjust=0.8, label="Low-decision (C+D)")
        else:
            ax.hist(hi, bins=bins_disc, alpha=0.6, color=COLOR_HIGH,
                    label="High-decision (A+B)")
            ax.hist(lo, bins=bins_disc, alpha=0.6, color=COLOR_LOW,
                    label="Low-decision (C+D)")

        tag = "continuous" if m in CONT_SET else "discrete"
        ax.set_title(f"{MODEL_LABEL[m]}  ({tag})", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("decision_p1", fontsize=9)
        if i % 4 == 0:
            ax.set_ylabel("density")
        ax.tick_params(labelsize=8)
        if handles_store[0] is None:
            h, _ = ax.get_legend_handles_labels()
            if len(h) >= 2:
                handles_store = h[:2]
        if ax.get_legend():
            ax.get_legend().remove()

        print(f"  {m}: A+B mean={hi.mean():.3f}  "
              f"C+D mean={lo.mean():.3f}  "
              f"uniq decision_p1={len(df[f'{m}_decision_p1'].unique())}")

    # 8th cell — legend
    ax = axes[1, 3]
    ax.axis("off")
    if handles_store[0] is not None:
        ax.legend(handles_store,
                  ["High-decision by design (A+B, 20 articles)",
                   "Low-decision by design (C+D, 20 articles)"],
                  loc="center", fontsize=11, frameon=True)
    fig.suptitle(
        "Concern 3 — Decision-dimension separation per model "
        "(Borderline excluded)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = PLOTS / "plot_density_decision.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ── PLOT 2 ────────────────────────────────────────────────────────────────

def plot_per_method_calibration_top(df: pd.DataFrame) -> None:
    """Per-method calibration: green dots (labeled with method) vs
    grey distribution (all other articles)."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    x_positions = np.arange(len(MODELS))
    rng = np.random.default_rng(42)

    for i, meth in enumerate(METHODS):
        ax = axes[i // 2, i % 2]
        # For each model, compute:
        #   hi_vals = scores on articles labeled with this method
        #   lo_vals = scores on articles NOT labeled with this method
        hi_per_model = []
        lo_per_model = []
        for m in MODELS:
            col = f"{m}_method_{meth}_p1"
            hi_mask = df["method_if_positive"] == meth
            hi_per_model.append(df.loc[hi_mask, col].values)
            lo_per_model.append(df.loc[~hi_mask, col].values)

        # Grey violin at each x position for "not labeled with method M"
        # (skip if all values equal — violin complains)
        lo_positions = []
        lo_data = []
        for xi, data in enumerate(lo_per_model):
            if len(data) >= 2 and len(np.unique(data)) > 1:
                lo_positions.append(xi)
                lo_data.append(data)
        if lo_data:
            parts = ax.violinplot(lo_data, positions=lo_positions,
                                    widths=0.7, showmeans=False,
                                    showmedians=False, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(COLOR_LOW)
                body.set_alpha(0.45)
                body.set_edgecolor("black")
                body.set_linewidth(0.3)

        # Green dots for "labeled with method M"
        for xi, hi_vals in enumerate(hi_per_model):
            if len(hi_vals) == 0:
                continue
            x_j = xi + rng.uniform(-0.12, 0.12, size=len(hi_vals))
            ax.scatter(x_j, hi_vals, color=COLOR_HIGH, s=100, alpha=0.9,
                        edgecolor="black", linewidth=0.6, zorder=5)
            if len(hi_vals) == 1:
                ax.annotate("n=1", (xi, float(hi_vals[0])),
                            xytext=(6, 4), textcoords="offset points",
                            fontsize=7, color="#2c3e50")

        ax.set_xticks(x_positions)
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS],
                            rotation=30, ha="right", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", alpha=0.2, lw=0.5)
        ax.set_ylabel(f"{meth} score", fontsize=10)
        ax.set_title(f"method_{meth}", fontsize=11)
        ax.grid(True, axis="y", alpha=0.15)

        n_hi = (df["method_if_positive"] == meth).sum()
        print(f"  method={meth}: n labeled = {n_hi}, "
              f"n not labeled = {len(df) - n_hi}")

    # Shared legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=COLOR_HIGH, markersize=10,
                    markeredgecolor="black", markeredgewidth=0.5),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_LOW, alpha=0.5,
                       edgecolor="black"),
    ]
    labels = [
        "Labeled with this method  (expected HIGH)",
        "Not labeled with this method  (expected LOW)",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=2,
                fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Concern 4 — Per-method calibration: do models score labeled "
        "articles high and others low?",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = PLOTS / "plot_scatter_decision_method.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ── PLOT 3 ────────────────────────────────────────────────────────────────

def plot_decision_ranking(df: pd.DataFrame) -> None:
    models = CONTINUOUS
    rel = {m: df[f"{m}_decision_p1"].values for m in models}
    n = len(models)
    mat = np.ones((n, n))
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i == j:
                continue
            rho, _ = spearmanr(rel[a], rel[b])
            mat[i, j] = 0.0 if np.isnan(rho) else rho
    labels = ["M3", "M4-v3", "M5", "M6"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues",
                 vmin=0, vmax=1, square=True,
                 xticklabels=labels, yticklabels=labels,
                 cbar_kws={"label": "Spearman ρ"}, ax=ax)
    ax.set_title(
        "Concern 5 — Pairwise Spearman on decision_p1 ranking "
        "(46 articles)",
        fontsize=11,
    )
    fig.tight_layout()
    out = PLOTS / "plot_ranking_agreement.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    off = [mat[i, j] for i in range(n) for j in range(i + 1, n)]
    print(f"  decision_p1 ranking mean Spearman = {np.mean(off):.3f}")
    print(f"wrote {out}")


# ── PLOT 4 ────────────────────────────────────────────────────────────────

def plot_per_method_density(df: pd.DataFrame) -> None:
    """6 methods × 4 continuous models. Density per cell."""
    n_methods = len(METHODS)
    n_models = len(CONTINUOUS)
    fig, axes = plt.subplots(n_methods, n_models, figsize=(12, 14))
    for r, meth in enumerate(METHODS):
        for c, m in enumerate(CONTINUOUS):
            ax = axes[r, c]
            col = f"{m}_method_{meth}_p1"
            hi = df[df["method_if_positive"] == meth][col].dropna().values
            lo = df[df["method_if_positive"] != meth][col].dropna().values

            if len(lo) >= 2:
                sns.kdeplot(lo, ax=ax, fill=True, alpha=0.4,
                             color=COLOR_LOW, clip=(0, 1), cut=0,
                             bw_adjust=0.8)
            if len(hi) >= 3:
                sns.kdeplot(hi, ax=ax, fill=True, alpha=0.5,
                             color=COLOR_HIGH, clip=(0, 1), cut=0,
                             bw_adjust=0.8)
            elif len(hi) > 0:
                for v in hi:
                    ax.axvline(v, color=COLOR_HIGH, lw=2, alpha=0.8)
                ax.text(0.5, 0.92, f"n(hi)={len(hi)}",
                        transform=ax.transAxes, ha="center",
                        fontsize=7, color="#2c3e50")
            else:
                ax.text(0.5, 0.5, "n(hi)=0",
                        transform=ax.transAxes, ha="center",
                        va="center", fontsize=9, color="#7f7f7f")

            ax.set_xlim(-0.02, 1.02)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(MODEL_LABEL[m], fontsize=10)
            if c == 0:
                ax.set_ylabel(meth, fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=30)
            else:
                ax.set_ylabel("")
            if r == n_methods - 1:
                ax.set_xlabel("method_p1", fontsize=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_HIGH, alpha=0.5),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_LOW, alpha=0.4),
    ]
    fig.legend(handles,
                ["Labeled with this method (expected HIGH)",
                 "Not labeled with this method (expected LOW)"],
                loc="lower center", ncol=2, fontsize=10,
                bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Per-method density per continuous model "
        "(rows=methods, cols=continuous models)",
        fontsize=11, y=0.999,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = PLOTS / "plot_per_method.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ── PLOT 4b ───────────────────────────────────────────────────────────────

def plot_per_method_ranking(df: pd.DataFrame) -> None:
    """6 per-method 4×4 Spearman heatmaps in a 3×2 grid."""
    models = CONTINUOUS
    labels = ["M3", "M4-v3", "M5", "M6"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    for i, meth in enumerate(METHODS):
        ax = axes[i // 2, i % 2]
        rel = {m: df[f"{m}_method_{meth}_p1"].values for m in models}
        n = len(models)
        mat = np.ones((n, n))
        for a_i, a in enumerate(models):
            for b_i, b in enumerate(models):
                if a_i == b_i:
                    continue
                rho, _ = spearmanr(rel[a], rel[b])
                mat[a_i, b_i] = 0.0 if np.isnan(rho) else rho
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues",
                     vmin=0, vmax=1, square=True,
                     xticklabels=labels, yticklabels=labels,
                     cbar=(i == len(METHODS) - 1), ax=ax,
                     cbar_kws={"label": "Spearman ρ"}
                               if i == len(METHODS) - 1 else None)
        ax.set_title(f"method_{meth}", fontsize=11)
    fig.suptitle(
        "Per-method pairwise Spearman rank correlation "
        "(4 continuous models, 46 articles)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = PLOTS / "plot_per_method_ranking.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    df = load_data()
    print(f"merged: {len(df)} articles; "
          f"{df['method_if_positive'].notna().sum()} have "
          f"ground-truth method\n")

    print("Plot 1 — decision dimension density (Concern 3):")
    plot_decision_density(df)

    print("\nPlot 2 — per-method calibration (Concern 4):")
    plot_per_method_calibration_top(df)

    print("\nPlot 3 — decision_p1 ranking 4×4 Spearman (Concern 5):")
    plot_decision_ranking(df)

    print("\nPlot 4 — per-method density (Concern 3 fold):")
    plot_per_method_density(df)

    print("\nPlot 4b — per-method ranking heatmaps (Concern 5 fold):")
    plot_per_method_ranking(df)


if __name__ == "__main__":
    main()
