"""Build the two plots used by the Stage B dashboard section.

Saves to outputs/stage_b/v2/plots/:
  plot_topic_overlap.png        — horizontal bar of training-% per topic
  plot_article_stability.png    — companion-stability histogram, split
                                  by in_training
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
V2 = ROOT / "outputs" / "stage_b" / "v2"
OUT = V2 / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# Clean labels matching the dashboard table
SHORT_LABELS = {
    0: "International affairs and rights",
    1: "US politics",
    2: "Media and press coverage",
    3: "Youth services and mental health",
    4: "Mixed (water / schools / families)",
    5: "War and international security",
    6: "UK economic policy",
    7: "Mixed (corporate / aid)",
    8: "Australian government reviews",
    9: "Medical trials and research",
    10: "Housing and household costs",
    11: "Climate and energy policy",
    12: "Covid and vaccines",
    13: "Criminal justice",
    14: "UK benefits and carer's allowance",
}


def parse_index_md(path: Path) -> list[dict]:
    """Parse the INDEX.md table rows."""
    rows: list[dict] = []
    text = path.read_text()
    pat = re.compile(
        r"\|\s*T(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)%\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|"
    )
    for m in pat.finditer(text):
        rows.append({
            "topic": int(m.group(1)),
            "size": int(m.group(2)),
            "training_pct": int(m.group(3)),
            "raw_label": m.group(4).strip(),
            "relevance": m.group(5).strip(),
        })
    return rows


def plot_topic_overlap(rows: list[dict]) -> None:
    # Sort by training % desc
    rows_sorted = sorted(rows, key=lambda r: r["training_pct"], reverse=True)
    print("Plot A — parsed topic rows:")
    for r in rows_sorted:
        print(f"  T{r['topic']:2d} size={r['size']:3d} "
              f"train={r['training_pct']:2d}%  {SHORT_LABELS.get(r['topic'])}")

    labels = [
        f"T{r['topic']} — {SHORT_LABELS.get(r['topic'], r['raw_label'])}  "
        f"(n={r['size']})"
        for r in rows_sorted
    ]
    pcts = [r["training_pct"] for r in rows_sorted]

    def color_for(p: int) -> str:
        if p >= 60:
            return "#2ca02c"  # green
        if p >= 30:
            return "#4472C4"  # blue
        return "#ED7D31"      # orange

    colors = [color_for(p) for p in pcts]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(rows_sorted))
    ax.barh(y, pcts, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # highest training% at top
    ax.set_xlabel("% of topic's articles that appear in expert training data")
    ax.set_xlim(0, 100)
    ax.axvline(30, color="gray", linestyle=":", alpha=0.5, lw=0.8)
    ax.axvline(60, color="gray", linestyle=":", alpha=0.5, lw=0.8)
    # Annotate training % on each bar
    for i, p in enumerate(pcts):
        ax.text(p + 1, i, f"{p}%", va="center", fontsize=8)

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2ca02c"),
        plt.Rectangle((0, 0), 1, 1, color="#4472C4"),
        plt.Rectangle((0, 0), 1, 1, color="#ED7D31"),
    ]
    ax.legend(handles,
              ["Strong prior (≥60%)", "Moderate (30–60%)",
               "Little prior (<30%)"],
              loc="lower right", fontsize=9)
    ax.set_title("Training-data overlap per topic")
    fig.tight_layout()
    out = OUT / "plot_topic_overlap.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_article_stability() -> None:
    csv_path = V2 / "article_stability" / "article_companion_stability.csv"
    if not csv_path.exists():
        print(f"  MISSING {csv_path} — skipping Plot B")
        return
    df = pd.read_csv(csv_path)
    print("Plot B — article_companion_stability.csv columns:",
          df.columns.tolist())

    # Figure out the columns
    stab_col = "companion_stability"
    if stab_col not in df.columns:
        # fallback: any column with "stab" in name
        cands = [c for c in df.columns if "stab" in c.lower()]
        stab_col = cands[0] if cands else None
    training_col = "in_training"

    stab = df[stab_col].dropna().values
    in_training_raw = df[training_col].fillna("no").astype(str).str.lower()
    train_mask = in_training_raw.isin(["yes", "true", "1"]).values

    tr = stab[train_mask]
    nw = stab[~train_mask]
    print(f"  total n={len(stab)}; training={len(tr)} non-training={len(nw)}")
    for tag, arr in (("all", stab), ("training", tr), ("non-training", nw)):
        if len(arr) > 0:
            p10, p25, p50, p75, p90 = np.percentile(
                arr, [10, 25, 50, 75, 90])
            print(f"  {tag:12s} P10={p10:.3f} P25={p25:.3f} "
                  f"median={p50:.3f} P75={p75:.3f} P90={p90:.3f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(tr, bins=bins, alpha=0.6, color="#4472C4",
            label=f"Training articles (n={len(tr)})", edgecolor="black",
            linewidth=0.3)
    ax.hist(nw, bins=bins, alpha=0.55, color="#ED7D31",
            label=f"New articles (n={len(nw)})", edgecolor="black",
            linewidth=0.3)
    med_tr = float(np.median(tr)) if len(tr) else 0.0
    med_nw = float(np.median(nw)) if len(nw) else 0.0
    ax.axvline(med_tr, color="#1f3f88", linestyle="--", lw=1.2,
                label=f"Training median = {med_tr:.2f}")
    ax.axvline(med_nw, color="#a04b00", linestyle="--", lw=1.2,
                label=f"New median = {med_nw:.2f}")
    ax.set_xlabel("Article companion stability (0 = inconsistent cluster "
                  "membership, 1 = identical across seeds)")
    ax.set_ylabel("Count of articles")
    ax.set_title(
        "Article-cluster stability across 5 LDA runs (K=15, different seeds)"
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    out = OUT / "plot_article_stability.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    rows = parse_index_md(V2 / "topic_profiles" / "INDEX.md")
    if len(rows) != 15:
        print(f"WARNING: parsed {len(rows)} topics from INDEX.md, "
              f"expected 15")
    plot_topic_overlap(rows)
    print()
    plot_article_stability()


if __name__ == "__main__":
    main()
