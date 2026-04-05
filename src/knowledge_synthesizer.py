"""
K* Knowledge Synthesizer
Implements abduction-induction-optimization loop (Wang, Sudhir & Zhou 2025)
to discover what makes Guardian articles relevant to policy experimentation research.
"""

import os
import json
import math
import random
import time
from pathlib import Path

import pandas as pd
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv

# ---------- globals ----------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
KB_DIR.mkdir(exist_ok=True)

api_calls = {"count": 0, "input_tokens": 0, "output_tokens": 0}


def get_client():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=True)
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def track(response):
    api_calls["count"] += 1
    api_calls["input_tokens"] += response.usage.input_tokens
    api_calls["output_tokens"] += response.usage.output_tokens


def estimated_cost():
    # Sonnet: $3/M in, $15/M out; Haiku: $0.80/M in, $4/M out
    # rough blend — we track but don't split by model
    cost_in = api_calls["input_tokens"] / 1_000_000 * 3.0
    cost_out = api_calls["output_tokens"] / 1_000_000 * 15.0
    return cost_in + cost_out


# ============================================================
# A) build_pairs
# ============================================================

def _excerpt(text, max_chars=800):
    if pd.isna(text) or not text:
        return ""
    return str(text)[:max_chars]


def _detect_body_col(df):
    """Find the body/text column in a dataframe."""
    for col in ["body", "article_body", "body_text", "bodyText"]:
        if col in df.columns:
            return col
    return None


def _is_scored_file(df):
    """Check if a dataframe has numeric Method and Decision columns."""
    if "Method" not in df.columns or "Decision" not in df.columns:
        return False
    try:
        pd.to_numeric(df["Method"], errors="raise")
        pd.to_numeric(df["Decision"], errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def _scan_data_files():
    """Scan all CSVs in data/ and classify as SCORED or ALL-HIGH dynamically."""
    all_high_files = []   # (filename, df, body_col)
    scored_files = []     # (filename, df, body_col)

    for fpath in sorted(DATA_DIR.glob("*.csv")):
        df = pd.read_csv(fpath)
        body_col = _detect_body_col(df)
        if body_col is None:
            print(f"    {fpath.name}: SKIPPED (no body column found)")
            continue

        if _is_scored_file(df):
            n_high = len(df[(df["Method"] >= 0.5) & (df["Decision"] >= 0.5)])
            n_low = len(df[(df["Method"] == 0) & (df["Decision"] == 0)])
            print(f"    {fpath.name}: SCORED ({len(df)} rows, {n_high} high, {n_low} low)")
            scored_files.append((fpath.name, df, body_col))
        else:
            print(f"    {fpath.name}: ALL-HIGH ({len(df)} rows)")
            all_high_files.append((fpath.name, df, body_col))

    return all_high_files, scored_files


def _load_articles_from_scan(all_high_files, scored_files, high_only=False, low_only=False):
    """Extract article dicts from scanned files."""
    articles = []
    file_count = 0

    if not low_only:
        # All articles from ALL-HIGH files
        for fname, df, body_col in all_high_files:
            count = 0
            for _, row in df.iterrows():
                articles.append({
                    "title": row.get("title", "Untitled"),
                    "excerpt": _excerpt(row[body_col]),
                    "method_category": fname.replace(".csv", ""),
                    "source_file": fname,
                    "label": "HIGH",
                })
                count += 1
            if count > 0:
                file_count += 1

        # HIGH rows from scored files
        for fname, df, body_col in scored_files:
            subset = df[(df["Method"] >= 0.5) & (df["Decision"] >= 0.5)]
            count = 0
            for _, row in subset.iterrows():
                articles.append({
                    "title": row.get("title", "Untitled"),
                    "excerpt": _excerpt(row[body_col]),
                    "method_category": fname.replace(".csv", "") + "_scored",
                    "source_file": fname,
                    "label": "HIGH",
                })
                count += 1
            if count > 0:
                file_count += 1

    if not high_only:
        # LOW rows from scored files only
        for fname, df, body_col in scored_files:
            subset = df[(df["Method"] == 0) & (df["Decision"] == 0)]
            count = 0
            for _, row in subset.iterrows():
                articles.append({
                    "title": row.get("title", "Untitled"),
                    "excerpt": _excerpt(row[body_col]),
                    "method_category": fname.replace(".csv", "") + "_scored",
                    "source_file": fname,
                    "label": "LOW",
                })
                count += 1
            if count > 0:
                file_count += 1

    return articles, file_count


def build_pairs(n_pairs=100):
    """Build HIGH/LOW article pairs for abduction."""
    print("  Scanning data/ for CSV files...")
    all_high_files, scored_files = _scan_data_files()

    # Load HIGHs
    high_articles, high_fc = _load_articles_from_scan(all_high_files, scored_files, high_only=True)
    highs = [a for a in high_articles if a["label"] == "HIGH"]
    print(f"  HIGH pool: {len(highs)} articles from {high_fc} files")

    # Load LOWs
    low_articles, low_fc = _load_articles_from_scan(all_high_files, scored_files, low_only=True)
    lows = [a for a in low_articles if a["label"] == "LOW"]
    print(f"  LOW pool: {len(lows)} articles from {low_fc} scored files")

    # Pair randomly
    random.shuffle(highs)
    random.shuffle(lows)
    n = min(n_pairs, len(highs), len(lows))
    pairs = []
    for i in range(n):
        h = highs[i % len(highs)]
        l = lows[i % len(lows)]
        pairs.append({
            "high_title": h["title"],
            "high_excerpt": h["excerpt"],
            "low_title": l["title"],
            "low_excerpt": l["excerpt"],
            "method_category": h.get("method_category", "unknown"),
        })

    print(f"  Pairs built: {n}")
    cats = {}
    for p in pairs:
        cats[p["method_category"]] = cats.get(p["method_category"], 0) + 1
    print(f"  Categories: {cats}")
    return pairs


# ============================================================
# B) run_abduction
# ============================================================

def run_abduction(client, pairs, batch_size=10, num_batches=12):
    """Generate hypotheses from HIGH/LOW pairs using Claude Sonnet (reasoner)."""
    all_hypotheses = []

    total_batches = min(num_batches, max(1, len(pairs) // batch_size))
    print(f"\n--- Abduction: {total_batches} batches of {batch_size} pairs ---")

    for b in range(total_batches):
        start = (b * batch_size) % len(pairs)
        batch = []
        for i in range(batch_size):
            batch.append(pairs[(start + i) % len(pairs)])

        pairs_text = ""
        for i, p in enumerate(batch, 1):
            pairs_text += (
                f"\nPair {i}\n"
                f"  HIGH: {p['high_title']} — {p['high_excerpt'][:500]}\n"
                f"  LOW: {p['low_title']} — {p['low_excerpt'][:500]}\n"
            )

        prompt = (
            "Here are pairs of Guardian newspaper articles. The FIRST in each pair "
            "is HIGHLY RELEVANT to research on policy experimentation — it describes "
            "an organization testing or comparing policy options. The SECOND is NOT relevant.\n\n"
            "Analyze structural and logical differences. Generate hypotheses about WHY "
            "the first articles are relevant. Focus on article STRUCTURE (how information "
            "is organized), FRAMING (how choices are presented), and LOGIC (causal reasoning "
            "about policy), NOT on specific keywords or topics.\n\n"
            "Format each hypothesis on its own line starting with ##\n\n"
            f"{pairs_text}"
        )

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        track(response)

        text = response.content[0].text
        batch_hyps = [
            line.strip().lstrip("#").strip()
            for line in text.split("\n")
            if line.strip().startswith("##")
        ]
        print(f"  Batch {b+1}/{total_batches}: {len(batch_hyps)} hypotheses")
        all_hypotheses.extend(batch_hyps)

    # Deduplicate by word overlap
    unique = _deduplicate(all_hypotheses)
    print(f"\n  Raw hypotheses: {len(all_hypotheses)}, after dedup: {len(unique)}")
    return unique


def _deduplicate(hypotheses, threshold=0.9):
    """Remove hypotheses with >threshold word overlap."""
    if not hypotheses:
        return hypotheses

    def word_set(h):
        return set(h.lower().split())

    kept = [hypotheses[0]]
    for h in hypotheses[1:]:
        ws = word_set(h)
        is_dup = False
        for k in kept:
            ks = word_set(k)
            if not ws or not ks:
                continue
            overlap = len(ws & ks) / min(len(ws), len(ks))
            if overlap > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(h)
    return kept


# ============================================================
# C) run_induction
# ============================================================

def run_induction(client, hypothesis_set, val_articles):
    """Score a hypothesis set against labeled validation articles."""
    if not val_articles:
        return 0.0

    rubric = "\n".join(f"- {h}" for h in hypothesis_set)
    correct = 0

    for art in val_articles:
        prompt = (
            "Given these principles about what makes a Guardian article relevant "
            "to policy experimentation research (an organization testing or comparing "
            "policy options and making decisions based on results):\n\n"
            f"{rubric}\n\n"
            "Now classify this article as HIGH or LOW relevance:\n"
            f"Title: {art['title']}\n"
            f"Excerpt: {art['excerpt'][:600]}\n\n"
            "Answer only HIGH or LOW."
        )

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        track(response)

        answer = response.content[0].text.strip().upper()
        predicted = "HIGH" if "HIGH" in answer else "LOW"
        if predicted == art["label"]:
            correct += 1

    accuracy = correct / len(val_articles)
    return accuracy


# ============================================================
# D) run_optimization
# ============================================================

def _build_val_articles(n_high=30, n_low=30):
    """Build labeled validation set from known data."""
    print("  Scanning data/ for validation set...")
    all_high_files, scored_files = _scan_data_files()

    all_articles, _ = _load_articles_from_scan(all_high_files, scored_files)
    highs = [a for a in all_articles if a["label"] == "HIGH"]
    lows = [a for a in all_articles if a["label"] == "LOW"]

    random.seed(42)
    random.shuffle(highs)
    random.shuffle(lows)

    val = []
    for art in highs[:n_high]:
        val.append({"title": art["title"], "excerpt": art["excerpt"], "label": "HIGH"})
    for art in lows[:n_low]:
        val.append({"title": art["title"], "excerpt": art["excerpt"], "label": "LOW"})

    random.shuffle(val)
    print(f"  Validation set: {len(val)} articles ({sum(1 for v in val if v['label']=='HIGH')} HIGH, {sum(1 for v in val if v['label']=='LOW')} LOW)")
    return val


def run_optimization(client, pool, val_articles=None, iterations=60):
    """Simulated annealing to find optimal hypothesis set K*."""
    if val_articles is None:
        val_articles = _build_val_articles()

    if len(pool) < 6:
        print("  Pool too small for optimization, returning all.")
        acc = run_induction(client, pool, val_articles)
        return {"hypotheses": pool, "accuracy": acc, "pool_size": len(pool)}

    # Start with random subset of 6
    K = random.sample(pool, min(6, len(pool)))
    best_K = list(K)
    best_acc = run_induction(client, K, val_articles)
    current_acc = best_acc

    T = 1.0
    T_min = 0.01
    cooling = (T_min / T) ** (1.0 / max(iterations - 1, 1))

    print(f"\n--- Optimization: {iterations} iterations ---")
    print(f"  Initial accuracy: {best_acc:.3f}")

    for i in range(iterations):
        # Propose K'
        K_prime = list(K)
        move = random.choice(["add", "remove", "swap"])

        pool_remaining = [h for h in pool if h not in K_prime]

        if move == "add" and len(K_prime) < 10 and pool_remaining:
            K_prime.append(random.choice(pool_remaining))
        elif move == "remove" and len(K_prime) > 4:
            K_prime.remove(random.choice(K_prime))
        elif move == "swap" and pool_remaining:
            K_prime.remove(random.choice(K_prime))
            K_prime.append(random.choice(pool_remaining))

        new_acc = run_induction(client, K_prime, val_articles)
        delta = new_acc - current_acc

        if delta > 0 or random.random() < math.exp(delta / max(T, 0.001)):
            K = K_prime
            current_acc = new_acc

        if new_acc > best_acc:
            best_K = list(K_prime)
            best_acc = new_acc

        T *= cooling

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Iter {i+1}/{iterations} | T={T:.4f} | current={current_acc:.3f} | best={best_acc:.3f} | |K|={len(K)}")

    return {"hypotheses": best_K, "accuracy": best_acc, "pool_size": len(pool)}


# ============================================================
# E) main
# ============================================================

def main(dry_run=False, abduction_batches=12, abduction_batch_size=10):
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    client = get_client()

    print("=" * 60)
    print("K* Knowledge Synthesizer")
    print("=" * 60)

    # Step 1: Build pairs
    print("\n[1] Building HIGH/LOW pairs...")
    pairs = build_pairs(n_pairs=100)

    if dry_run:
        print(f"\n[DRY RUN] Built {len(pairs)} pairs across {len(set(p['method_category'] for p in pairs))} categories")
        print("\n[DRY RUN] Running 1 abduction batch (batch_size=5)...")
        hypotheses = run_abduction(client, pairs, batch_size=5, num_batches=1)
        print("\n--- Generated Hypotheses ---")
        for i, h in enumerate(hypotheses, 1):
            print(f"  {i}. {h}")
        print(f"\nAPI calls: {api_calls['count']}")
        print(f"Tokens: {api_calls['input_tokens']} in, {api_calls['output_tokens']} out")
        print(f"Estimated cost: ${estimated_cost():.4f}")
        return {"hypotheses": hypotheses, "pairs_count": len(pairs)}

    # Step 2: Abduction
    print(f"\n[2] Running abduction ({abduction_batches} batches of {abduction_batch_size})...")
    pool = run_abduction(client, pairs, batch_size=abduction_batch_size, num_batches=abduction_batches)

    # Step 3: Optimization
    print("\n[3] Building validation set...")
    val_articles = _build_val_articles()

    print("\n[4] Running simulated annealing optimization...")
    result = run_optimization(client, pool, val_articles, iterations=60)

    # Save K*
    output = {
        "hypotheses": result["hypotheses"],
        "accuracy": result["accuracy"],
        "pool_size": result["pool_size"],
        "api_calls": api_calls.copy(),
        "estimated_cost_usd": estimated_cost(),
    }
    out_path = KB_DIR / "K_star.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("K* RESULTS")
    print("=" * 60)
    print(f"Validation accuracy: {result['accuracy']:.3f}")
    print(f"Hypothesis pool size: {result['pool_size']}")
    print(f"K* contains {len(result['hypotheses'])} hypotheses:\n")
    for i, h in enumerate(result["hypotheses"], 1):
        print(f"  {i}. {h}")
    print(f"\nAPI calls: {api_calls['count']}")
    print(f"Tokens: {api_calls['input_tokens']} in / {api_calls['output_tokens']} out")
    print(f"Estimated cost: ${estimated_cost():.4f}")
    print(f"\nSaved to: {out_path}")
    return result


if __name__ == "__main__":
    import sys
    dry = "--dry" in sys.argv
    main(dry_run=dry)
