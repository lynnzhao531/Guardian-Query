"""
K* v2: Targeted abduction by method type + re-optimization with larger K*.
Addresses the 50% HIGH recall problem by generating hypotheses specific to
each evaluation method (not just explicit trial/pilot framing).
"""

import os
import json
import math
import random
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from knowledge_synthesizer import (
    _scan_data_files,
    _load_articles_from_scan,
    _detect_body_col,
    _excerpt,
    _deduplicate,
    run_induction,
    DATA_DIR,
    KB_DIR,
    api_calls,
    track,
    estimated_cost,
)

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def get_client():
    load_dotenv(ENV_PATH, override=True)
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ============================================================
# Targeted abduction batches
# ============================================================

METHOD_BATCHES = [
    {
        "name": "RCT",
        "description": "randomised controlled trials, A/B tests, or field experiments",
        "scored_file": "rct 2.csv",
        "gold_files": ["rct.csv"],
    },
    {
        "name": "PrePost",
        "description": "pre-post, before-after, or time-series comparisons of a policy change",
        "scored_file": "prepost 2.csv",
        "gold_files": ["prepost.csv"],
    },
    {
        "name": "CaseStudy",
        "description": "case studies of policy implementation without formal trials — learning from how a policy played out in practice",
        "scored_file": "case studies.csv",
        "gold_files": ["casestudy.csv"],
    },
    {
        "name": "Expert_Secondary",
        "description": "expert reviews, secondary data analysis, or quantitative assessment of existing policy outcomes using published data",
        "scored_file": "quantitative.csv",
        "gold_files": ["expert_secondary_quant.csv"],
    },
    {
        "name": "Expert_Qual",
        "description": "expert qualitative assessments — panels, consultations, commissions, or qualitative reviews of policy effectiveness",
        "scored_file": None,
        "gold_files": ["expert_qual.csv"],
    },
    {
        "name": "Gut_Decision",
        "description": "gut decisions — an organization chose a policy without formal evidence, or explicitly ignored evidence",
        "scored_file": "gut.csv",
        "gold_files": ["gut_decision.csv"],
    },
]


def _load_pairs_for_batch(batch_cfg, n_pairs=10):
    """Build HIGH/LOW pairs for a specific method type."""
    highs = []
    lows = []

    # Gold files (all HIGH)
    for fname in batch_cfg["gold_files"]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        body_col = _detect_body_col(df)
        if not body_col:
            continue
        for _, row in df.iterrows():
            highs.append({
                "title": row.get("title", "Untitled"),
                "excerpt": _excerpt(row[body_col]),
            })

    # Scored file HIGH rows
    if batch_cfg["scored_file"]:
        fpath = DATA_DIR / batch_cfg["scored_file"]
        if fpath.exists():
            df = pd.read_csv(fpath)
            body_col = _detect_body_col(df)
            if body_col and "Method" in df.columns:
                scored_high = df[(df["Method"] >= 0.5) & (df["Decision"] >= 0.5)]
                for _, row in scored_high.iterrows():
                    highs.append({
                        "title": row.get("title", "Untitled"),
                        "excerpt": _excerpt(row[body_col]),
                    })
                # LOW rows from same scored file
                scored_low = df[(df["Method"] == 0) & (df["Decision"] == 0)]
                for _, row in scored_low.iterrows():
                    lows.append({
                        "title": row.get("title", "Untitled"),
                        "excerpt": _excerpt(row[body_col]),
                    })

    # If no LOWs from scored file, pull from all scored files
    if not lows:
        for scored_name in ["case studies.csv", "gut.csv", "quantitative.csv", "rct 2.csv", "prepost 2.csv"]:
            fpath = DATA_DIR / scored_name
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath)
            body_col = _detect_body_col(df)
            if not body_col or "Method" not in df.columns:
                continue
            scored_low = df[(df["Method"] == 0) & (df["Decision"] == 0)]
            for _, row in scored_low.iterrows():
                lows.append({
                    "title": row.get("title", "Untitled"),
                    "excerpt": _excerpt(row[body_col]),
                })

    random.shuffle(highs)
    random.shuffle(lows)
    n = min(n_pairs, len(highs), len(lows))
    pairs = []
    for i in range(n):
        pairs.append({
            "high_title": highs[i]["title"],
            "high_excerpt": highs[i]["excerpt"],
            "low_title": lows[i]["title"],
            "low_excerpt": lows[i]["excerpt"],
        })
    return pairs


def run_targeted_abduction(client, n_pairs_per_batch=10):
    """Run one abduction batch per method type with method-specific prompts."""
    all_hypotheses = []

    for batch_cfg in METHOD_BATCHES:
        name = batch_cfg["name"]
        desc = batch_cfg["description"]
        pairs = _load_pairs_for_batch(batch_cfg, n_pairs=n_pairs_per_batch)
        if not pairs:
            print(f"  {name}: no pairs available, skipping")
            continue

        pairs_text = ""
        for i, p in enumerate(pairs, 1):
            pairs_text += (
                f"\nPair {i}\n"
                f"  HIGH: {p['high_title']} — {p['high_excerpt'][:500]}\n"
                f"  LOW: {p['low_title']} — {p['low_excerpt'][:500]}\n"
            )

        prompt = (
            f"These are pairs of Guardian newspaper articles about policy evaluation. "
            f"Specifically, these pairs are about **{desc}**.\n\n"
            f"The FIRST article in each pair scores HIGH on relevance to policy "
            f"experimentation research — it describes an organization evaluating or "
            f"choosing between policy options via {desc}. "
            f"The SECOND article does NOT score high.\n\n"
            f"What structural features distinguish a strong {desc} description? "
            f"Focus on:\n"
            f"- How the article FRAMES the evaluation (implicit vs explicit)\n"
            f"- How EVIDENCE or REASONING is structured\n"
            f"- How DECISIONS or CONCLUSIONS follow from the evaluation\n"
            f"- What distinguishes this from mere reporting or commentary\n\n"
            f"Generate hypotheses. Format each on its own line starting with ##\n\n"
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
        print(f"  {name}: {len(pairs)} pairs -> {len(batch_hyps)} hypotheses")
        all_hypotheses.extend(batch_hyps)

    unique = _deduplicate(all_hypotheses)
    print(f"  Total new: {len(all_hypotheses)}, after dedup: {len(unique)}")
    return unique


# ============================================================
# Re-optimization with larger K* target
# ============================================================

def run_optimization_v2(client, pool, val_articles, iterations=80, min_k=8, max_k=12):
    """Simulated annealing targeting 8-12 hypotheses."""
    if len(pool) < min_k:
        print(f"  Pool too small ({len(pool)}), returning all.")
        acc = run_induction(client, pool, val_articles)
        return {"hypotheses": pool, "accuracy": acc, "pool_size": len(pool)}

    # Start with random subset
    start_size = min(10, len(pool))
    K = random.sample(pool, start_size)
    best_K = list(K)
    best_acc = run_induction(client, K, val_articles)
    current_acc = best_acc

    T = 1.0
    T_min = 0.01
    cooling = (T_min / T) ** (1.0 / max(iterations - 1, 1))

    print(f"\n--- Optimization v2: {iterations} iterations, target |K|={min_k}-{max_k} ---")
    print(f"  Initial accuracy: {best_acc:.3f} with {len(K)} hypotheses")

    for i in range(iterations):
        K_prime = list(K)
        pool_remaining = [h for h in pool if h not in K_prime]

        # Choose move, respecting size bounds
        moves = []
        if len(K_prime) < max_k and pool_remaining:
            moves.append("add")
        if len(K_prime) > min_k:
            moves.append("remove")
        if pool_remaining:
            moves.append("swap")
        if not moves:
            moves = ["swap"] if pool_remaining else ["remove"]

        move = random.choice(moves)

        if move == "add" and pool_remaining:
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
# Testing with per-method breakdown
# ============================================================

def _get_method_label(article):
    """Infer method type from source file."""
    src = article.get("source_file", "").lower()
    if "rct" in src:
        return "RCT"
    elif "prepost" in src:
        return "PrePost"
    elif "case" in src:
        return "CaseStudy"
    elif "expert_qual" in src:
        return "Expert_Qual"
    elif "expert_secondary" in src or "quantitative" in src:
        return "Expert_Secondary"
    elif "gut" in src:
        return "Gut"
    elif "training" in src:
        cat = article.get("method_category", "").lower()
        if "rct" in cat:
            return "RCT"
        elif "prepost" in cat or "before" in cat:
            return "PrePost"
        elif "case" in cat:
            return "CaseStudy"
        elif "expert" in cat and "qual" in cat:
            return "Expert_Qual"
        elif "expert" in cat:
            return "Expert_Secondary"
        elif "gut" in cat:
            return "Gut"
    return "Unknown"


def run_test(client, hypotheses, test_articles, label="Test"):
    """Score articles and print results with per-method breakdown."""
    rubric = "\n".join(f"- {h}" for h in hypotheses)
    correct = 0
    results = []

    print(f"\n{'#':<3} {'Pred':<6} {'Actual':<6} {'OK?':<4} {'Method':<16} Title")
    print("-" * 100)

    for i, art in enumerate(test_articles, 1):
        prompt = (
            "Given these principles about what makes a Guardian article relevant "
            "to policy experimentation research (an organization testing, evaluating, "
            "or comparing policy options and making decisions based on results — "
            "including formal trials, pre-post comparisons, expert reviews, case studies, "
            "secondary data analysis, and even gut decisions where evidence was ignored):\n\n"
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
        is_correct = predicted == art["label"]
        if is_correct:
            correct += 1

        method = art.get("method_type", "Unknown")
        ok_str = "Y" if is_correct else "N"
        title_short = art["title"][:45] + "..." if len(art["title"]) > 45 else art["title"]
        print(f"{i:<3} {predicted:<6} {art['label']:<6} {ok_str:<4} {method:<16} {title_short}")
        results.append({"predicted": predicted, "actual": art["label"], "correct": is_correct, "method": method})

    accuracy = correct / len(test_articles)
    print("-" * 100)
    print(f"{label} accuracy: {correct}/{len(test_articles)} = {accuracy:.1%}")

    # Per-label accuracy
    for lbl in ["HIGH", "LOW"]:
        subset = [r for r in results if r["actual"] == lbl]
        if subset:
            sub_correct = sum(1 for r in subset if r["correct"])
            print(f"  {lbl}: {sub_correct}/{len(subset)} = {sub_correct/len(subset):.1%}")

    # Per-method accuracy
    methods = sorted(set(r["method"] for r in results))
    if len(methods) > 1:
        print("  Per method:")
        for m in methods:
            subset = [r for r in results if r["method"] == m]
            sub_correct = sum(1 for r in subset if r["correct"])
            print(f"    {m}: {sub_correct}/{len(subset)} = {sub_correct/len(subset):.1%}")

    return accuracy, results


# ============================================================
# Main
# ============================================================

def main():
    load_dotenv(ENV_PATH, override=True)
    client = get_client()

    print("=" * 60)
    print("K* v2: Targeted Abduction + Re-optimization")
    print("=" * 60)

    # Load existing pool from K* v1
    kstar_path = KB_DIR / "K_star.json"
    with open(kstar_path) as f:
        kstar_v1 = json.load(f)
    existing_hypotheses = kstar_v1.get("hypotheses", [])
    print(f"\nExisting K* v1: {len(existing_hypotheses)} hypotheses, accuracy={kstar_v1['accuracy']:.3f}")

    # We need the full pool from v1. Since we didn't save it, reload from abduction.
    # Actually, let's just use the v1 K* hypotheses as seed and build fresh from targeted batches.
    # The v1 pool had 128 hypotheses but we only kept 4. Let's regenerate.

    # Step 1: Targeted abduction
    print("\n[1] Running targeted abduction batches...")
    new_hypotheses = run_targeted_abduction(client, n_pairs_per_batch=10)

    # Combine with existing K* hypotheses
    combined_pool = list(existing_hypotheses) + new_hypotheses
    combined_pool = _deduplicate(combined_pool)
    print(f"\n  Combined pool: {len(combined_pool)} unique hypotheses")

    # Step 2: Build validation set
    print("\n[2] Building validation set...")
    all_high_files, scored_files = _scan_data_files()
    all_articles, _ = _load_articles_from_scan(all_high_files, scored_files)
    highs = [a for a in all_articles if a["label"] == "HIGH"]
    lows = [a for a in all_articles if a["label"] == "LOW"]

    random.seed(42)
    random.shuffle(highs)
    random.shuffle(lows)

    val = []
    for art in highs[:30]:
        art["method_type"] = _get_method_label(art)
        val.append({"title": art["title"], "excerpt": art["excerpt"], "label": "HIGH", "method_type": art["method_type"]})
    for art in lows[:30]:
        art["method_type"] = _get_method_label(art)
        val.append({"title": art["title"], "excerpt": art["excerpt"], "label": "LOW", "method_type": art["method_type"]})
    random.shuffle(val)
    print(f"  Validation: {len(val)} articles ({sum(1 for v in val if v['label']=='HIGH')} HIGH, {sum(1 for v in val if v['label']=='LOW')} LOW)")

    # Step 3: Re-optimization
    print("\n[3] Running optimization v2 (80 iterations, target 8-12 hypotheses)...")
    result = run_optimization_v2(client, combined_pool, val, iterations=80, min_k=8, max_k=12)

    # Save K* v2
    output = {
        "version": 2,
        "hypotheses": result["hypotheses"],
        "accuracy": result["accuracy"],
        "pool_size": result["pool_size"],
        "api_calls": api_calls.copy(),
        "estimated_cost_usd": estimated_cost(),
    }
    with open(kstar_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("K* v2 RESULTS")
    print("=" * 60)
    print(f"Validation accuracy: {result['accuracy']:.3f}")
    print(f"Pool size: {result['pool_size']}")
    print(f"K* contains {len(result['hypotheses'])} hypotheses:\n")
    for i, h in enumerate(result["hypotheses"], 1):
        print(f"  {i}. {h}")

    # Step 4: Test on original 20 articles
    print("\n\n[4] Testing on original 20 articles (same seed as v1 test)...")
    random.seed(999)
    random.shuffle(highs)
    random.shuffle(lows)
    test_orig = []
    for art in highs[:10]:
        test_orig.append({"title": art["title"], "excerpt": art["excerpt"], "label": "HIGH", "method_type": _get_method_label(art)})
    for art in lows[:10]:
        test_orig.append({"title": art["title"], "excerpt": art["excerpt"], "label": "LOW", "method_type": _get_method_label(art)})
    random.shuffle(test_orig)
    orig_acc, _ = run_test(client, result["hypotheses"], test_orig, label="Original 20")

    # Step 5: Test on 10 additional from underrepresented methods
    print("\n\n[5] Testing on 10 additional articles from underrepresented methods...")
    # Gather articles from expert_qual, expert_secondary, casestudy gold, gut_decision gold
    underrep_highs = []
    underrep_lows = []
    for art in all_articles:
        method = _get_method_label(art)
        if method in ("Expert_Qual", "Expert_Secondary", "CaseStudy", "Gut"):
            if art["label"] == "HIGH":
                art_copy = dict(art)
                art_copy["method_type"] = method
                underrep_highs.append(art_copy)
            elif art["label"] == "LOW":
                art_copy = dict(art)
                art_copy["method_type"] = method
                underrep_lows.append(art_copy)

    random.seed(777)
    random.shuffle(underrep_highs)
    random.shuffle(underrep_lows)
    # 5 HIGH + 5 LOW from underrepresented methods
    test_underrep = []
    for art in underrep_highs[:5]:
        test_underrep.append({"title": art["title"], "excerpt": art["excerpt"], "label": "HIGH", "method_type": art["method_type"]})
    for art in underrep_lows[:5]:
        test_underrep.append({"title": art["title"], "excerpt": art["excerpt"], "label": "LOW", "method_type": art["method_type"]})
    random.shuffle(test_underrep)

    if test_underrep:
        underrep_acc, _ = run_test(client, result["hypotheses"], test_underrep, label="Underrepresented methods")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"K* v2: {len(result['hypotheses'])} hypotheses")
    print(f"Validation accuracy: {result['accuracy']:.1%}")
    print(f"Original 20 test: {orig_acc:.1%}")
    if test_underrep:
        print(f"Underrepresented methods test: {underrep_acc:.1%}")
    print(f"\nTotal API calls: {api_calls['count']}")
    print(f"Total tokens: {api_calls['input_tokens']} in / {api_calls['output_tokens']} out")
    print(f"Estimated cost (this run): ${estimated_cost():.4f}")
    print(f"\nSaved to: {kstar_path}")


if __name__ == "__main__":
    main()
