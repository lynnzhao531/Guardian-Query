"""
K* Audit: quality gate evaluation with per-method breakdown.
Tests 15 HIGH (across 6 method types) + 15 LOW articles.
"""

import os
import json
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
    DATA_DIR,
    KB_DIR,
    track,
    api_calls,
    estimated_cost,
)

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# Method type detection
METHOD_SOURCES = {
    "rct.csv": "RCT",
    "rct 2.csv": "RCT",
    "prepost.csv": "PrePost",
    "prepost 2.csv": "PrePost",
    "casestudy.csv": "CaseStudy",
    "case studies.csv": "CaseStudy",
    "expert_qual.csv": "Expert_Qual",
    "expert_secondary_quant.csv": "Expert_Secondary",
    "quantitative.csv": "Expert_Secondary",
    "gut.csv": "Gut",
    "gut_decision.csv": "Gut",
}

TRAINING_CAT_MAP = {
    "RCT_Field_AB": "RCT",
    "PrePost_BeforeAfter": "PrePost",
    "CaseStudy": "CaseStudy",
    "Expert_Qualitative": "Expert_Qual",
    "Expert_SecondaryData": "Expert_Secondary",
    "Gut_NoLabel": "Gut",
}


def get_method_type(article):
    src = article.get("source_file", "")
    if src in METHOD_SOURCES:
        return METHOD_SOURCES[src]
    # Training_cases: use method_category
    if "training" in src.lower():
        cat = article.get("method_category", "")
        for k, v in TRAINING_CAT_MAP.items():
            if k.lower() in cat.lower():
                return v
    # Scored files with _scored suffix
    for key, val in METHOD_SOURCES.items():
        if key.replace(".csv", "") in src.replace("_scored", ""):
            return val
    return "Unknown"


def run_audit():
    load_dotenv(ENV_PATH, override=True)
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Load K*
    kstar_path = KB_DIR / "K_star.json"
    with open(kstar_path) as f:
        kstar = json.load(f)
    hypotheses = kstar["hypotheses"]

    print("=" * 70)
    print(f"K* AUDIT (v{kstar.get('version', '?')}, {len(hypotheses)} hypotheses)")
    print("=" * 70)
    print(f"Validation accuracy from optimization: {kstar['accuracy']:.3f}")
    print(f"\nHypotheses:")
    for i, h in enumerate(hypotheses, 1):
        print(f"  {i}. {h}")

    # Load all articles with method types
    print("\n\nScanning data files...")
    all_high_files, scored_files = _scan_data_files()
    all_articles, _ = _load_articles_from_scan(all_high_files, scored_files)

    for art in all_articles:
        art["method_type"] = get_method_type(art)

    highs = [a for a in all_articles if a["label"] == "HIGH"]
    lows = [a for a in all_articles if a["label"] == "LOW"]

    # Build test set: 15 HIGH across all 6 method types + 15 LOW
    # Aim for ~2-3 per method type for HIGHs
    random.seed(2026)
    target_methods = ["RCT", "PrePost", "CaseStudy", "Expert_Qual", "Expert_Secondary", "Gut"]

    test_highs = []
    for method in target_methods:
        method_arts = [a for a in highs if a["method_type"] == method]
        random.shuffle(method_arts)
        # Take 2-3 per method, more for methods with more data
        take = min(3, len(method_arts))
        if take == 0:
            print(f"  WARNING: no HIGH articles for {method}")
        test_highs.extend(method_arts[:take])

    # Trim to 15 if we got too many
    random.shuffle(test_highs)
    test_highs = test_highs[:15]

    # 15 LOW
    random.shuffle(lows)
    test_lows = lows[:15]
    for a in test_lows:
        a["method_type"] = get_method_type(a)

    test_articles = test_highs + test_lows
    random.shuffle(test_articles)

    print(f"\nTest set: {len(test_articles)} articles ({len(test_highs)} HIGH, {len(test_lows)} LOW)")
    method_counts = {}
    for a in test_highs:
        m = a["method_type"]
        method_counts[m] = method_counts.get(m, 0) + 1
    print(f"HIGH by method: {method_counts}")

    # Score each article
    rubric = "\n".join(f"- {h}" for h in hypotheses)
    results = []

    print(f"\n{'#':<3} {'Pred':<6} {'Actual':<6} {'OK?':<4} {'Method':<16} Title")
    print("-" * 105)

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

        ok_str = "Y" if is_correct else "N"
        method = art.get("method_type", "Unknown")
        title_short = art["title"][:48] + "..." if len(art["title"]) > 48 else art["title"]
        print(f"{i:<3} {predicted:<6} {art['label']:<6} {ok_str:<4} {method:<16} {title_short}")

        results.append({
            "title": art["title"],
            "predicted": predicted,
            "actual": art["label"],
            "correct": is_correct,
            "method": method,
        })

    # Compute metrics
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    overall_acc = correct / total

    high_results = [r for r in results if r["actual"] == "HIGH"]
    low_results = [r for r in results if r["actual"] == "LOW"]
    high_correct = sum(1 for r in high_results if r["correct"])
    low_correct = sum(1 for r in low_results if r["correct"])
    high_acc = high_correct / len(high_results) if high_results else 0
    low_acc = low_correct / len(low_results) if low_results else 0

    print("-" * 105)
    print(f"\nOVERALL ACCURACY: {correct}/{total} = {overall_acc:.1%}")
    print(f"HIGH ACCURACY:   {high_correct}/{len(high_results)} = {high_acc:.1%}")
    print(f"LOW ACCURACY:    {low_correct}/{len(low_results)} = {low_acc:.1%}")

    # Per-method accuracy
    print("\nPER-METHOD ACCURACY:")
    method_accs = {}
    for method in target_methods + ["Unknown"]:
        method_results = [r for r in results if r["method"] == method]
        if not method_results:
            continue
        mc = sum(1 for r in method_results if r["correct"])
        ma = mc / len(method_results)
        method_accs[method] = ma
        high_in_method = [r for r in method_results if r["actual"] == "HIGH"]
        low_in_method = [r for r in method_results if r["actual"] == "LOW"]
        h_str = f"HIGH: {sum(1 for r in high_in_method if r['correct'])}/{len(high_in_method)}" if high_in_method else ""
        l_str = f"LOW: {sum(1 for r in low_in_method if r['correct'])}/{len(low_in_method)}" if low_in_method else ""
        detail = ", ".join(filter(None, [h_str, l_str]))
        print(f"  {method:<20} {mc}/{len(method_results)} = {ma:.1%}  ({detail})")

    # Quality gate
    methods_above_50 = sum(1 for m in target_methods if method_accs.get(m, 0) >= 0.50)

    print("\n" + "=" * 70)
    print("QUALITY GATE EVALUATION")
    print("=" * 70)
    checks = [
        ("overall_accuracy >= 0.65", overall_acc >= 0.65, f"{overall_acc:.1%}"),
        ("HIGH_accuracy >= 0.60", high_acc >= 0.60, f"{high_acc:.1%}"),
        ("LOW_accuracy >= 0.80", low_acc >= 0.80, f"{low_acc:.1%}"),
        (f">=4 of 6 methods >= 0.50", methods_above_50 >= 4, f"{methods_above_50}/6"),
    ]

    all_pass = True
    for desc, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc} (actual: {val})")

    if all_pass:
        print("\n>>> K* PASSED QUALITY GATE <<<")
    else:
        print("\n>>> K* FAILED QUALITY GATE <<<")
        # Identify weakest methods
        weak = [(m, method_accs.get(m, 0)) for m in target_methods if method_accs.get(m, 0) < 0.50]
        if weak:
            print(f"  Weakest methods: {', '.join(f'{m} ({a:.1%})' for m, a in weak)}")

    print(f"\nAPI calls this audit: {api_calls['count']}")
    print(f"Estimated cost: ${estimated_cost():.4f}")

    # Save audit results
    audit_output = {
        "overall_accuracy": overall_acc,
        "high_accuracy": high_acc,
        "low_accuracy": low_acc,
        "per_method": method_accs,
        "methods_above_50": methods_above_50,
        "passed": all_pass,
        "results": results,
    }
    audit_path = KB_DIR / "audit_results.json"
    with open(audit_path, "w") as f:
        json.dump(audit_output, f, indent=2)
    print(f"Audit saved to: {audit_path}")

    return all_pass, audit_output


if __name__ == "__main__":
    run_audit()
