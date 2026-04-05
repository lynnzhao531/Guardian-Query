"""
Model Agreement Monitor per MASTER_PLAN_v3.md §19.
Computes pairwise agreement, dimension correlation, outlier detection, trends.
"""

import json
from pathlib import Path
from collections import defaultdict

METHODS = ["decision", "method_rct", "method_prepost", "method_case_study",
           "method_expert_qual", "method_expert_secondary", "method_gut"]

MODEL_NAMES = ["model1", "model2", "model3", "model4"]


def compute_agreement_report(scored_articles):
    """
    Compute agreement metrics across models for a batch of scored articles.

    Args:
        scored_articles: list of dicts, each with:
            url, title, model1: {dim: {score, p0, p05, p1}}, model2: ..., etc.
            model2 may be None if unavailable.

    Returns:
        dict with agreement metrics
    """
    available_models = []
    for m in MODEL_NAMES:
        if any(art.get(m) is not None for art in scored_articles):
            available_models.append(m)

    n_articles = len(scored_articles)
    if n_articles == 0:
        return {"error": "no articles to analyze"}

    # A) Pairwise agreement: % articles where model pair agrees on high/low
    pairwise = {}
    for i, m1 in enumerate(available_models):
        for m2 in available_models[i+1:]:
            agree_count = 0
            total = 0
            for art in scored_articles:
                s1 = art.get(m1)
                s2 = art.get(m2)
                if s1 is None or s2 is None:
                    continue
                # High = decision score == 1 and max method p1 >= 0.80
                h1 = _is_high(s1)
                h2 = _is_high(s2)
                if h1 == h2:
                    agree_count += 1
                total += 1
            rate = agree_count / total if total > 0 else 0
            pairwise[f"{m1}_vs_{m2}"] = {"agreement_rate": round(rate, 3), "n": total}

    # B) Dimension-level agreement: per dimension, mean absolute score diff
    dim_agreement = {}
    for dim in METHODS:
        diffs = []
        for i, m1 in enumerate(available_models):
            for m2 in available_models[i+1:]:
                for art in scored_articles:
                    s1 = art.get(m1, {}).get(dim, {}).get("score")
                    s2 = art.get(m2, {}).get(dim, {}).get("score")
                    if s1 is not None and s2 is not None:
                        diffs.append(abs(s1 - s2))
        mean_diff = sum(diffs) / len(diffs) if diffs else 0
        dim_agreement[dim] = {"mean_abs_diff": round(mean_diff, 3), "n_comparisons": len(diffs)}

    # C) Systematic disagreement: articles where 1 model HIGH, rest LOW
    outlier_counts = defaultdict(int)
    for art in scored_articles:
        highs = []
        lows = []
        for m in available_models:
            s = art.get(m)
            if s is None:
                continue
            if _is_high(s):
                highs.append(m)
            else:
                lows.append(m)
        if len(highs) == 1 and len(lows) >= 2:
            outlier_counts[highs[0]] += 1

    # D) Overall agreement rate
    all_agree = 0
    for art in scored_articles:
        verdicts = []
        for m in available_models:
            s = art.get(m)
            if s is None:
                continue
            verdicts.append(_is_high(s))
        if len(verdicts) >= 2 and len(set(verdicts)) == 1:
            all_agree += 1
    overall_rate = all_agree / n_articles if n_articles > 0 else 0

    # Alerts
    alerts = []
    if overall_rate < 0.3:
        alerts.append(f"CRITICAL: overall agreement {overall_rate:.1%} < 30%")
    elif overall_rate < 0.5:
        alerts.append(f"WARNING: overall agreement {overall_rate:.1%} < 50%")

    for dim, info in dim_agreement.items():
        if info["mean_abs_diff"] > 0.7:
            alerts.append(f"WARNING: {dim} mean diff {info['mean_abs_diff']:.2f} > 0.7")

    return {
        "n_articles": n_articles,
        "available_models": available_models,
        "overall_agreement_rate": round(overall_rate, 3),
        "pairwise_agreement": pairwise,
        "dimension_agreement": dim_agreement,
        "outlier_model_counts": dict(outlier_counts),
        "alerts": alerts,
    }


def _is_high(model_scores):
    """Per-model high relevance: decision.score==1 AND max(method.p1)>=0.80"""
    if not model_scores:
        return False
    dec = model_scores.get("decision", {})
    if dec.get("score", 0) != 1:
        return False
    max_method_p1 = max(
        model_scores.get(m, {}).get("p1", 0)
        for m in METHODS if m != "decision"
    )
    return max_method_p1 >= 0.80


def format_report(report):
    """Format agreement report as readable text."""
    lines = [
        f"Model Agreement Report ({report['n_articles']} articles)",
        f"Models: {', '.join(report['available_models'])}",
        f"Overall agreement: {report['overall_agreement_rate']:.1%}",
        "",
        "Pairwise agreement:",
    ]
    for pair, info in report.get("pairwise_agreement", {}).items():
        lines.append(f"  {pair}: {info['agreement_rate']:.1%} (n={info['n']})")

    lines.append("\nDimension agreement (mean abs score diff):")
    for dim, info in report.get("dimension_agreement", {}).items():
        lines.append(f"  {dim}: {info['mean_abs_diff']:.3f}")

    if report.get("outlier_model_counts"):
        lines.append("\nOutlier counts (1 HIGH vs rest LOW):")
        for model, count in report["outlier_model_counts"].items():
            lines.append(f"  {model}: {count}")

    if report.get("alerts"):
        lines.append("\nALERTS:")
        for alert in report["alerts"]:
            lines.append(f"  {alert}")

    return "\n".join(lines)
