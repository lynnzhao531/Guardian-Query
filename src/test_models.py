"""Phase 2H: Test all available models on 20 articles (10 known-high, 10 known-low)."""

import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all_data
import model1_llm_judge as m1
import model3_embedding_classifier as m3
import model4_hypothesis_classifier as m4
from consensus import compute_consensus
from model_agreement_monitor import compute_agreement_report, format_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIMS = ["decision", "method_rct", "method_prepost", "method_case_study",
        "method_expert_qual", "method_expert_secondary", "method_gut"]


def main():
    print("=" * 70)
    print("Phase 2H: Model Test on 20 Articles")
    print("=" * 70)

    # Load data
    df = load_all_data()

    # Select 10 HIGH + 10 LOW
    high = df[df["decision"] == 1].copy()
    low = df[df["decision"] == 0].copy()

    random.seed(123)
    high_sample = high.sample(min(10, len(high)), random_state=123)
    low_sample = low.sample(min(10, len(low)), random_state=123)

    test_articles = []
    for _, row in high_sample.iterrows():
        test_articles.append({"title": row["title"], "body_text": str(row["body_text"])[:800],
                              "true_decision": 1, "method_type": row["method_type"]})
    for _, row in low_sample.iterrows():
        test_articles.append({"title": row["title"], "body_text": str(row["body_text"])[:800],
                              "true_decision": 0, "method_type": row["method_type"]})
    random.shuffle(test_articles)

    print(f"\nTest set: {len(test_articles)} articles ({len(high_sample)} HIGH, {len(low_sample)} LOW)")

    # Load Model 3
    print("\nLoading Model 3...")
    m3.load_models()

    # Load Model 4
    print("Loading Model 4...")
    m4.load_models()

    # Score each article with all models
    scored = []
    print(f"\n{'#':<3} {'M1_dec':<7} {'M3_dec':<7} {'M4_dec':<7} {'True':<5} Title")
    print("-" * 90)

    for i, art in enumerate(test_articles, 1):
        title = art["title"]
        body = art["body_text"]

        # Model 1: Sonnet LLM Judge
        try:
            s1 = m1.score_article(title, body)
        except Exception as e:
            print(f"  M1 error: {e}")
            s1 = None

        # Model 3: Embedding classifier
        try:
            s3 = m3.score_article(title, body)
        except Exception as e:
            print(f"  M3 error: {e}")
            s3 = None

        # Model 4: Hypothesis classifier
        try:
            s4 = m4.score_article(title, body)
        except Exception as e:
            print(f"  M4 error: {e}")
            s4 = None

        # Print decision scores
        d1 = s1["decision"]["score"] if s1 else "ERR"
        d3 = s3["decision"]["score"] if s3 else "ERR"
        d4 = s4["decision"]["score"] if s4 else "ERR"
        true_d = art["true_decision"]
        title_short = title[:50] + "..." if len(title) > 50 else title
        print(f"{i:<3} {d1:<7} {d3:<7} {d4:<7} {true_d:<5} {title_short}")

        scored.append({
            "url": f"test_{i}",
            "title": title,
            "model1": s1,
            "model2": None,  # unavailable
            "model3": s3,
            "model4": s4,
            "true_decision": true_d,
        })

    # Compute accuracy per model
    print("\n" + "=" * 70)
    print("Per-Model Decision Accuracy")
    print("=" * 70)

    for model_key in ["model1", "model3", "model4"]:
        correct = 0
        total = 0
        for art in scored:
            ms = art.get(model_key)
            if ms is None:
                continue
            pred = ms.get("decision", {}).get("score", -1)
            true = art["true_decision"]
            # Map: pred 1 = HIGH, pred 0 = LOW, pred 0.5 = MID
            pred_binary = 1 if pred >= 0.5 else 0
            if pred_binary == true:
                correct += 1
            total += 1
        acc = correct / total if total > 0 else 0
        print(f"  {model_key}: {correct}/{total} = {acc:.1%}")

    # Agreement report
    print("\n" + "=" * 70)
    print("Agreement Report")
    print("=" * 70)
    report = compute_agreement_report(scored)
    print(format_report(report))

    # Save results
    results_path = PROJECT_ROOT / "output" / "model_test_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"n_articles": len(scored), "report": report}, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
