"""Test K* hypotheses on held-out HIGH and LOW articles."""

import json
import random
import os
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from knowledge_synthesizer import (
    _scan_data_files,
    _load_articles_from_scan,
    _excerpt,
    DATA_DIR,
    KB_DIR,
    track,
    api_calls,
    estimated_cost,
)

def main():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=True)
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Load K*
    kstar_path = KB_DIR / "K_star.json"
    with open(kstar_path) as f:
        kstar = json.load(f)

    hypotheses = kstar["hypotheses"]
    print(f"Loaded K* with {len(hypotheses)} hypotheses (validation accuracy: {kstar['accuracy']:.3f})")
    print()

    # Load all articles
    all_high_files, scored_files = _scan_data_files()
    all_articles, _ = _load_articles_from_scan(all_high_files, scored_files)
    highs = [a for a in all_articles if a["label"] == "HIGH"]
    lows = [a for a in all_articles if a["label"] == "LOW"]

    # Sample 10 HIGH + 10 LOW (use different seed from validation to avoid overlap)
    random.seed(999)
    random.shuffle(highs)
    random.shuffle(lows)
    test_articles = highs[:10] + lows[:10]
    random.shuffle(test_articles)

    # Score each article
    rubric = "\n".join(f"- {h}" for h in hypotheses)
    correct = 0
    results = []

    print(f"{'#':<3} {'Predicted':<10} {'Actual':<8} {'OK?':<5} Title")
    print("-" * 90)

    for i, art in enumerate(test_articles, 1):
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
        is_correct = predicted == art["label"]
        if is_correct:
            correct += 1

        ok_str = "Y" if is_correct else "N"
        title_short = art["title"][:55] + "..." if len(art["title"]) > 55 else art["title"]
        print(f"{i:<3} {predicted:<10} {art['label']:<8} {ok_str:<5} {title_short}")

        results.append({
            "title": art["title"],
            "predicted": predicted,
            "actual": art["label"],
            "correct": is_correct,
        })

    accuracy = correct / len(test_articles)
    print("-" * 90)
    print(f"Test accuracy: {correct}/{len(test_articles)} = {accuracy:.1%}")
    print(f"API calls: {api_calls['count']}, est. cost: ${estimated_cost():.4f}")


if __name__ == "__main__":
    main()
