"""
K* Retry: Fix HIGH recall by targeting weak methods and improving scoring prompt.
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
    DATA_DIR,
    KB_DIR,
    track,
    api_calls,
    estimated_cost,
)
from kstar_v2 import (
    _load_pairs_for_batch,
    METHOD_BATCHES,
)
from audit_kstar import run_audit, get_method_type, METHOD_SOURCES, TRAINING_CAT_MAP

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def get_client():
    load_dotenv(ENV_PATH, override=True)
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def run_induction_balanced(client, hypothesis_set, val_articles):
    """Improved scoring that reduces LOW bias."""
    if not val_articles:
        return 0.0

    rubric = "\n".join(f"{i+1}. {h}" for i, h in enumerate(hypothesis_set))
    correct = 0

    for art in val_articles:
        prompt = (
            "You are classifying Guardian newspaper articles for relevance to "
            "research on POLICY EXPERIMENTATION. An article is HIGH relevance if "
            "it describes an organization that tested, evaluated, compared, reviewed, "
            "or chose between policy options — using ANY method: formal trials, "
            "pre-post analysis, expert review, case study assessment, secondary data "
            "analysis, or even a gut decision where evidence was available but ignored.\n\n"
            "An article is LOW relevance if it is general news, opinion, commentary, "
            "or reporting that does NOT describe a specific organization evaluating "
            "or choosing between policy approaches.\n\n"
            "SCORING RUBRIC — an article is HIGH if it matches ANY of these:\n"
            f"{rubric}\n\n"
            "IMPORTANT: Many relevant articles do NOT use words like 'trial' or "
            "'experiment'. Look for the UNDERLYING STRUCTURE: did an organization "
            "face a choice, gather or consider evidence (even informally), and make "
            "a decision? If yes, classify as HIGH.\n\n"
            f"Title: {art['title']}\n"
            f"Excerpt: {art['excerpt'][:600]}\n\n"
            "Classify as HIGH or LOW. Answer with just one word."
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

    return correct / len(val_articles)


def targeted_abduction_weak(client, weak_methods, pairs_per_method=10):
    """Run targeted abduction focusing on weak method types."""
    new_hyps = []

    for method_name in weak_methods:
        # Find matching batch config
        batch_cfg = None
        for b in METHOD_BATCHES:
            if method_name.lower() in b["name"].lower():
                batch_cfg = b
                break

        if not batch_cfg:
            print(f"  {method_name}: no batch config found, skipping")
            continue

        pairs = _load_pairs_for_batch(batch_cfg, n_pairs=pairs_per_method)
        if not pairs:
            print(f"  {method_name}: no pairs available, skipping")
            continue

        desc = batch_cfg["description"]
        pairs_text = ""
        for i, p in enumerate(pairs, 1):
            pairs_text += (
                f"\nPair {i}\n"
                f"  HIGH: {p['high_title']} — {p['high_excerpt'][:500]}\n"
                f"  LOW: {p['low_title']} — {p['low_excerpt'][:500]}\n"
            )

        prompt = (
            f"These are pairs of Guardian newspaper articles. The FIRST in each pair "
            f"is relevant to policy experimentation research via **{desc}**. "
            f"The SECOND is NOT relevant.\n\n"
            f"IMPORTANT CONTEXT: Previous hypotheses focused too much on explicit "
            f"trial/experiment language and missed subtler forms of policy evaluation. "
            f"For {desc}, the article may NOT mention 'trial', 'experiment', or 'pilot'. "
            f"Instead, look for:\n"
            f"- How an organization FRAMED its choice or evaluation\n"
            f"- What type of evidence or reasoning is described (even informal)\n"
            f"- How conclusions or decisions are CONNECTED to the evaluation\n"
            f"- Structural patterns that distinguish genuine policy evaluation from "
            f"mere reporting\n\n"
            f"Generate hypotheses about what makes these HIGH articles recognizable as "
            f"policy evaluation articles. Be specific about structural and framing "
            f"features. Format each on its own line starting with ##\n\n"
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
        print(f"  {method_name}: {len(pairs)} pairs -> {len(batch_hyps)} hypotheses")
        new_hyps.extend(batch_hyps)

    unique = _deduplicate(new_hyps)
    print(f"  New hypotheses: {len(new_hyps)}, after dedup: {len(unique)}")
    return unique


def optimize_v3(client, pool, val_articles, iterations=80, min_k=10, max_k=14):
    """Optimization using balanced scoring prompt."""
    if len(pool) < min_k:
        acc = run_induction_balanced(client, pool, val_articles)
        return {"hypotheses": pool, "accuracy": acc, "pool_size": len(pool)}

    start_size = min(12, len(pool))
    K = random.sample(pool, start_size)
    best_K = list(K)
    best_acc = run_induction_balanced(client, K, val_articles)
    current_acc = best_acc

    T = 1.0
    T_min = 0.01
    cooling = (T_min / T) ** (1.0 / max(iterations - 1, 1))

    print(f"\n--- Optimization v3: {iterations} iters, target |K|={min_k}-{max_k} ---")
    print(f"  Initial accuracy: {best_acc:.3f} with {len(K)} hypotheses")

    for i in range(iterations):
        K_prime = list(K)
        pool_remaining = [h for h in pool if h not in K_prime]

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

        new_acc = run_induction_balanced(client, K_prime, val_articles)
        delta = new_acc - current_acc

        if delta > 0 or random.random() < math.exp(delta / max(T, 0.001)):
            K = K_prime
            current_acc = new_acc

        if new_acc > best_acc:
            best_K = list(K_prime)
            best_acc = new_acc

        T *= cooling

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Iter {i+1}/{iterations} | T={T:.4f} | curr={current_acc:.3f} | best={best_acc:.3f} | |K|={len(K)}")

    return {"hypotheses": best_K, "accuracy": best_acc, "pool_size": len(pool)}


def build_val_balanced(all_articles, n_high=30, n_low=30):
    """Build balanced validation set with method-type diversity."""
    highs = [a for a in all_articles if a["label"] == "HIGH"]
    lows = [a for a in all_articles if a["label"] == "LOW"]

    target_methods = ["RCT", "PrePost", "CaseStudy", "Expert_Qual", "Expert_Secondary", "Gut"]

    random.seed(42)
    val_highs = []
    for method in target_methods:
        method_arts = [a for a in highs if get_method_type(a) == method]
        random.shuffle(method_arts)
        take = min(5, len(method_arts), n_high // len(target_methods))
        val_highs.extend(method_arts[:take])

    random.shuffle(val_highs)
    val_highs = val_highs[:n_high]

    random.shuffle(lows)
    val_lows = lows[:n_low]

    val = []
    for art in val_highs:
        val.append({"title": art["title"], "excerpt": art["excerpt"], "label": "HIGH",
                     "method_type": get_method_type(art)})
    for art in val_lows:
        val.append({"title": art["title"], "excerpt": art["excerpt"], "label": "LOW",
                     "method_type": get_method_type(art)})
    random.shuffle(val)
    return val


def main():
    load_dotenv(ENV_PATH, override=True)
    client = get_client()

    max_retries = 3
    best_overall = None

    # Load existing pool
    kstar_path = KB_DIR / "K_star.json"
    with open(kstar_path) as f:
        kstar = json.load(f)
    current_pool = list(kstar["hypotheses"])

    # Also reload the v2 pool if we have it — we need the full 75
    # For now, start with current K* hypotheses as seed
    print("Loading full article set...")
    all_high_files, scored_files = _scan_data_files()
    all_articles, _ = _load_articles_from_scan(all_high_files, scored_files)
    for art in all_articles:
        art["method_type"] = get_method_type(art)

    for retry in range(1, max_retries + 1):
        print(f"\n{'='*60}")
        print(f"RETRY {retry}/{max_retries}")
        print(f"{'='*60}")

        # Identify weak methods from previous audit
        audit_path = KB_DIR / "audit_results.json"
        weak_methods = []
        if audit_path.exists():
            with open(audit_path) as f:
                prev_audit = json.load(f)
            per_method = prev_audit.get("per_method", {})
            for m in ["RCT", "PrePost", "CaseStudy", "Expert_Qual", "Expert_Secondary", "Gut"]:
                if per_method.get(m, 1.0) < 0.50:
                    weak_methods.append(m)

        if not weak_methods:
            weak_methods = ["Expert_Secondary", "Gut", "CaseStudy"]

        print(f"Targeting weak methods: {weak_methods}")

        # Step 1: Targeted abduction for weak methods
        print(f"\n[{retry}.1] Targeted abduction for weak methods...")
        new_hyps = targeted_abduction_weak(client, weak_methods, pairs_per_method=10)

        # Combine with existing pool
        current_pool = current_pool + new_hyps
        current_pool = _deduplicate(current_pool)
        print(f"Combined pool: {len(current_pool)} hypotheses")

        # Step 2: Build balanced validation set
        print(f"\n[{retry}.2] Building balanced validation set...")
        val = build_val_balanced(all_articles, n_high=30, n_low=30)
        print(f"Validation: {sum(1 for v in val if v['label']=='HIGH')} HIGH, {sum(1 for v in val if v['label']=='LOW')} LOW")

        # Step 3: Optimize with larger target
        target_min = 10 + (retry - 1) * 2  # 10, 12, 14
        target_max = target_min + 4
        print(f"\n[{retry}.3] Optimizing (target |K|={target_min}-{target_max}, 80 iterations)...")
        result = optimize_v3(client, current_pool, val, iterations=80,
                             min_k=target_min, max_k=target_max)

        # Save K*
        output = {
            "version": 2 + retry,
            "retry": retry,
            "hypotheses": result["hypotheses"],
            "accuracy": result["accuracy"],
            "pool_size": result["pool_size"],
            "api_calls": api_calls.copy(),
            "estimated_cost_usd": estimated_cost(),
        }
        with open(kstar_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nK* v{2+retry}: {len(result['hypotheses'])} hypotheses, val accuracy={result['accuracy']:.3f}")
        for i, h in enumerate(result["hypotheses"], 1):
            print(f"  {i}. {h}")

        # Step 4: Run audit
        print(f"\n[{retry}.4] Running audit...")
        passed, audit = run_audit()

        if best_overall is None or audit["overall_accuracy"] > best_overall["overall_accuracy"]:
            best_overall = audit
            best_overall["hypotheses"] = result["hypotheses"]
            best_overall["version"] = 2 + retry

        if passed:
            print(f"\n>>> K* PASSED QUALITY GATE on retry {retry} <<<")
            break
        else:
            print(f"\nRetry {retry} failed. Continuing...")
            # Update pool with current K* for next retry
            current_pool = list(set(current_pool))

    else:
        # All retries exhausted
        print(f"\n{'='*60}")
        print(f"K* DID NOT PASS AFTER {max_retries} RETRIES.")
        print(f"Best accuracy: {best_overall['overall_accuracy']:.1%}")
        print(f"Best HIGH accuracy: {best_overall['high_accuracy']:.1%}")
        print(f"Best LOW accuracy: {best_overall['low_accuracy']:.1%}")
        print("Pausing for human review.")
        print(f"{'='*60}")

        # Save best K*
        if best_overall:
            output = {
                "version": best_overall.get("version", "best"),
                "hypotheses": best_overall.get("hypotheses", []),
                "accuracy": best_overall["overall_accuracy"],
                "high_accuracy": best_overall["high_accuracy"],
                "low_accuracy": best_overall["low_accuracy"],
                "per_method": best_overall["per_method"],
                "note": f"Best of {max_retries} retries. Did not pass quality gate.",
                "api_calls": api_calls.copy(),
                "estimated_cost_usd": estimated_cost(),
            }
            with open(kstar_path, "w") as f:
                json.dump(output, f, indent=2)

    print(f"\nTotal API calls: {api_calls['count']}")
    print(f"Total cost: ${estimated_cost():.4f}")
    return best_overall


if __name__ == "__main__":
    main()
