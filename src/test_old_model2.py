"""Test old fine-tuned Model 2 against Models 1/3/4 on 20-article set."""
from __future__ import annotations
import csv, json, os, sys, time
csv.field_size_limit(10_000_000)
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data"
KSTAR_PATH = ROOT / "knowledge_base" / "K_star.json"
OLD_MODEL = "ft:gpt-4.1-mini-2025-04-14:personal::DAMjnKOH"

DIMS = ["decision", "method_rct", "method_prepost", "method_case_study",
        "method_expert_qual", "method_expert_secondary", "method_gut"]
BODY_COLS = ["body", "article_body", "body_text", "bodyText"]

# ── Data loading ─────────────────────────────────────────────────────────────

def _body(row):
    for c in BODY_COLS:
        if c in row and row[c] and str(row[c]).strip():
            return str(row[c])
    return ""

def _read_csv(fn):
    p = DATA_DIR / fn
    if not p.exists(): return []
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def _kstar():
    with open(KSTAR_PATH) as f:
        return json.load(f)["hypotheses"]

# ── Build test set ───────────────────────────────────────────────────────────

def build_test_set():
    """20 articles: 11 HIGH, 3 MID, 6 LOW."""
    articles = []

    # HIGH: 2 per method from all-high files
    gold_files = [
        ("rct.csv", "method_rct"),
        ("prepost.csv", "method_prepost"),
        ("casestudy.csv", "method_case_study"),
        ("expert_secondary_quant.csv", "method_expert_secondary"),
        ("expert_qual.csv", "method_expert_qual"),
        ("gut_decision.csv", "method_gut"),
    ]
    for fn, mdim in gold_files:
        rows = _read_csv(fn)
        count = 0
        for r in rows:
            if not r.get("title") or not _body(r).strip():
                continue
            articles.append({
                "title": r["title"], "body": _body(r),
                "actual_method": mdim, "actual_method_score": 1.0,
                "actual_decision": 1.0, "category": "HIGH",
            })
            count += 1
            if count >= 2:
                break

    # MID: 3 articles with Method=0.5 from scored files
    scored_files = [
        ("rct 2.csv", "method_rct", "Method", "Decision"),
        ("case studies.csv", "method_case_study", "Method", "Decision"),
        ("gut.csv", "method_gut", "Method", "Decision"),
    ]
    mid_count = 0
    for fn, mdim, mcol, dcol in scored_files:
        if mid_count >= 3:
            break
        rows = _read_csv(fn)
        for r in rows:
            if mid_count >= 3:
                break
            try:
                ms = float(r.get(mcol, -1))
            except (ValueError, TypeError):
                continue
            if ms != 0.5 or not r.get("title") or not _body(r).strip():
                continue
            ds = 0.5
            try:
                ds = float(r.get(dcol, 0.5))
            except (ValueError, TypeError):
                pass
            articles.append({
                "title": r["title"], "body": _body(r),
                "actual_method": mdim, "actual_method_score": 0.5,
                "actual_decision": ds, "category": "MID",
            })
            mid_count += 1

    # LOW: 6 articles with Method=0, Decision=0
    low_scored = [
        ("rct 2.csv", "method_rct", "Method", "Decision"),
        ("prepost 2.csv", "method_prepost", "Method", "Decision"),
        ("case studies.csv", "method_case_study", "Method", "Decision"),
        ("quantitative.csv", "method_expert_secondary", "Method", "Decision"),
        ("gut.csv", "method_gut", "Method", "Decision"),
    ]
    low_count = 0
    for fn, mdim, mcol, dcol in low_scored:
        if low_count >= 6:
            break
        rows = _read_csv(fn)
        for r in rows:
            if low_count >= 6:
                break
            try:
                ms = float(r.get(mcol, -1))
                ds = float(r.get(dcol, -1))
            except (ValueError, TypeError):
                continue
            if ms != 0 or ds != 0 or not r.get("title") or not _body(r).strip():
                continue
            articles.append({
                "title": r["title"], "body": _body(r),
                "actual_method": mdim, "actual_method_score": 0.0,
                "actual_decision": 0.0, "category": "LOW",
            })
            low_count += 1

    return articles

# ── Scoring functions ────────────────────────────────────────────────────────

def _sys_prompt_no_kstar():
    return (
        "You score Guardian newspaper articles for relevance to policy "
        "experimentation research on 7 dimensions. Each 0, 0.5, or 1. "
        "Use -1 for dimensions you cannot assess. Output JSON only.\n"
        "Keys: " + ", ".join(DIMS)
    )

def _sys_prompt_with_kstar(hyps):
    hb = "\n".join(f"  H{i+1}: {h}" for i, h in enumerate(hyps))
    return (
        "You score Guardian newspaper articles for relevance to policy "
        "experimentation research on 7 dimensions. Each 0, 0.5, or 1. "
        "Use -1 for dimensions you cannot assess. Output JSON only.\n\n"
        "## K* Hypotheses (validated knowledge)\n" + hb + "\n\n"
        "Keys: " + ", ".join(DIMS)
    )

def score_old_model(title, body, sys_prompt):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=OLD_MODEL, max_tokens=300, temperature=0,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Title: {title}\nExcerpt: {body[:800]}"}
        ]
    )
    txt = resp.choices[0].message.content.strip()
    if "```" in txt:
        txt = txt[txt.find("{"):txt.rfind("}") + 1]
    return json.loads(txt)

def score_model1(title, body):
    import model1_llm_judge as m1
    return m1.score_article(title, body[:800])

def score_model3(title, body):
    import model3_embedding_classifier as m3
    return m3.score_article(title, body[:800])

def score_model4(title, body):
    import model4_hypothesis_classifier as m4
    return m4.score_article(title, body[:800])

def _extract_decision_p1(scores):
    """Extract decision p1 from either flat or nested format."""
    d = scores.get("decision")
    if isinstance(d, dict):
        return d.get("p1", d.get("score", 0))
    return float(d) if d is not None else 0

def _extract_method_score(scores, mdim):
    """Extract method score from either flat or nested format."""
    v = scores.get(mdim)
    if isinstance(v, dict):
        return v.get("score", 0)
    return float(v) if v is not None else 0

def _extract_top_method(scores):
    """Get the method with highest score."""
    methods = [d for d in DIMS if d != "decision"]
    best_m, best_v = None, -1
    for m in methods:
        v = scores.get(m)
        if isinstance(v, dict):
            v = v.get("score", v.get("p1", 0))
        elif v is not None:
            v = float(v)
        else:
            v = 0
        if v > best_v:
            best_v = v
            best_m = m
    return best_m, best_v

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("Testing Old Fine-Tuned Model 2 vs Models 1/3/4")
    print("=" * 100)

    articles = build_test_set()
    print(f"\nTest set: {len(articles)} articles "
          f"({sum(1 for a in articles if a['category']=='HIGH')} HIGH, "
          f"{sum(1 for a in articles if a['category']=='MID')} MID, "
          f"{sum(1 for a in articles if a['category']=='LOW')} LOW)")

    hyps = _kstar()
    sys_no_k = _sys_prompt_no_kstar()
    sys_with_k = _sys_prompt_with_kstar(hyps)

    results = []
    for i, art in enumerate(articles):
        title = art["title"]
        body = art["body"]
        cat = art["category"]
        mdim = art["actual_method"]
        print(f"\n[{i+1}/{len(articles)}] {cat} | {mdim} | {title[:60]}...")

        # Score with all models
        try:
            old_no_k = score_old_model(title, body, sys_no_k)
        except Exception as e:
            print(f"  Old-noK* ERROR: {e}")
            old_no_k = {d: 0 for d in DIMS}

        time.sleep(0.5)

        try:
            old_with_k = score_old_model(title, body, sys_with_k)
        except Exception as e:
            print(f"  Old+K* ERROR: {e}")
            old_with_k = {d: 0 for d in DIMS}

        try:
            m1 = score_model1(title, body[:800])
        except Exception as e:
            print(f"  M1 ERROR: {e}")
            m1 = {d: {"score": 0, "p0": 1, "p05": 0, "p1": 0} for d in DIMS}

        try:
            m3 = score_model3(title, body[:800])
        except Exception as e:
            print(f"  M3 ERROR: {e}")
            m3 = {d: {"score": 0, "p0": 1, "p05": 0, "p1": 0} for d in DIMS}

        try:
            m4 = score_model4(title, body[:800])
        except Exception as e:
            print(f"  M4 ERROR: {e}")
            m4 = {d: {"score": 0, "p0": 1, "p05": 0, "p1": 0} for d in DIMS}

        results.append({
            "title": title, "category": cat, "actual_method": mdim,
            "actual_method_score": art["actual_method_score"],
            "actual_decision": art["actual_decision"],
            "old_no_k": old_no_k, "old_with_k": old_with_k,
            "m1": m1, "m3": m3, "m4": m4,
        })

        # Print inline
        def _fmt(scores, mdim):
            d = scores.get("decision")
            m = scores.get(mdim)
            if isinstance(d, dict): d = d.get("score", 0)
            if isinstance(m, dict): m = m.get("score", 0)
            return f"D={d} M={m}"

        print(f"  Old-noK*: {_fmt(old_no_k, mdim)}")
        print(f"  Old+K*:   {_fmt(old_with_k, mdim)}")
        print(f"  M1:       {_fmt(m1, mdim)}")
        print(f"  M3:       {_fmt(m3, mdim)}")
        print(f"  M4:       {_fmt(m4, mdim)}")

    # ── Print results table ──
    print("\n" + "=" * 120)
    print(f"{'Title':42s} {'Cat':5s} {'Actual':8s} {'Old-noK*':10s} {'Old+K*':10s} {'M1':10s} {'M3':10s} {'M4':10s}")
    print("-" * 120)
    for r in results:
        mdim = r["actual_method"]
        ms = r["actual_method_score"]
        ds = r["actual_decision"]
        actual = f"D{ds}M{ms}"

        def _short(scores, mdim):
            d = scores.get("decision")
            m = scores.get(mdim)
            if isinstance(d, dict): d = d.get("score", 0)
            if isinstance(m, dict): m = m.get("score", 0)
            if d is None: d = 0
            if m is None: m = 0
            return f"D{float(d):.0g}M{float(m):.0g}"

        print(f"{r['title'][:40]:42s} {r['category']:5s} {actual:8s} "
              f"{_short(r['old_no_k'], mdim):10s} {_short(r['old_with_k'], mdim):10s} "
              f"{_short(r['m1'], mdim):10s} {_short(r['m3'], mdim):10s} {_short(r['m4'], mdim):10s}")

    # ── Compute metrics ──
    print("\n" + "=" * 100)
    print("ACCURACY METRICS")
    print("=" * 100)

    def compute_metrics(results, model_key, label):
        """Compute accuracy for a model."""
        high_correct = high_total = 0
        mid_correct = mid_total = 0
        low_correct = low_total = 0
        method_correct = {}
        method_total = {}

        for r in results:
            cat = r["category"]
            mdim = r["actual_method"]
            actual_ms = r["actual_method_score"]
            actual_ds = r["actual_decision"]
            scores = r[model_key]

            # Extract predicted scores
            pred_d = scores.get("decision")
            pred_m = scores.get(mdim)
            if isinstance(pred_d, dict): pred_d = pred_d.get("score", 0)
            if isinstance(pred_m, dict): pred_m = pred_m.get("score", 0)
            if pred_d is None: pred_d = 0
            if pred_m is None: pred_m = 0
            pred_d = float(pred_d)
            pred_m = float(pred_m)

            # Decision accuracy: correct if within 0.25 of actual
            d_correct = abs(pred_d - actual_ds) <= 0.25
            # Method accuracy: correct if within 0.25 of actual
            m_correct = abs(pred_m - actual_ms) <= 0.25

            both_correct = d_correct and m_correct

            if cat == "HIGH":
                high_total += 1
                if both_correct: high_correct += 1
            elif cat == "MID":
                mid_total += 1
                if m_correct: mid_correct += 1  # MID: method accuracy
            else:
                low_total += 1
                if both_correct: low_correct += 1

            # Per-method
            short_m = mdim.replace("method_", "")
            if short_m not in method_total:
                method_total[short_m] = 0
                method_correct[short_m] = 0
            method_total[short_m] += 1
            if m_correct:
                method_correct[short_m] += 1

        h_acc = high_correct / high_total * 100 if high_total else 0
        m_acc = mid_correct / mid_total * 100 if mid_total else 0
        l_acc = low_correct / low_total * 100 if low_total else 0
        overall = (high_correct + mid_correct + low_correct) / len(results) * 100

        print(f"\n{label}:")
        print(f"  HIGH accuracy: {high_correct}/{high_total} = {h_acc:.0f}%")
        print(f"  MID accuracy:  {mid_correct}/{mid_total} = {m_acc:.0f}%")
        print(f"  LOW accuracy:  {low_correct}/{low_total} = {l_acc:.0f}%")
        print(f"  Overall:       {overall:.0f}%")
        print(f"  Per-method:")
        for m in sorted(method_total.keys()):
            acc = method_correct[m] / method_total[m] * 100 if method_total[m] else 0
            print(f"    {m:20s}: {method_correct[m]}/{method_total[m]} = {acc:.0f}%")

        return h_acc, m_acc, l_acc, overall

    h_old_nok, _, _, _ = compute_metrics(results, "old_no_k", "Old Model WITHOUT K*")
    h_old_k, _, _, _ = compute_metrics(results, "old_with_k", "Old Model WITH K*")
    h_m1, _, _, _ = compute_metrics(results, "m1", "Model 1 (Claude Sonnet)")
    h_m3, _, _, _ = compute_metrics(results, "m3", "Model 3 (Embedding+MLP)")
    h_m4, _, _, _ = compute_metrics(results, "m4", "Model 4 (K*+Ridge)")

    # ── Decision ──
    print("\n" + "=" * 100)
    print("DECISION")
    print("=" * 100)

    diff = h_m1 - h_old_k
    if h_old_k == 0 and h_m1 == 0:
        # Check if format is at least correct
        valid = all(isinstance(r["old_with_k"].get("decision"), (int, float))
                    for r in results)
        if valid:
            print("Both models score 0% HIGH — need more data to compare.")
        else:
            print("OLD MODEL INCOMPATIBLE — different output format.")
    elif diff <= 5:
        print(f"OLD MODEL USABLE. HIGH acc diff = {diff:.0f}% (within 5% of M1)")
        print(f"  Old+K*: {h_old_k:.0f}% vs M1: {h_m1:.0f}%")
        print(f"\nSave as Model 2: {OLD_MODEL}")
    elif diff <= 10:
        print(f"OLD MODEL MARGINAL. HIGH acc diff = {diff:.0f}%")
        print(f"  Old+K*: {h_old_k:.0f}% vs M1: {h_m1:.0f}%")
        print("Consider: usable but new fine-tuning recommended.")
    else:
        print(f"OLD MODEL UNDERPERFORMS. HIGH acc diff = {diff:.0f}% (>10%)")
        print(f"  Old+K*: {h_old_k:.0f}% vs M1: {h_m1:.0f}%")
        print("Proceed with new fine-tuning when OpenAI recovers.")

    # K* improvement
    k_improvement = h_old_k - h_old_nok
    print(f"\nK* improvement: {h_old_nok:.0f}% -> {h_old_k:.0f}% ({k_improvement:+.0f}%)")

    print(f"\nRecommendation on new fine-tuning:")
    print(f"  Old model was trained on unknown data from previous phase.")
    print(f"  New model would use K*-guided boundary pairs (387 T1, 417 T2, 46 hard neg).")
    if h_old_k >= h_m1 - 5:
        print(f"  Recommend: YES — old model is usable now, but boundary pairs should improve 0.5 discrimination.")
    else:
        print(f"  Recommend: YES — old model underperforms, boundary pairs should help significantly.")


if __name__ == "__main__":
    main()
