"""
K* v3: Complete rebuild with all 6 fixes.
Fix 1: Expert notes as seed hypotheses
Fix 2: Prototype vs rubric APPEARANCE-GAP pairs
Fix 3: MID articles in validation (3-class scoring)
Fix 5: Method×Decision DIVERGENT pairs
Fix 6: Sonnet scorer, 0-5 scoring, few-shot examples
"""

import os
import json
import math
import random
from pathlib import Path

import pandas as pd
import numpy as np
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

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def get_client():
    load_dotenv(ENV_PATH, override=True)
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ============================================================
# FIX 1: Expert notes as seed hypotheses
# ============================================================

def load_expert_seeds():
    """Extract seed hypotheses from Training_cases.csv notes column."""
    tc = pd.read_csv(DATA_DIR / "Training_cases.csv")
    seeds = []
    by_method = {}

    for _, row in tc.iterrows():
        note = row.get("notes", "")
        cat = row.get("method_category", "unknown")
        if pd.isna(note) or not note.strip():
            continue
        seeds.append({
            "text": note.strip(),
            "method_category": cat,
            "provenance": "human_expert",
        })
        by_method.setdefault(cat, []).append(note.strip())

    print(f"  Loaded {len(seeds)} expert notes as seed hypotheses")
    for cat, notes in by_method.items():
        print(f"    {cat}: {len(notes)} notes")

    return [s["text"] for s in seeds], seeds


# ============================================================
# FIX 2: APPEARANCE-GAP pairs (prototype vs rubric divergence)
# ============================================================

def build_appearance_gap_pairs(n_pairs=30):
    """Articles that LOOK like experiments but aren't vs articles that
    DON'T look like experiments but ARE relevant."""
    tc = pd.read_csv(DATA_DIR / "Training_cases.csv")
    tc = tc.dropna(subset=["prototype_score_0to5", "rubric_score_0to5"])

    # Looks like experiment but isn't relevant: proto > rubric
    looks_like = tc[tc["prototype_score_0to5"] > tc["rubric_score_0to5"]].copy()
    # Relevant but doesn't look like experiment: rubric > proto
    is_relevant = tc[tc["rubric_score_0to5"] > tc["prototype_score_0to5"]].copy()

    pairs = []
    looks_list = looks_like.to_dict("records")
    relevant_list = is_relevant.to_dict("records")
    random.shuffle(looks_list)
    random.shuffle(relevant_list)

    for i in range(min(n_pairs, len(looks_list), len(relevant_list))):
        ll = looks_list[i % len(looks_list)]
        rl = relevant_list[i % len(relevant_list)]
        pairs.append({
            "false_positive_title": ll["title"],
            "false_positive_excerpt": _excerpt(ll.get("bodyText", "")),
            "false_positive_note": ll.get("notes", ""),
            "true_positive_title": rl["title"],
            "true_positive_excerpt": _excerpt(rl.get("bodyText", "")),
            "true_positive_note": rl.get("notes", ""),
        })

    print(f"  APPEARANCE-GAP pairs: {len(pairs)} (from {len(looks_list)} surface-match, {len(relevant_list)} deep-relevant)")
    return pairs


# ============================================================
# FIX 5: DIVERGENT pairs (Method high/Decision low vs reverse)
# ============================================================

def build_divergent_pairs(n_pairs=20):
    """Pair high-Method/low-Decision articles with low-Method/high-Decision."""
    high_method = []
    high_decision = []

    for fname in ["rct 2.csv", "prepost 2.csv", "case studies.csv", "quantitative.csv", "gut.csv"]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        body_col = _detect_body_col(df)
        if not body_col:
            continue

        hm = df[(df["Method"] >= 0.5) & (df["Decision"] == 0)]
        for _, row in hm.iterrows():
            high_method.append({
                "title": row.get("title", "Untitled"),
                "excerpt": _excerpt(row[body_col]),
                "source": fname,
            })

        hd = df[(df["Method"] == 0) & (df["Decision"] >= 0.5)]
        for _, row in hd.iterrows():
            high_decision.append({
                "title": row.get("title", "Untitled"),
                "excerpt": _excerpt(row[body_col]),
                "source": fname,
            })

    random.shuffle(high_method)
    random.shuffle(high_decision)
    n = min(n_pairs, len(high_method), len(high_decision))
    pairs = []
    for i in range(n):
        pairs.append({
            "high_method_title": high_method[i]["title"],
            "high_method_excerpt": high_method[i]["excerpt"],
            "high_decision_title": high_decision[i]["title"],
            "high_decision_excerpt": high_decision[i]["excerpt"],
        })

    print(f"  DIVERGENT pairs: {len(pairs)} (from {len(high_method)} highM/lowD, {len(high_decision)} lowM/highD)")
    return pairs


# ============================================================
# Standard CLEAR pairs (Fix 2 applied: use rubric_score for Training_cases)
# ============================================================

def build_clear_pairs(n_pairs=50):
    """HIGH vs LOW pairs for standard abduction. Training_cases uses rubric_score."""
    all_high_files, scored_files = _scan_data_files()
    highs = []
    lows = []

    # All-high files EXCEPT Training_cases (handled separately with rubric filter)
    for fname, df, body_col in all_high_files:
        if "training" in fname.lower():
            # Use rubric score: HIGH >= 3, LOW <= 1
            tc = df.copy()
            tc_high = tc[tc["rubric_score_0to5"] >= 3] if "rubric_score_0to5" in tc.columns else tc
            for _, row in tc_high.iterrows():
                highs.append({"title": row.get("title", ""), "excerpt": _excerpt(row[body_col])})
            tc_low = tc[tc["rubric_score_0to5"] <= 1] if "rubric_score_0to5" in tc.columns else pd.DataFrame()
            for _, row in tc_low.iterrows():
                lows.append({"title": row.get("title", ""), "excerpt": _excerpt(row[body_col])})
        else:
            for _, row in df.iterrows():
                highs.append({"title": row.get("title", ""), "excerpt": _excerpt(row[body_col])})

    # Scored files
    for fname, df, body_col in scored_files:
        scored_high = df[(df["Method"] >= 0.5) & (df["Decision"] >= 0.5)]
        for _, row in scored_high.iterrows():
            highs.append({"title": row.get("title", ""), "excerpt": _excerpt(row[body_col])})
        scored_low = df[(df["Method"] == 0) & (df["Decision"] == 0)]
        for _, row in scored_low.iterrows():
            lows.append({"title": row.get("title", ""), "excerpt": _excerpt(row[body_col])})

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
    print(f"  CLEAR pairs: {len(pairs)} (from {len(highs)} HIGH, {len(lows)} LOW)")
    return pairs


# ============================================================
# METHOD-EDGE pairs (per-method)
# ============================================================

def build_method_edge_pairs(n_per_method=10):
    """Within each scored file: Method>=0.5 vs Method=0."""
    method_files = {
        "RCT": "rct 2.csv",
        "PrePost": "prepost 2.csv",
        "CaseStudy": "case studies.csv",
        "Expert_Secondary": "quantitative.csv",
        "Gut": "gut.csv",
    }
    all_pairs = []
    for method_name, fname in method_files.items():
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        body_col = _detect_body_col(df)
        if not body_col:
            continue
        high = df[df["Method"] >= 0.5].to_dict("records")
        low = df[df["Method"] == 0].to_dict("records")
        random.shuffle(high)
        random.shuffle(low)
        n = min(n_per_method, len(high), len(low))
        for i in range(n):
            all_pairs.append({
                "high_title": high[i].get("title", ""),
                "high_excerpt": _excerpt(high[i].get(body_col, "")),
                "low_title": low[i].get("title", ""),
                "low_excerpt": _excerpt(low[i].get(body_col, "")),
                "method": method_name,
            })
    print(f"  METHOD-EDGE pairs: {len(all_pairs)} across {len(method_files)} methods")
    return all_pairs


# ============================================================
# DECISION-EDGE pairs
# ============================================================

def build_decision_edge_pairs(n_pairs=20):
    """Decision>=0.5 vs Decision=0 from any scored file."""
    high_d = []
    low_d = []
    for fname in ["rct 2.csv", "prepost 2.csv", "case studies.csv", "quantitative.csv", "gut.csv"]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        body_col = _detect_body_col(df)
        if not body_col:
            continue
        for _, row in df[df["Decision"] >= 0.5].iterrows():
            high_d.append({"title": row.get("title", ""), "excerpt": _excerpt(row[body_col])})
        for _, row in df[df["Decision"] == 0].iterrows():
            low_d.append({"title": row.get("title", ""), "excerpt": _excerpt(row[body_col])})

    random.shuffle(high_d)
    random.shuffle(low_d)
    n = min(n_pairs, len(high_d), len(low_d))
    pairs = []
    for i in range(n):
        pairs.append({
            "high_title": high_d[i]["title"], "high_excerpt": high_d[i]["excerpt"],
            "low_title": low_d[i]["title"], "low_excerpt": low_d[i]["excerpt"],
        })
    print(f"  DECISION-EDGE pairs: {len(pairs)}")
    return pairs


# ============================================================
# Abduction: 5 pair types with type-specific prompts
# ============================================================

def run_abduction_v3(client, clear_pairs, method_edge_pairs, decision_edge_pairs,
                     appearance_gap_pairs, divergent_pairs, batch_size=10):
    """Run abduction across 5 pair types with specialized prompts."""
    all_hypotheses = []

    # --- CLEAR batches (5 batches) ---
    n_clear = min(5, max(1, len(clear_pairs) // batch_size))
    print(f"\n  CLEAR abduction: {n_clear} batches")
    for b in range(n_clear):
        batch = clear_pairs[b*batch_size:(b+1)*batch_size]
        if not batch:
            break
        pairs_text = _format_clear_pairs(batch)
        prompt = (
            "Here are pairs of Guardian newspaper articles. The FIRST is HIGHLY RELEVANT "
            "to policy experimentation research — it describes an organization testing or "
            "comparing policy options. The SECOND is NOT relevant.\n\n"
            "Analyze structural and logical differences. Generate hypotheses about WHY "
            "the first articles are relevant. Focus on STRUCTURE, FRAMING, and LOGIC.\n\n"
            "Format each hypothesis on its own line starting with ##\n\n" + pairs_text
        )
        hyps = _call_abduction(client, prompt)
        print(f"    Batch {b+1}/{n_clear}: {len(hyps)} hypotheses")
        all_hypotheses.extend(hyps)

    # --- METHOD-EDGE batches (3 batches, grouped by method) ---
    methods = list(set(p.get("method", "") for p in method_edge_pairs))
    n_method = min(3, len(methods))
    print(f"\n  METHOD-EDGE abduction: {n_method} batches")
    random.shuffle(method_edge_pairs)
    for b in range(n_method):
        batch = method_edge_pairs[b*batch_size:(b+1)*batch_size]
        if not batch:
            break
        method_name = batch[0].get("method", "unknown")
        pairs_text = _format_clear_pairs(batch)
        prompt = (
            f"These pairs are specifically about **{method_name}** evaluation methodology.\n"
            f"The FIRST article in each pair has a strong METHOD description. "
            f"The SECOND does not.\n\n"
            f"What structural features distinguish a strong {method_name} description "
            f"from general reporting? Focus on how evidence and methodology are framed.\n\n"
            f"Format each hypothesis on its own line starting with ##\n\n" + pairs_text
        )
        hyps = _call_abduction(client, prompt)
        print(f"    Batch {b+1}/{n_method} ({method_name}): {len(hyps)} hypotheses")
        all_hypotheses.extend(hyps)

    # --- DECISION-EDGE batches (2 batches) ---
    n_dec = min(2, max(1, len(decision_edge_pairs) // batch_size))
    print(f"\n  DECISION-EDGE abduction: {n_dec} batches")
    for b in range(n_dec):
        batch = decision_edge_pairs[b*batch_size:(b+1)*batch_size]
        if not batch:
            break
        pairs_text = _format_clear_pairs(batch)
        prompt = (
            "These pairs focus on DECISION framing. The FIRST article has strong "
            "decision framing (an organization choosing between options). The SECOND lacks this.\n\n"
            "What structural features indicate that an organization is making or has made "
            "a deliberate policy choice based on (or despite) evidence?\n\n"
            "Format each hypothesis on its own line starting with ##\n\n" + pairs_text
        )
        hyps = _call_abduction(client, prompt)
        print(f"    Batch {b+1}/{n_dec}: {len(hyps)} hypotheses")
        all_hypotheses.extend(hyps)

    # --- APPEARANCE-GAP batches (3 batches) — Fix 2 ---
    n_app = min(3, max(1, len(appearance_gap_pairs) // batch_size))
    print(f"\n  APPEARANCE-GAP abduction: {n_app} batches")
    for b in range(n_app):
        batch = appearance_gap_pairs[b*batch_size:(b+1)*batch_size]
        if not batch:
            break
        pairs_text = ""
        for i, p in enumerate(batch, 1):
            pairs_text += (
                f"\nPair {i}\n"
                f"  LOOKS LIKE EXPERIMENT BUT ISN'T: {p['false_positive_title']}\n"
                f"    Expert note: {p['false_positive_note']}\n"
                f"    Excerpt: {p['false_positive_excerpt'][:400]}\n"
                f"  DOESN'T LOOK LIKE EXPERIMENT BUT IS RELEVANT: {p['true_positive_title']}\n"
                f"    Expert note: {p['true_positive_note']}\n"
                f"    Excerpt: {p['true_positive_excerpt'][:400]}\n"
            )
        prompt = (
            "These are unusual pairs. The FIRST article APPEARS to be about a policy "
            "experiment (it uses experimental language, describes trials) but is actually "
            "NOT very relevant to our research. The SECOND article does NOT look like a "
            "typical experiment but IS highly relevant.\n\n"
            "What DEEPER structural features distinguish genuine policy experimentation "
            "relevance from surface resemblance? What makes an article relevant even when "
            "it doesn't use trial/experiment language?\n\n"
            "Format each hypothesis on its own line starting with ##\n\n" + pairs_text
        )
        hyps = _call_abduction(client, prompt)
        print(f"    Batch {b+1}/{n_app}: {len(hyps)} hypotheses")
        all_hypotheses.extend(hyps)

    # --- DIVERGENT batches (2 batches) — Fix 5 ---
    n_div = min(2, max(1, len(divergent_pairs) // batch_size))
    print(f"\n  DIVERGENT abduction: {n_div} batches")
    for b in range(n_div):
        batch = divergent_pairs[b*batch_size:(b+1)*batch_size]
        if not batch:
            break
        pairs_text = ""
        for i, p in enumerate(batch, 1):
            pairs_text += (
                f"\nPair {i}\n"
                f"  STRONG METHOD, WEAK DECISION: {p['high_method_title']}\n"
                f"    {p['high_method_excerpt'][:400]}\n"
                f"  WEAK METHOD, STRONG DECISION: {p['high_decision_title']}\n"
                f"    {p['high_decision_excerpt'][:400]}\n"
            )
        prompt = (
            "Both articles are about policy. The FIRST describes a strong evaluation "
            "METHOD but weak decision framing (no clear policy choice). The SECOND has "
            "a clear DECISION being made but no rigorous evaluation method.\n\n"
            "What structural features distinguish strong METHOD descriptions from strong "
            "DECISION framing? How can we tell these apart in article text?\n\n"
            "Format each hypothesis on its own line starting with ##\n\n" + pairs_text
        )
        hyps = _call_abduction(client, prompt)
        print(f"    Batch {b+1}/{n_div}: {len(hyps)} hypotheses")
        all_hypotheses.extend(hyps)

    unique = _deduplicate(all_hypotheses)
    print(f"\n  Total LLM hypotheses: {len(all_hypotheses)}, after dedup: {len(unique)}")
    return unique


def _format_clear_pairs(pairs):
    text = ""
    for i, p in enumerate(pairs, 1):
        text += (
            f"\nPair {i}\n"
            f"  HIGH: {p.get('high_title','')} — {p.get('high_excerpt','')[:400]}\n"
            f"  LOW: {p.get('low_title','')} — {p.get('low_excerpt','')[:400]}\n"
        )
    return text


def _call_abduction(client, prompt):
    import time as _time
    for attempt in range(4):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            track(response)
            text = response.content[0].text
            return [line.strip().lstrip("#").strip() for line in text.split("\n") if line.strip().startswith("##")]
        except Exception as e:
            if attempt < 3:
                wait = 2 ** (attempt + 1)
                print(f"    Abduction API error (attempt {attempt+1}): {e}, retrying in {wait}s...")
                _time.sleep(wait)
            else:
                print(f"    Abduction API error (final): {e}, returning empty")
                return []


# ============================================================
# FIX 3 + FIX 6: Improved scoring (Sonnet, 0-5, few-shot, 3-class)
# ============================================================

def _build_few_shot_examples():
    """Build few-shot examples: 1 HIGH per method type + 2 MID + 1 LOW."""
    examples = []

    # HIGH examples from gold files (one per method)
    gold_files = {
        "RCT": ("rct.csv", "body"),
        "PrePost": ("prepost.csv", "article_body"),
        "CaseStudy": ("casestudy.csv", "article_body"),
        "Expert_Qual": ("expert_qual.csv", "body_text"),
        "Expert_Secondary": ("expert_secondary_quant.csv", "body_text"),
        "Gut": ("gut_decision.csv", "body_text"),
    }
    for method, (fname, body_col) in gold_files.items():
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        actual_col = _detect_body_col(df) or body_col
        if actual_col not in df.columns:
            continue
        row = df.iloc[0]
        examples.append({
            "title": row.get("title", ""),
            "excerpt": _excerpt(row[actual_col], 300),
            "label": "HIGH",
            "method": method,
            "score": 5,
            "explanation": f"Clear {method} evaluation with decision framing.",
        })

    # MID examples from scored files (Method=0.5)
    for fname in ["case studies.csv", "gut.csv"]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        body_col = _detect_body_col(df)
        if not body_col:
            continue
        mid = df[df["Method"] == 0.5].head(1)
        for _, row in mid.iterrows():
            examples.append({
                "title": row.get("title", ""),
                "excerpt": _excerpt(row[body_col], 300),
                "label": "MID",
                "score": 2,
                "explanation": "Partial method description — mentions evaluation but not structured.",
            })

    # LOW example
    for fname in ["case studies.csv"]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        body_col = _detect_body_col(df)
        if not body_col:
            continue
        low = df[(df["Method"] == 0) & (df["Decision"] == 0)].head(1)
        for _, row in low.iterrows():
            examples.append({
                "title": row.get("title", ""),
                "excerpt": _excerpt(row[body_col], 300),
                "label": "LOW",
                "score": 0,
                "explanation": "General news, no policy evaluation or decision.",
            })

    return examples


def _build_scoring_prompt(hypotheses, article, few_shot_examples):
    """Build the improved scoring prompt (Fix 6)."""
    rubric = "\n".join(f"{i+1}. {h}" for i, h in enumerate(hypotheses))

    examples_text = ""
    for ex in few_shot_examples[:4]:  # Keep prompt manageable
        examples_text += (
            f"\n  Example ({ex['label']}, score={ex['score']}): \"{ex['title'][:60]}\"\n"
            f"    {ex['explanation']}\n"
        )

    return (
        "You are classifying Guardian newspaper articles for relevance to "
        "research on POLICY EXPERIMENTATION. An article is relevant if it "
        "describes an organization that tested, evaluated, compared, reviewed, "
        "or chose between policy options — using ANY method:\n"
        "- Formal trials (RCT, A/B, field experiment)\n"
        "- Pre-post / before-after comparisons\n"
        "- Expert reviews, panels, commissions, consultations\n"
        "- Case studies of implementation\n"
        "- Secondary data analysis (econometric, observational)\n"
        "- Gut decisions (chose WITHOUT formal evidence — the ABSENCE of method IS the signal)\n\n"
        "SCORING RUBRIC — match ANY of these:\n"
        f"{rubric}\n\n"
        "CALIBRATION EXAMPLES:" + examples_text + "\n\n"
        "IMPORTANT: Score on a 0-5 scale:\n"
        "  0 = clearly irrelevant, general news/opinion\n"
        "  1 = vaguely related topic but no evaluation described\n"
        "  2 = mentions policy evaluation but structure is weak or unclear\n"
        "  3 = clear policy evaluation with some method + decision framing\n"
        "  4 = strong evaluation with clear method AND decision\n"
        "  5 = textbook example of policy experimentation\n\n"
        f"Title: {article['title']}\n"
        f"Excerpt: {article['excerpt'][:600]}\n\n"
        "Score this article 0-5. Answer with JUST the number."
    )


def score_article_v3(client, hypotheses, article, few_shot_examples, model="claude-haiku-4-5-20251001"):
    """Score an article 0-5 with retry logic for API errors."""
    prompt = _build_scoring_prompt(hypotheses, article, few_shot_examples)
    import time as _time
    for attempt in range(4):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            track(response)
            text = response.content[0].text.strip()
            try:
                score = int(text[0])
                return min(max(score, 0), 5)
            except (ValueError, IndexError):
                return 2
        except Exception as e:
            if attempt < 3:
                wait = 2 ** (attempt + 1)
                print(f"    API error (attempt {attempt+1}): {e}, retrying in {wait}s...")
                _time.sleep(wait)
            else:
                print(f"    API error (final): {e}, defaulting to 2")
                return 2


def map_score_to_class(score):
    """Map 0-5 score to HIGH/MID/LOW."""
    if score >= 3:
        return "HIGH"
    elif score >= 2:
        return "MID"
    else:
        return "LOW"


# ============================================================
# FIX 3: Validation with 3-class (HIGH, MID, LOW) + weighted accuracy
# ============================================================

def build_validation_set_v3():
    """15 HIGH + 15 MID + 15 LOW = 45 articles."""
    all_high_files, scored_files = _scan_data_files()

    highs, mids, lows = [], [], []

    # Training_cases: use rubric_score
    tc = pd.read_csv(DATA_DIR / "Training_cases.csv")
    tc_body = _detect_body_col(tc)
    if tc_body:
        for _, row in tc.iterrows():
            rs = row.get("rubric_score_0to5", 0)
            if pd.isna(rs):
                continue
            art = {"title": row.get("title", ""), "excerpt": _excerpt(row[tc_body]),
                   "method_type": row.get("method_category", "unknown")}
            if rs >= 3:
                art["label"] = "HIGH"
                highs.append(art)
            elif rs >= 2:
                art["label"] = "MID"
                mids.append(art)
            else:
                art["label"] = "LOW"
                lows.append(art)

    # All-high files (excluding Training_cases)
    for fname, df, body_col in all_high_files:
        if "training" in fname.lower():
            continue
        for _, row in df.iterrows():
            highs.append({"title": row.get("title", ""), "excerpt": _excerpt(row[body_col]),
                          "label": "HIGH", "method_type": fname.replace(".csv", "")})

    # Scored files
    for fname, df, body_col in scored_files:
        for _, row in df.iterrows():
            m, d = row.get("Method", 0), row.get("Decision", 0)
            art = {"title": row.get("title", ""), "excerpt": _excerpt(row[body_col]),
                   "method_type": fname.replace(".csv", "")}
            if m >= 0.5 and d >= 0.5:
                art["label"] = "HIGH"
                highs.append(art)
            elif m == 0.5 or d == 0.5:
                art["label"] = "MID"
                mids.append(art)
            elif m == 0 and d == 0:
                art["label"] = "LOW"
                lows.append(art)

    random.seed(42)
    random.shuffle(highs)
    random.shuffle(mids)
    random.shuffle(lows)

    val = highs[:15] + mids[:15] + lows[:15]
    random.shuffle(val)
    h = sum(1 for v in val if v["label"] == "HIGH")
    m = sum(1 for v in val if v["label"] == "MID")
    l = sum(1 for v in val if v["label"] == "LOW")
    print(f"  Validation set: {len(val)} articles ({h} HIGH, {m} MID, {l} LOW)")
    return val


def run_induction_v3(client, hypothesis_set, val_articles, few_shot_examples, model="claude-haiku-4-5-20251001"):
    """Score validation set with 0-5 scoring, weighted accuracy."""
    if not val_articles:
        return 0.0

    total_score = 0.0
    n = len(val_articles)

    for art in val_articles:
        score = score_article_v3(client, hypothesis_set, art, few_shot_examples, model=model)
        predicted = map_score_to_class(score)
        actual = art["label"]

        # Weighted accuracy: exact=1.0, adjacent=0.5, far=0.0
        class_order = {"LOW": 0, "MID": 1, "HIGH": 2}
        diff = abs(class_order.get(predicted, 1) - class_order.get(actual, 1))
        if diff == 0:
            total_score += 1.0
        elif diff == 1:
            total_score += 0.5
        # diff == 2: 0.0

    return total_score / n


# ============================================================
# Optimization v3
# ============================================================

def run_optimization_v3(client, pool, val_articles, few_shot_examples,
                        iterations=100, min_k=10, max_k=15):
    """Simulated annealing with 3-class validation."""
    if len(pool) < min_k:
        acc = run_induction_v3(client, pool, val_articles, few_shot_examples)
        return {"hypotheses": pool, "accuracy": acc, "pool_size": len(pool)}

    start_size = min(12, len(pool))
    K = random.sample(pool, start_size)
    best_K = list(K)
    best_acc = run_induction_v3(client, K, val_articles, few_shot_examples)
    current_acc = best_acc

    T = 1.0
    T_min = 0.01
    cooling = (T_min / T) ** (1.0 / max(iterations - 1, 1))

    print(f"\n--- Optimization v3: {iterations} iters, target |K|={min_k}-{max_k} ---")
    print(f"  Initial weighted accuracy: {best_acc:.3f} with {len(K)} hypotheses")

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

        new_acc = run_induction_v3(client, K_prime, val_articles, few_shot_examples)
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


# ============================================================
# Audit
# ============================================================

METHOD_SOURCES = {
    "rct.csv": "RCT", "rct 2.csv": "RCT",
    "prepost.csv": "PrePost", "prepost 2.csv": "PrePost",
    "casestudy.csv": "CaseStudy", "case studies.csv": "CaseStudy",
    "expert_qual.csv": "Expert_Qual",
    "expert_secondary_quant.csv": "Expert_Secondary", "quantitative.csv": "Expert_Secondary",
    "gut.csv": "Gut", "gut_decision.csv": "Gut",
}

TRAINING_CAT_MAP = {
    "RCT_Field_AB": "RCT", "PrePost_BeforeAfter": "PrePost",
    "CaseStudy": "CaseStudy", "Expert_Qualitative": "Expert_Qual",
    "Expert_SecondaryData": "Expert_Secondary", "Gut_NoLabel": "Gut",
}


def get_method_type(article):
    src = article.get("source_file", article.get("method_type", ""))
    if src in METHOD_SOURCES:
        return METHOD_SOURCES[src]
    for key, val in METHOD_SOURCES.items():
        if key.replace(".csv", "") in str(src):
            return val
    cat = article.get("method_category", article.get("method_type", ""))
    for k, v in TRAINING_CAT_MAP.items():
        if k.lower() in str(cat).lower():
            return v
    return str(cat) if cat else "Unknown"


def run_audit_v3(client, hypotheses, few_shot_examples):
    """Audit on 30 articles: 10 HIGH + 10 MID + 10 LOW with per-method breakdown."""
    all_high_files, scored_files = _scan_data_files()
    all_articles, _ = _load_articles_from_scan(all_high_files, scored_files)

    highs, mids, lows = [], [], []

    # Classify all articles
    tc = pd.read_csv(DATA_DIR / "Training_cases.csv")
    tc_body = _detect_body_col(tc)

    for art in all_articles:
        art["method_type"] = get_method_type(art)
        if art["label"] == "HIGH":
            highs.append(art)
        elif art["label"] == "LOW":
            lows.append(art)

    # MID articles from scored files
    for fname, df, body_col in scored_files:
        mid_rows = df[((df["Method"] == 0.5) | (df["Decision"] == 0.5)) &
                      ~((df["Method"] >= 0.5) & (df["Decision"] >= 0.5)) &
                      ~((df["Method"] == 0) & (df["Decision"] == 0))]
        for _, row in mid_rows.iterrows():
            mids.append({
                "title": row.get("title", ""),
                "excerpt": _excerpt(row[body_col]),
                "label": "MID",
                "method_type": get_method_type({"source_file": fname}),
            })

    random.seed(2026)
    random.shuffle(highs)
    random.shuffle(mids)
    random.shuffle(lows)

    # Build test set with method diversity for HIGHs
    target_methods = ["RCT", "PrePost", "CaseStudy", "Expert_Qual", "Expert_Secondary", "Gut"]
    test_highs = []
    for method in target_methods:
        method_arts = [a for a in highs if a["method_type"] == method]
        random.shuffle(method_arts)
        test_highs.extend(method_arts[:2])
    random.shuffle(test_highs)
    test_highs = test_highs[:10]

    test_articles = test_highs + mids[:10] + lows[:10]
    random.shuffle(test_articles)

    print(f"\nAudit set: {len(test_articles)} articles")
    print(f"  HIGH: {sum(1 for a in test_articles if a['label']=='HIGH')}")
    print(f"  MID: {sum(1 for a in test_articles if a['label']=='MID')}")
    print(f"  LOW: {sum(1 for a in test_articles if a['label']=='LOW')}")

    results = []
    print(f"\n{'#':<3} {'Score':<6} {'Pred':<6} {'Actual':<6} {'OK?':<4} {'Method':<16} Title")
    print("-" * 110)

    for i, art in enumerate(test_articles, 1):
        score = score_article_v3(client, hypotheses, art, few_shot_examples, model="claude-sonnet-4-6")
        predicted = map_score_to_class(score)
        actual = art["label"]

        class_order = {"LOW": 0, "MID": 1, "HIGH": 2}
        diff = abs(class_order.get(predicted, 1) - class_order.get(actual, 1))
        weighted = 1.0 if diff == 0 else (0.5 if diff == 1 else 0.0)

        ok_str = "Y" if diff == 0 else ("~" if diff == 1 else "N")
        method = art.get("method_type", "Unknown")
        title_short = art["title"][:45] + "..." if len(art["title"]) > 45 else art["title"]
        print(f"{i:<3} {score:<6} {predicted:<6} {actual:<6} {ok_str:<4} {method:<16} {title_short}")

        results.append({
            "title": art["title"], "score": score, "predicted": predicted,
            "actual": actual, "weighted": weighted, "method": method,
        })

    # Compute metrics
    total_weighted = sum(r["weighted"] for r in results) / len(results)

    for label in ["HIGH", "MID", "LOW"]:
        subset = [r for r in results if r["actual"] == label]
        if subset:
            exact = sum(1 for r in subset if r["predicted"] == label)
            weighted_sum = sum(r["weighted"] for r in subset)
            print(f"\n  {label}: {exact}/{len(subset)} exact, weighted={weighted_sum/len(subset):.2f}")

    # Per-method
    method_accs = {}
    for m in target_methods:
        subset = [r for r in results if r["method"] == m]
        if subset:
            w = sum(r["weighted"] for r in subset) / len(subset)
            method_accs[m] = w
            print(f"  {m}: weighted={w:.2f} ({len(subset)} articles)")

    # Quality gate
    high_results = [r for r in results if r["actual"] == "HIGH"]
    low_results = [r for r in results if r["actual"] == "LOW"]
    mid_results = [r for r in results if r["actual"] == "MID"]

    high_exact = sum(1 for r in high_results if r["predicted"] == "HIGH") / max(len(high_results), 1)
    low_exact = sum(1 for r in low_results if r["predicted"] == "LOW") / max(len(low_results), 1)
    mid_weighted = sum(r["weighted"] for r in mid_results) / max(len(mid_results), 1)
    methods_above_50 = sum(1 for m in target_methods if method_accs.get(m, 0) >= 0.50)

    print(f"\n{'='*60}")
    print("QUALITY GATE")
    print(f"{'='*60}")
    checks = [
        ("overall weighted >= 0.65", total_weighted >= 0.65, f"{total_weighted:.2f}"),
        ("HIGH exact accuracy >= 0.60", high_exact >= 0.60, f"{high_exact:.1%}"),
        ("LOW exact accuracy >= 0.80", low_exact >= 0.80, f"{low_exact:.1%}"),
        ("MID weighted accuracy >= 0.40", mid_weighted >= 0.40, f"{mid_weighted:.2f}"),
        (f">=4/6 methods weighted >= 0.50", methods_above_50 >= 4, f"{methods_above_50}/6"),
    ]

    passed = True
    for desc, ok, val in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            passed = False
        print(f"  [{status}] {desc} (actual: {val})")

    if passed:
        print("\n>>> K* v3 PASSED QUALITY GATE <<<")
    else:
        print("\n>>> K* v3 FAILED QUALITY GATE <<<")

    return passed, {
        "overall_weighted": total_weighted,
        "high_exact": high_exact,
        "low_exact": low_exact,
        "mid_weighted": mid_weighted,
        "per_method": method_accs,
        "results": results,
    }


# ============================================================
# Main
# ============================================================

def main():
    load_dotenv(ENV_PATH, override=True)
    client = get_client()

    print("=" * 60)
    print("K* v3: Complete Rebuild with All Fixes")
    print("=" * 60)

    # Fix 1: Load expert seeds
    print("\n[1] Loading expert notes as seed hypotheses (Fix 1)...")
    expert_seeds, expert_meta = load_expert_seeds()

    # Build all pair types
    print("\n[2] Building 5 pair types...")
    random.seed(2026)
    clear_pairs = build_clear_pairs(50)
    method_edge_pairs = build_method_edge_pairs(10)
    decision_edge_pairs = build_decision_edge_pairs(20)
    appearance_gap_pairs = build_appearance_gap_pairs(30)
    divergent_pairs = build_divergent_pairs(20)

    # Run abduction
    print("\n[3] Running 5-type abduction (15 batches)...")
    llm_hypotheses = run_abduction_v3(
        client, clear_pairs, method_edge_pairs, decision_edge_pairs,
        appearance_gap_pairs, divergent_pairs, batch_size=10,
    )

    # Combine expert seeds + LLM hypotheses
    combined_pool = expert_seeds + llm_hypotheses
    combined_pool = _deduplicate(combined_pool)
    print(f"\n  Combined pool: {len(expert_seeds)} expert + {len(llm_hypotheses)} LLM -> {len(combined_pool)} unique")

    # Build few-shot examples (Fix 6)
    print("\n[4] Building few-shot examples (Fix 6)...")
    few_shot_examples = _build_few_shot_examples()
    print(f"  {len(few_shot_examples)} few-shot examples")

    # Build validation set (Fix 3)
    print("\n[5] Building 3-class validation set (Fix 3)...")
    val_articles = build_validation_set_v3()

    # Optimization
    print("\n[6] Running optimization (100 iterations, target 10-15 hypotheses)...")
    result = run_optimization_v3(client, combined_pool, val_articles, few_shot_examples,
                                 iterations=100, min_k=10, max_k=15)

    # Save K*
    output = {
        "version": 3,
        "hypotheses": result["hypotheses"],
        "accuracy": result["accuracy"],
        "pool_size": result["pool_size"],
        "expert_seeds_count": len(expert_seeds),
        "llm_hypotheses_count": len(llm_hypotheses),
        "api_calls": api_calls.copy(),
        "estimated_cost_usd": estimated_cost(),
    }
    kstar_path = KB_DIR / "K_star.json"
    with open(kstar_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"K* v3: {len(result['hypotheses'])} hypotheses, weighted accuracy={result['accuracy']:.3f}")
    print(f"{'='*60}")
    for i, h in enumerate(result["hypotheses"], 1):
        print(f"  {i}. {h}")

    # Audit
    print(f"\n[7] Running audit...")
    passed, audit = run_audit_v3(client, result["hypotheses"], few_shot_examples)

    # Save audit
    audit["passed"] = passed
    audit_path = KB_DIR / "audit_results.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)

    if not passed:
        # One retry with larger K*
        print("\n[8] FAILED — retrying with larger K* (target 15-20)...")
        result2 = run_optimization_v3(client, combined_pool, val_articles, few_shot_examples,
                                      iterations=100, min_k=15, max_k=20)
        output2 = {
            "version": "3.1",
            "hypotheses": result2["hypotheses"],
            "accuracy": result2["accuracy"],
            "pool_size": result2["pool_size"],
            "api_calls": api_calls.copy(),
            "estimated_cost_usd": estimated_cost(),
        }
        with open(kstar_path, "w") as f:
            json.dump(output2, f, indent=2)

        print(f"\nK* v3.1: {len(result2['hypotheses'])} hypotheses, weighted accuracy={result2['accuracy']:.3f}")
        for i, h in enumerate(result2["hypotheses"], 1):
            print(f"  {i}. {h}")

        passed2, audit2 = run_audit_v3(client, result2["hypotheses"], few_shot_examples)
        audit2["passed"] = passed2
        with open(audit_path, "w") as f:
            json.dump(audit2, f, indent=2, default=str)

        if passed2:
            print("\n>>> K* v3.1 PASSED <<<")
        else:
            print("\n>>> K* v3.1 FAILED — stopping for human review <<<")

        passed = passed2

    print(f"\nTotal API calls: {api_calls['count']}")
    print(f"Total cost: ${estimated_cost():.4f}")
    return passed


if __name__ == "__main__":
    main()
