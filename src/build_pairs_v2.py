"""Model 2 v2: Embedding-guided preference pair builder.

4-tier strategy: boundary (1v0.5), floor (0.5v0), extreme (1v0), hard negatives.
Uses sentence-transformers cosine similarity to find hardest contrasts.
"""
from __future__ import annotations
import csv, json, os, random, sys, time
csv.field_size_limit(10_000_000)
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

DATA_DIR = ROOT / "data"
KB_DIR = ROOT / "knowledge_base"
KSTAR_PATH = KB_DIR / "K_star.json"
STATE_PATH = ROOT / "project_state" / "STATE.json"
TRAIN_PATH = KB_DIR / "openai_training_7vec.jsonl"
VAL_PATH = KB_DIR / "openai_validation_7vec.jsonl"
BASE_MODEL = "gpt-4o-mini-2024-07-18"
EPOCHS = 3

DIMS = ["decision", "method_rct", "method_prepost", "method_case_study",
        "method_expert_qual", "method_expert_secondary", "method_gut"]

# Scored files: (filename, method_dim, method_col, decision_col)
SCORED_FILES = [
    ("rct 2.csv",        "method_rct",              "Method", "Decision"),
    ("prepost 2.csv",    "method_prepost",           "Method", "Decision"),
    ("case studies.csv", "method_case_study",        "Method", "Decision"),
    ("quantitative.csv", "method_expert_secondary",  "Method", "Decision"),
    ("gut.csv",          "method_gut",               "Method", "Decision"),
]
# All-high (gold) files: (filename, method_dim)
GOLD_FILES = [
    ("rct.csv",                   "method_rct"),
    ("prepost.csv",               "method_prepost"),
    ("casestudy.csv",             "method_case_study"),
    ("expert_secondary_quant.csv","method_expert_secondary"),
    ("expert_qual.csv",           "method_expert_qual"),
    ("gut_decision.csv",          "method_gut"),
]
# Training cases method_category -> dim
CATEGORY_MAP = {
    "RCT_Field_AB": "method_rct", "PrePost_BeforeAfter": "method_prepost",
    "CaseStudy": "method_case_study", "Expert_Qualitative": "method_expert_qual",
    "Expert_SecondaryData": "method_expert_secondary", "Gut_NoLabel": "method_gut",
}
BODY_COLS = ["body", "article_body", "body_text", "bodyText"]
MAX_BODY = 800
EMBED_INPUT_LEN = 400  # title + first N chars for embedding

_PROB = {0: {"p0": .80, "p05": .15, "p1": .05},
         0.5: {"p0": .20, "p05": .60, "p1": .20},
         1: {"p0": .05, "p05": .15, "p1": .80}}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _body(row: dict) -> str:
    for c in BODY_COLS:
        if c in row and row[c] and str(row[c]).strip():
            return str(row[c])
    return ""


def _parse_score(val) -> Optional[float]:
    if val is None or str(val).strip() == "":
        return None
    try:
        v = float(val)
    except (ValueError, TypeError):
        return None
    return {0: 0.0, 0.5: 0.5, 1: 1.0}.get(v)


def _read_csv(fn: str) -> list:
    p = DATA_DIR / fn
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _kstar() -> list:
    with open(KSTAR_PATH, encoding="utf-8") as f:
        return json.load(f)["hypotheses"]


def _state() -> dict:
    with open(STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_state(s: dict):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)


# ── Step 1: Embeddings ──────────────────────────────────────────────────────

def compute_embeddings(articles: List[dict]) -> np.ndarray:
    """Embed title + first 400 chars of body for all articles."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = []
    for a in articles:
        t = a.get("title", "")
        b = _body(a)[:EMBED_INPUT_LEN]
        texts.append(f"{t}. {b}" if b else t)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return np.array(embeddings, dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(d / n) if n > 0 else 0.0


def cosine_sim_matrix(embs: np.ndarray) -> np.ndarray:
    """Compute NxN cosine similarity matrix."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embs / norms
    return normed @ normed.T


# ── Step 2: Build pairs ─────────────────────────────────────────────────────

def _make_art_record(row: dict, mdim: str, mscore: float,
                     dscore: Optional[float], notes: str = "") -> dict:
    """Create a standardized article record with scores."""
    sc = {d: -1 for d in DIMS}
    sc[mdim] = mscore
    sc["decision"] = dscore if dscore is not None else -1
    return {
        "title": row.get("title", ""),
        "body": _body(row),
        "scores": sc,
        "notes": notes,
        "_method_dim": mdim,
        "_method_score": mscore,
        "_decision_score": dscore if dscore is not None else -1,
    }


def _find_hardest_pair(target_emb: np.ndarray, candidate_embs: np.ndarray,
                       candidate_indices: List[int]) -> Tuple[int, float]:
    """Find the candidate with highest cosine similarity to target."""
    if len(candidate_indices) == 0:
        return -1, 0.0
    sims = []
    for idx in candidate_indices:
        sims.append(cosine_sim(target_emb, candidate_embs[idx]))
    best = np.argmax(sims)
    return candidate_indices[best], sims[best]


def build_method_pairs(articles: List[dict], embeddings: np.ndarray,
                       mdim: str) -> Tuple[List[dict], dict]:
    """Build 4-tier pairs for a single method dimension."""
    # Bucket by method score
    bkt = {0.0: [], 0.5: [], 1.0: []}
    for i, a in enumerate(articles):
        ms = a["_method_score"]
        if ms in bkt:
            bkt[ms].append(i)

    pairs = []
    stats = {"tier1": 0, "tier2": 0, "tier3": 0, "tier4": 0,
             "tier1_sims": [], "tier3_sims": []}

    # TIER 1: Boundary (1 vs 0.5) — hardest contrast, cosine-matched
    for i_hi in bkt[1.0]:
        if not bkt[0.5]:
            break
        best_idx, sim = _find_hardest_pair(embeddings[i_hi], embeddings, bkt[0.5])
        if best_idx >= 0:
            pairs.append({
                "winner": articles[i_hi], "loser": articles[best_idx],
                "dimension": mdim, "pair_type": "method_boundary_high",
                "cosine_sim": sim,
            })
            stats["tier1"] += 1
            stats["tier1_sims"].append(sim)

    # TIER 2: Floor (0.5 vs 0) — cosine-matched
    for i_mid in bkt[0.5]:
        if not bkt[0.0]:
            break
        best_idx, sim = _find_hardest_pair(embeddings[i_mid], embeddings, bkt[0.0])
        if best_idx >= 0:
            pairs.append({
                "winner": articles[i_mid], "loser": articles[best_idx],
                "dimension": mdim, "pair_type": "method_boundary_low",
                "cosine_sim": sim,
            })
            stats["tier2"] += 1

    # TIER 3: Extreme (1 vs 0) — random, for calibration
    n_extreme = max(1, int(len(pairs) * 0.15 / 0.85)) if pairs else min(len(bkt[1.0]), len(bkt[0.0]))
    if bkt[1.0] and bkt[0.0]:
        hi_sample = random.sample(bkt[1.0], min(n_extreme, len(bkt[1.0])))
        lo_sample = bkt[0.0]
        for i_hi in hi_sample:
            i_lo = random.choice(lo_sample)
            sim = cosine_sim(embeddings[i_hi], embeddings[i_lo])
            pairs.append({
                "winner": articles[i_hi], "loser": articles[i_lo],
                "dimension": mdim, "pair_type": "method_extreme",
                "cosine_sim": sim,
            })
            stats["tier3"] += 1
            stats["tier3_sims"].append(sim)

    # TIER 4: Hard negatives — high similarity, different scores
    sim_matrix = cosine_sim_matrix(embeddings)
    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            si, sj = articles[i]["_method_score"], articles[j]["_method_score"]
            if abs(si - sj) < 0.5:
                continue
            if sim_matrix[i, j] <= 0.7:
                continue
            winner_idx = i if si > sj else j
            loser_idx = j if si > sj else i
            pairs.append({
                "winner": articles[winner_idx], "loser": articles[loser_idx],
                "dimension": mdim, "pair_type": "method_hard_negative",
                "cosine_sim": float(sim_matrix[i, j]),
            })
            stats["tier4"] += 1

    return pairs, stats


def build_decision_pairs(all_scored_articles: List[dict],
                         all_embeddings: np.ndarray) -> Tuple[List[dict], dict]:
    """Build 4-tier pairs for the decision dimension, pooled across files."""
    bkt = {0.0: [], 0.5: [], 1.0: []}
    for i, a in enumerate(all_scored_articles):
        ds = a["_decision_score"]
        if ds in bkt:
            bkt[ds].append(i)

    pairs = []
    stats = {"tier1": 0, "tier2": 0, "tier3": 0, "tier4": 0,
             "tier1_sims": [], "tier3_sims": []}

    # TIER 1: Decision 1 vs 0.5
    for i_hi in bkt[1.0]:
        if not bkt[0.5]:
            break
        best_idx, sim = _find_hardest_pair(all_embeddings[i_hi], all_embeddings, bkt[0.5])
        if best_idx >= 0:
            pairs.append({
                "winner": all_scored_articles[i_hi],
                "loser": all_scored_articles[best_idx],
                "dimension": "decision", "pair_type": "decision_boundary_high",
                "cosine_sim": sim,
            })
            stats["tier1"] += 1
            stats["tier1_sims"].append(sim)

    # TIER 2: Decision 0.5 vs 0
    for i_mid in bkt[0.5]:
        if not bkt[0.0]:
            break
        best_idx, sim = _find_hardest_pair(all_embeddings[i_mid], all_embeddings, bkt[0.0])
        if best_idx >= 0:
            pairs.append({
                "winner": all_scored_articles[i_mid],
                "loser": all_scored_articles[best_idx],
                "dimension": "decision", "pair_type": "decision_boundary_low",
                "cosine_sim": sim,
            })
            stats["tier2"] += 1

    # TIER 3: Decision 1 vs 0
    n_extreme = max(1, int(len(pairs) * 0.15 / 0.85)) if pairs else min(len(bkt[1.0]), len(bkt[0.0]))
    if bkt[1.0] and bkt[0.0]:
        hi_sample = random.sample(bkt[1.0], min(n_extreme, len(bkt[1.0])))
        for i_hi in hi_sample:
            i_lo = random.choice(bkt[0.0])
            sim = cosine_sim(all_embeddings[i_hi], all_embeddings[i_lo])
            pairs.append({
                "winner": all_scored_articles[i_hi],
                "loser": all_scored_articles[i_lo],
                "dimension": "decision", "pair_type": "decision_extreme",
                "cosine_sim": sim,
            })
            stats["tier3"] += 1
            stats["tier3_sims"].append(sim)

    # TIER 4: High similarity, different decision scores
    # Only compute pairwise sim for a random subsample if too many articles
    n = len(all_scored_articles)
    if n > 500:
        # Sample 500 articles for hard negative search
        sample_idx = random.sample(range(n), 500)
    else:
        sample_idx = list(range(n))
    sub_embs = all_embeddings[sample_idx]
    sim_matrix = cosine_sim_matrix(sub_embs)
    for ii in range(len(sample_idx)):
        for jj in range(ii + 1, len(sample_idx)):
            i, j = sample_idx[ii], sample_idx[jj]
            di = all_scored_articles[i]["_decision_score"]
            dj = all_scored_articles[j]["_decision_score"]
            if abs(di - dj) < 0.5:
                continue
            if sim_matrix[ii, jj] <= 0.7:
                continue
            winner_idx = i if di > dj else j
            loser_idx = j if di > dj else i
            pairs.append({
                "winner": all_scored_articles[winner_idx],
                "loser": all_scored_articles[loser_idx],
                "dimension": "decision", "pair_type": "decision_hard_negative",
                "cosine_sim": float(sim_matrix[ii, jj]),
            })
            stats["tier4"] += 1

    return pairs, stats


def build_gold_pairs(gold_articles: List[dict], gold_embeddings: np.ndarray,
                     scored_articles: List[dict],
                     scored_embeddings: np.ndarray) -> Tuple[List[dict], dict]:
    """Gold pairs: all-high winners vs scored losers (0 and 0.5)."""
    pairs = []
    stats = {"gold_extreme": 0, "gold_boundary": 0}

    # Find scored articles with method=0 and method=0.5
    zero_idx = [i for i, a in enumerate(scored_articles) if a["_method_score"] == 0.0]
    half_idx = [i for i, a in enumerate(scored_articles) if a["_method_score"] == 0.5]

    for gi, ga in enumerate(gold_articles):
        mdim = ga["_method_dim"]
        # Gold extreme: gold (1) vs scored method=0
        if zero_idx:
            n_ex = min(2, len(zero_idx))
            for li in random.sample(zero_idx, n_ex):
                pairs.append({
                    "winner": ga, "loser": scored_articles[li],
                    "dimension": mdim, "pair_type": "gold_extreme",
                    "cosine_sim": cosine_sim(gold_embeddings[gi], scored_embeddings[li]),
                })
                stats["gold_extreme"] += 1
        # Gold boundary: gold (1) vs scored method=0.5
        if half_idx:
            n_bd = min(2, len(half_idx))
            for li in random.sample(half_idx, n_bd):
                pairs.append({
                    "winner": ga, "loser": scored_articles[li],
                    "dimension": mdim, "pair_type": "gold_boundary",
                    "cosine_sim": cosine_sim(gold_embeddings[gi], scored_embeddings[li]),
                })
                stats["gold_boundary"] += 1

    return pairs, stats


def build_training_cases_pairs(tc_articles: List[dict],
                               scored_articles: List[dict]) -> List[dict]:
    """Training_cases with rubric_score >= 3 as high-quality examples with notes."""
    pairs = []
    zero_scored = [a for a in scored_articles if a["_method_score"] == 0.0]
    half_scored = [a for a in scored_articles if a["_method_score"] == 0.5]

    for tc in tc_articles:
        if not tc.get("_rubric_score") or tc["_rubric_score"] < 3:
            continue
        mdim = tc["_method_dim"]
        # Pair with zero-scored
        if zero_scored:
            loser = random.choice(zero_scored)
            pairs.append({
                "winner": tc, "loser": loser,
                "dimension": mdim, "pair_type": "gold_extreme",
                "cosine_sim": 0.0,  # not computed
            })
        # Pair with half-scored
        if half_scored:
            loser = random.choice(half_scored)
            pairs.append({
                "winner": tc, "loser": loser,
                "dimension": mdim, "pair_type": "gold_boundary",
                "cosine_sim": 0.0,
            })
    return pairs


# ── Step 3: Reasoning generation ────────────────────────────────────────────

def generate_boundary_reasoning(articles: List[dict], hyps: List[str]) -> Dict[str, str]:
    """Generate reasoning for boundary articles using Haiku, batched."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    hyp_text = "\n".join(f"H{i+1}: {h}" for i, h in enumerate(hyps))
    reasoning_map = {}  # title -> reasoning

    # Collect unique boundary articles needing reasoning
    unique = {}
    for a in articles:
        t = a.get("title", "")
        if t and t not in unique:
            ms = a.get("_method_score", -1)
            ds = a.get("_decision_score", -1)
            mdim = a.get("_method_dim", "unknown")
            if ms in (0, 0.5, 1):
                unique[t] = {"title": t, "score": ms, "dim": mdim,
                             "body": _body(a)[:300]}

    items = list(unique.values())
    batch_size = 10
    print(f"  Generating reasoning for {len(items)} boundary articles...")

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start:batch_start + batch_size]
        article_list = "\n\n".join(
            f"Article {i+1}: \"{a['title']}\"\n"
            f"Score: {a['score']} on {a['dim']}\n"
            f"Excerpt: {a['body'][:200]}"
            for i, a in enumerate(batch)
        )
        prompt = (
            f"For each article below, write ONE sentence explaining why it scores "
            f"{{score}} on {{dimension}}, referencing these K* principles:\n\n"
            f"{hyp_text}\n\n"
            f"Focus on what evidence IS present vs what is MISSING that would "
            f"raise the score. Be specific and concise.\n\n{article_list}\n\n"
            f"Return a JSON array of strings, one reasoning per article."
        )
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # Parse JSON array
            if "```" in text:
                text = text[text.find("["):text.rfind("]") + 1]
            reasons = json.loads(text)
            for i, a in enumerate(batch):
                if i < len(reasons):
                    reasoning_map[a["title"]] = reasons[i]
        except Exception as e:
            print(f"  Warning: reasoning batch failed: {e}")
            for a in batch:
                reasoning_map[a["title"]] = ""
        if batch_start + batch_size < len(items):
            time.sleep(0.5)

    return reasoning_map


# ── Step 4: JSONL formatting ────────────────────────────────────────────────

def _sys_msg(hyps: List[str]) -> str:
    hb = "\n".join(f"  H{i+1}: {h}" for i, h in enumerate(hyps))
    return (
        "You are an expert classifier for experiment-aversion research.\n"
        "Score the Guardian article on 7 dimensions using {0, 0.5, 1} "
        "or -1 if unscored.\n\n"
        f"## K* Hypotheses (validated knowledge)\n{hb}\n\n"
        "## Dimensions\n"
        + ", ".join(DIMS) +
        "\n\nReturn ONLY a JSON object with these 7 keys. "
        "Each value is 0, 0.5, 1, or -1.\n"
        "Use -1 for dimensions without evidence."
    )


def format_jsonl(all_pairs: List[dict], reasoning_map: Dict[str, str],
                 hyps: List[str]) -> Tuple[int, int]:
    """Build 80/20 train/val JSONL. Both sides of each pair become examples."""
    sys_content = _sys_msg(hyps)
    seen = set()
    examples = []

    for pair in all_pairs:
        for side in ("winner", "loser"):
            art = pair[side]
            title = art.get("title", "")
            if not title or title in seen:
                continue
            seen.add(title)

            body = _body(art)[:MAX_BODY]
            user_content = f"Title: {title}\nExcerpt: {body}"

            # Add reasoning for boundary articles if available
            reasoning = reasoning_map.get(title, "")
            notes = art.get("notes", "")
            if notes:
                reasoning = notes  # Expert notes take priority

            scores = dict(art["scores"])
            if reasoning:
                # Include reasoning as part of assistant response
                output = {"scores": scores, "reasoning": reasoning}
                assistant_content = json.dumps(output)
            else:
                assistant_content = json.dumps(scores)

            examples.append({"messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]})

    random.shuffle(examples)
    split = int(len(examples) * 0.8)
    KB_DIR.mkdir(parents=True, exist_ok=True)

    for path, data in [(TRAIN_PATH, examples[:split]),
                       (VAL_PATH, examples[split:])]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")

    return split, len(examples) - split


# ── Main pipeline ───────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Model 2 v2: Embedding-Guided Preference Pair Builder")
    print("=" * 70)

    # ── Load all scored articles ──
    print("\n── Step 1: Loading articles and computing embeddings ──")
    scored_by_method = {}   # mdim -> list of article records
    all_scored = []         # flat list
    all_scored_rows = []    # raw rows for embedding

    for fn, mdim, mcol, dcol in SCORED_FILES:
        rows = _read_csv(fn)
        articles = []
        for r in rows:
            ms = _parse_score(r.get(mcol))
            ds = _parse_score(r.get(dcol))
            if ms is None or not r.get("title"):
                continue
            art = _make_art_record(r, mdim, ms, ds)
            articles.append(art)
        scored_by_method[mdim] = articles
        all_scored.extend(articles)
        print(f"  {fn:25s} -> {mdim:25s}  {len(articles)} articles "
              f"(1={sum(1 for a in articles if a['_method_score']==1)}, "
              f"0.5={sum(1 for a in articles if a['_method_score']==0.5)}, "
              f"0={sum(1 for a in articles if a['_method_score']==0)})")

    # Load gold articles
    gold_articles = []
    for fn, mdim in GOLD_FILES:
        rows = _read_csv(fn)
        for r in rows:
            if not r.get("title"):
                continue
            sc = {d: -1 for d in DIMS}
            sc[mdim] = 1
            sc["decision"] = 1
            art = {
                "title": r.get("title", ""), "body": _body(r),
                "scores": sc, "notes": "",
                "_method_dim": mdim, "_method_score": 1.0, "_decision_score": 1.0,
            }
            gold_articles.append(art)
    print(f"  Gold articles: {len(gold_articles)}")

    # Load Training_cases
    tc_articles = []
    tc_rows = _read_csv("Training_cases.csv")
    for r in tc_rows:
        if not r.get("title"):
            continue
        cat = r.get("method_category", "")
        mdim = CATEGORY_MAP.get(cat)
        if not mdim:
            continue
        rubric = 0
        try:
            rubric = float(r.get("rubric_score_0to5", 0))
        except (ValueError, TypeError):
            pass
        sc = {d: -1 for d in DIMS}
        sc[mdim] = 1
        sc["decision"] = 1
        art = {
            "title": r["title"], "body": _body(r), "scores": sc,
            "notes": r.get("notes", "") if rubric >= 3 else "",
            "_method_dim": mdim, "_method_score": 1.0, "_decision_score": 1.0,
            "_rubric_score": rubric,
        }
        tc_articles.append(art)
    print(f"  Training_cases: {len(tc_articles)} "
          f"({sum(1 for a in tc_articles if a.get('_rubric_score',0) >= 3)} with rubric>=3)")

    # Compute embeddings
    print("\n  Computing embeddings...")
    scored_embs = compute_embeddings(all_scored)
    gold_embs = compute_embeddings(gold_articles)
    print(f"  Scored embeddings shape: {scored_embs.shape}")
    print(f"  Gold embeddings shape: {gold_embs.shape}")

    # Per-method embedding subsets
    method_emb_ranges = {}
    offset = 0
    for fn, mdim, mcol, dcol in SCORED_FILES:
        n = len(scored_by_method[mdim])
        method_emb_ranges[mdim] = (offset, offset + n)
        offset += n

    # ── Step 2: Build pairs ──
    print("\n── Step 2: Building 4-tier pairs ──")
    all_pairs = []
    method_stats = {}

    for mdim, articles in scored_by_method.items():
        start, end = method_emb_ranges[mdim]
        embs = scored_embs[start:end]
        pairs, stats = build_method_pairs(articles, embs, mdim)
        all_pairs.extend(pairs)
        method_stats[mdim] = stats
        print(f"  {mdim:25s}: T1={stats['tier1']} T2={stats['tier2']} "
              f"T3={stats['tier3']} T4={stats['tier4']} total={len(pairs)}")

    # Decision pairs
    dec_pairs, dec_stats = build_decision_pairs(all_scored, scored_embs)
    all_pairs.extend(dec_pairs)
    method_stats["decision"] = dec_stats
    print(f"  {'decision':25s}: T1={dec_stats['tier1']} T2={dec_stats['tier2']} "
          f"T3={dec_stats['tier3']} T4={dec_stats['tier4']} total={len(dec_pairs)}")

    # Gold pairs
    gold_pairs, gold_stats = build_gold_pairs(gold_articles, gold_embs,
                                               all_scored, scored_embs)
    all_pairs.extend(gold_pairs)
    print(f"  Gold pairs: extreme={gold_stats['gold_extreme']} "
          f"boundary={gold_stats['gold_boundary']}")

    # Training_cases pairs
    tc_pairs = build_training_cases_pairs(tc_articles, all_scored)
    all_pairs.extend(tc_pairs)
    print(f"  Training_cases pairs: {len(tc_pairs)}")

    # ── Step 3: Print distribution table ──
    print("\n── Step 3: Pair Distribution ──")
    print(f"\n{'Method':25s} {'T1(1v0.5)':>10s} {'T2(0.5v0)':>10s} "
          f"{'T3(1v0)':>8s} {'T4(hard)':>9s} {'Gold':>6s} {'Total':>7s}")
    print("-" * 80)

    total_row = defaultdict(int)
    dim_keys = list(scored_by_method.keys()) + ["decision"]
    for mdim in dim_keys:
        st = method_stats.get(mdim, {})
        t1 = st.get("tier1", 0)
        t2 = st.get("tier2", 0)
        t3 = st.get("tier3", 0)
        t4 = st.get("tier4", 0)
        gold_n = sum(1 for p in all_pairs
                     if p["dimension"] == mdim and "gold" in p.get("pair_type", ""))
        row_total = t1 + t2 + t3 + t4 + gold_n
        print(f"{mdim:25s} {t1:>10d} {t2:>10d} {t3:>8d} {t4:>9d} {gold_n:>6d} {row_total:>7d}")
        total_row["t1"] += t1
        total_row["t2"] += t2
        total_row["t3"] += t3
        total_row["t4"] += t4
        total_row["gold"] += gold_n
        total_row["total"] += row_total

    print("-" * 80)
    print(f"{'TOTAL':25s} {total_row['t1']:>10d} {total_row['t2']:>10d} "
          f"{total_row['t3']:>8d} {total_row['t4']:>9d} {total_row['gold']:>6d} "
          f"{total_row['total']:>7d}")

    # Hard neg stats
    hard_neg_count = sum(1 for p in all_pairs if "hard_negative" in p.get("pair_type", ""))
    print(f"\nEmbedding-hard pairs found: {hard_neg_count} (most valuable)")

    # Avg cosine similarity comparison
    t1_sims = []
    t3_sims = []
    for mdim in dim_keys:
        st = method_stats.get(mdim, {})
        t1_sims.extend(st.get("tier1_sims", []))
        t3_sims.extend(st.get("tier3_sims", []))
    if t1_sims:
        print(f"Avg cosine sim Tier 1 (boundary) pairs: {np.mean(t1_sims):.4f}")
    if t3_sims:
        print(f"Avg cosine sim Tier 3 (extreme) pairs:  {np.mean(t3_sims):.4f}")
    if t1_sims and t3_sims:
        print(f"Tier 1 similarity is {'HIGHER' if np.mean(t1_sims) > np.mean(t3_sims) else 'lower'} "
              f"than Tier 3 (confirming harder contrasts)")

    # ── Generate boundary reasoning ──
    print("\n── Generating boundary reasoning via Haiku ──")
    hyps = _kstar()
    boundary_articles = []
    for p in all_pairs:
        if "boundary" in p.get("pair_type", "") or "hard_negative" in p.get("pair_type", ""):
            for side in ("winner", "loser"):
                boundary_articles.append(p[side])
    reasoning_map = generate_boundary_reasoning(boundary_articles, hyps)
    print(f"  Reasoning generated for {len(reasoning_map)} articles")

    # ── Step 4: Format JSONL ──
    print("\n── Step 4: Formatting JSONL ──")
    n_train, n_val = format_jsonl(all_pairs, reasoning_map, hyps)
    print(f"  Training examples: {n_train} -> {TRAIN_PATH.name}")
    print(f"  Validation examples: {n_val} -> {VAL_PATH.name}")

    # Token estimate
    total_examples = n_train + n_val
    avg_tokens_per_example = 600  # conservative estimate
    est_tokens = total_examples * avg_tokens_per_example * EPOCHS
    est_cost = est_tokens * 0.0000003  # gpt-4o-mini fine-tuning rate per token
    print(f"  Estimated total tokens: ~{est_tokens:,}")
    print(f"  Estimated fine-tuning cost: ~${est_cost:.2f}")

    # ── Step 5: Print 5 example entries ──
    print("\n── Step 5: Example entries ──")
    # Load back from JSONL
    examples_by_type = {}
    for p in all_pairs:
        pt = p.get("pair_type", "unknown")
        if pt not in examples_by_type:
            examples_by_type[pt] = p

    type_labels = [
        ("method_boundary_high", "Tier 1 (boundary 1v0.5)"),
        ("method_boundary_low", "Tier 2 (floor 0.5v0)"),
        ("method_hard_negative", "Tier 4 (hard negative)"),
        ("gold_extreme", "Gold extreme"),
        ("decision_boundary_high", "Decision boundary"),
    ]
    for pt, label in type_labels:
        pair = examples_by_type.get(pt)
        if not pair:
            print(f"\n--- {label}: No pairs found ---")
            continue
        print(f"\n--- {label} (cosine_sim={pair.get('cosine_sim', 0):.3f}) ---")
        print(f"  Winner: \"{pair['winner']['title'][:80]}\"")
        print(f"    Scores: { {k:v for k,v in pair['winner']['scores'].items() if v != -1} }")
        print(f"  Loser:  \"{pair['loser']['title'][:80]}\"")
        print(f"    Scores: { {k:v for k,v in pair['loser']['scores'].items() if v != -1} }")
        # Show a full JSONL message example for the winner
        r = reasoning_map.get(pair["winner"]["title"], "")
        print(f"  Winner reasoning: \"{r[:120]}...\"" if r else "  (no reasoning)")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(all_pairs)} pairs -> {n_train} train + {n_val} val examples")
    print("=" * 70)

    return all_pairs, n_train, n_val


if __name__ == "__main__":
    main()
