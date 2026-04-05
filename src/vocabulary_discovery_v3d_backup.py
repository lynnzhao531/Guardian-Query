"""Vocabulary discovery module (v3b, Fix 1).

Dual-LLM AND/OR discovery:
- Haiku (M6) PROPOSES terms as candidate method/decision vocabulary.
- Sonnet (M1) VALIDATES proposed terms with a 0-10 usefulness rating.
- AND gate (both agree) → strong pool, merged directly into METHOD_TERMS.
- OR gate (one agrees)  → trial pool, used at 20% rate for 3 rounds then graduated/dropped.

The discovery stage runs every 5 rounds from round_runner. Inputs are recent
Tier A + top Tier B titles pulled from the pool CSVs.

State is persisted at project_state/DISCOVERED_TERMS.json as:
    {
      "strong": {method: [term, term, ...]},
      "trial":  {method: [{term, trials_remaining, tier_a_produced, added_round}]},
      "dropped": {method: [term, ...]},
      "last_run_round": int,
    }
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STATE_DIR = _PROJECT_ROOT / "project_state"
_STATE_PATH = _STATE_DIR / "DISCOVERED_TERMS.json"
_POOLS_DIR = _PROJECT_ROOT / "outputs" / "pools"

METHODS = ["rct", "prepost", "case_study",
           "expert_qual", "expert_secondary", "gut"]

# ── Stopword / filter list ──────────────────────────────────────────────────
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "has", "have", "had", "will", "would", "could", "should",
    "that", "this", "these", "those", "it", "its", "they", "them", "he",
    "she", "his", "her", "who", "what", "when", "where", "why", "how",
    "uk", "us", "britain", "england", "scotland", "wales", "ireland",
    "london", "says", "said", "new", "old", "one", "two", "three",
    "government", "minister", "ministers", "pm", "mp", "mps", "labour",
    "tory", "tories", "conservative", "lib", "dem", "green",
}

TRIAL_DEFAULT_TRIALS = 3
TRIAL_USAGE_RATE = 0.20   # informational — used by query_builder random-fill
MAX_STRONG_PER_METHOD = 15
MAX_TRIAL_PER_METHOD = 10


# ── State I/O ────────────────────────────────────────────────────────────────

def _empty_state() -> dict:
    return {
        "strong": {m: [] for m in METHODS},
        "trial": {m: [] for m in METHODS},
        "dropped": {m: [] for m in METHODS},
        "last_run_round": 0,
    }


def load_state() -> dict:
    if not _STATE_PATH.exists():
        return _empty_state()
    try:
        with open(_STATE_PATH) as fh:
            s = json.load(fh)
        # Backfill missing keys
        base = _empty_state()
        for k in base:
            if k not in s:
                s[k] = base[k]
            elif isinstance(base[k], dict):
                for m in METHODS:
                    s[k].setdefault(m, [])
        return s
    except Exception as e:
        logger.warning("DISCOVERED_TERMS load failed: %s — starting empty", e)
        return _empty_state()


def save_state(state: dict) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    tmp.replace(_STATE_PATH)


def get_strong_terms() -> Dict[str, List[str]]:
    """Return {method: [term,...]} for strong pool (usable as normal terms)."""
    s = load_state()
    return {m: list(s["strong"].get(m, [])) for m in METHODS}


def get_trial_terms() -> Dict[str, List[str]]:
    """Return {method: [term,...]} for trial pool (used at 20% rate)."""
    s = load_state()
    out = {}
    for m in METHODS:
        out[m] = [t["term"] for t in s["trial"].get(m, [])
                  if t.get("trials_remaining", 0) > 0]
    return out


# ── Candidate extraction ────────────────────────────────────────────────────

def _read_pool_titles(method: str, limit: int = 100) -> List[str]:
    """Return Tier A overall + top Tier B candidate titles for a method."""
    import csv
    titles: List[str] = []
    for name in (f"pool_{method}_overall.csv", f"pool_{method}_candidates.csv"):
        p = _POOLS_DIR / name
        if not p.exists():
            continue
        try:
            with open(p) as fh:
                for row in csv.DictReader(fh):
                    t = (row.get("title") or "").strip()
                    if t:
                        titles.append(t)
        except Exception:
            pass
        if len(titles) >= limit:
            break
    return titles[:limit]


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def extract_candidate_phrases(titles: List[str],
                              existing_terms: set,
                              min_ngram: int = 2,
                              max_ngram: int = 4,
                              min_count: int = 2) -> List[str]:
    """Return candidate ngrams (lowercased) appearing >=min_count times."""
    from collections import Counter
    ngrams: Counter = Counter()
    for title in titles:
        words = [w.lower() for w in _WORD_RE.findall(title)]
        # Drop pure-stopword tails
        for n in range(min_ngram, max_ngram + 1):
            for i in range(len(words) - n + 1):
                seq = words[i:i + n]
                if seq[0] in _STOPWORDS or seq[-1] in _STOPWORDS:
                    continue
                # All-stopword skip
                if all(w in _STOPWORDS for w in seq):
                    continue
                phrase = " ".join(seq)
                if len(phrase) < 6:
                    continue
                if phrase in existing_terms:
                    continue
                ngrams[phrase] += 1

    return [p for p, c in ngrams.most_common(80) if c >= min_count]


# ── LLM classification ──────────────────────────────────────────────────────

def _anthropic_client():
    import os
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env", override=True)
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=30.0)


_HAIKU_MODEL = "claude-haiku-4-5"
_SONNET_MODEL = "claude-sonnet-4-5"


def haiku_classify_batch(phrases: List[str]) -> Dict[str, str]:
    """Haiku proposes a category per phrase. Returns {phrase: category}.

    Categories: method_rct | method_prepost | method_case_study |
                method_expert_qual | method_expert_secondary | method_gut |
                decision | noise
    """
    if not phrases:
        return {}
    try:
        client = _anthropic_client()
    except Exception as e:
        logger.warning("haiku_classify: no client (%s)", e)
        return {p: "noise" for p in phrases}

    prompt = (
        "Classify each phrase (which appeared in UK news article titles about "
        "policy evaluation) into ONE category:\n"
        "  method_rct — randomised/controlled trials\n"
        "  method_prepost — before/after, baseline/follow-up\n"
        "  method_case_study — single-site pilots, lessons learned\n"
        "  method_expert_qual — reviews, panels, inquiries, consultations\n"
        "  method_expert_secondary — observational / admin-data analysis\n"
        "  method_gut — decisions made WITHOUT evidence\n"
        "  decision — policy decision/implementation verbs\n"
        "  noise — too generic, not a useful query term\n\n"
        "Return JSON: {\"phrase\": \"category\", ...}\n\n"
        "Phrases:\n- " + "\n- ".join(phrases[:60])
    )
    try:
        resp = client.messages.create(
            model=_HAIKU_MODEL,
            max_tokens=1200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        txt = resp.content[0].text if resp.content else ""
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return {p: "noise" for p in phrases}
        data = json.loads(m.group(0))
        return {p: str(data.get(p, "noise")).strip().lower() for p in phrases}
    except Exception as e:
        logger.warning("haiku_classify: API failed (%s)", e)
        return {p: "noise" for p in phrases}


def sonnet_validate_batch(phrases_with_cats: List[Tuple[str, str]]) -> Dict[str, Tuple[int, str]]:
    """Sonnet validates: returns {phrase: (rating_0_10, sonnet_category)}.

    rating >= 6 counts as 'agree'.
    """
    if not phrases_with_cats:
        return {}
    try:
        client = _anthropic_client()
    except Exception as e:
        logger.warning("sonnet_validate: no client (%s)", e)
        return {p: (0, c) for p, c in phrases_with_cats}

    block = "\n".join(f"- {p}  (proposed: {c})" for p, c in phrases_with_cats[:60])
    prompt = (
        "You are rating candidate query terms for a UK Guardian search pipeline "
        "that looks for policy-evaluation articles across 6 methods: rct, prepost, "
        "case_study, expert_qual, expert_secondary, gut (decisions without evidence).\n\n"
        "For each phrase, rate 0-10 how useful it would be to search Guardian "
        "with this exact phrase to find articles of the proposed method, and "
        "give your own best category.\n\n"
        "Return JSON: {\"phrase\": {\"rating\": N, \"category\": \"...\"}, ...}\n\n"
        f"Phrases:\n{block}"
    )
    try:
        resp = client.messages.create(
            model=_SONNET_MODEL,
            max_tokens=1800,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        txt = resp.content[0].text if resp.content else ""
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return {p: (0, c) for p, c in phrases_with_cats}
        data = json.loads(m.group(0))
        out = {}
        for p, c in phrases_with_cats:
            entry = data.get(p) or {}
            try:
                r = int(float(entry.get("rating", 0)))
            except Exception:
                r = 0
            sc = str(entry.get("category", c)).strip().lower()
            out[p] = (r, sc)
        return out
    except Exception as e:
        logger.warning("sonnet_validate: API failed (%s)", e)
        return {p: (0, c) for p, c in phrases_with_cats}


# ── Guardian preflight ──────────────────────────────────────────────────────

def _preflight_term(term: str) -> int:
    """Return Guardian hit count for a term, or -1 on error."""
    try:
        from guardian_client import GuardianClient
        g = GuardianClient()
        q = f'"{term}"' if " " in term else term
        return int(g.preflight(q))
    except Exception as e:
        logger.debug("preflight failed for %s: %s", term, e)
        return -1


# ── Main entry point ────────────────────────────────────────────────────────

def run_discovery(round_id: int,
                  existing_method_terms: Dict[str, List[str]],
                  max_phrases_per_round: int = 40) -> dict:
    """Execute one discovery pass. Returns summary dict.

    existing_method_terms: current METHOD_TERMS dict from query_builder.
    """
    state = load_state()
    summary = {
        "round_id": round_id,
        "proposed": 0,
        "accepted_strong": 0,
        "accepted_trial": 0,
        "rejected": 0,
        "by_method": {},
    }

    # Build global "existing terms" set (from METHOD_TERMS + strong pool)
    existing: set = set()
    for m in METHODS:
        for t in existing_method_terms.get(m, []):
            existing.add(t.lower())
        for t in state["strong"].get(m, []):
            existing.add(t.lower())
        for entry in state["trial"].get(m, []):
            existing.add(entry["term"].lower())
        for t in state["dropped"].get(m, []):
            existing.add(t.lower())

    # Collect candidate phrases across all methods (via each method's own pool)
    all_candidates: Dict[str, List[str]] = {}
    for m in METHODS:
        titles = _read_pool_titles(m, limit=150)
        if len(titles) < 5:
            continue
        phrases = extract_candidate_phrases(titles, existing)
        if phrases:
            all_candidates[m] = phrases[:max_phrases_per_round // len(METHODS) + 2]

    # Flatten unique phrases
    flat: List[str] = []
    seen_flat: set = set()
    for m, plist in all_candidates.items():
        for p in plist:
            if p not in seen_flat:
                seen_flat.add(p)
                flat.append(p)
    flat = flat[:max_phrases_per_round]
    summary["proposed"] = len(flat)

    if not flat:
        state["last_run_round"] = round_id
        save_state(state)
        return summary

    logger.info("vocab_discovery: proposing %d phrases", len(flat))

    # Haiku classify
    haiku_out = haiku_classify_batch(flat)

    # Filter noise before Sonnet
    accepted = [(p, cat) for p, cat in haiku_out.items()
                if cat != "noise" and (cat.startswith("method_") or cat == "decision")]
    if not accepted:
        state["last_run_round"] = round_id
        save_state(state)
        return summary

    # Sonnet validate
    sonnet_out = sonnet_validate_batch(accepted)

    # AND/OR classification + Guardian preflight gating
    for phrase, haiku_cat in accepted:
        rating, sonnet_cat = sonnet_out.get(phrase, (0, haiku_cat))
        sonnet_agrees = (rating >= 6) and (sonnet_cat == haiku_cat)
        haiku_agrees = haiku_cat != "noise"

        # Only add to method pools (decision terms aren't tracked here)
        if not haiku_cat.startswith("method_"):
            continue
        method_key = haiku_cat.replace("method_", "")
        if method_key not in METHODS:
            continue

        # Guardian preflight sanity: 20 <= hits <= 5000
        hits = _preflight_term(phrase)
        if hits < 0:
            hits = 500  # assume OK on preflight error
        if hits < 20 or hits > 5000:
            summary["rejected"] += 1
            continue

        if sonnet_agrees and haiku_agrees:
            # AND → strong pool
            if len(state["strong"][method_key]) >= MAX_STRONG_PER_METHOD:
                summary["rejected"] += 1
                continue
            state["strong"][method_key].append(phrase)
            summary["accepted_strong"] += 1
            summary["by_method"].setdefault(method_key, {"strong": 0, "trial": 0})
            summary["by_method"][method_key]["strong"] += 1
            logger.info("  STRONG %s: %s (rating=%d, hits=%d)", method_key, phrase, rating, hits)
        elif rating >= 5:
            # OR → trial pool (v3c Fix 4: tightened from rating>=4 to rating>=5
            # to reject borderline terms like "court hears" / "due to poor")
            if len(state["trial"][method_key]) >= MAX_TRIAL_PER_METHOD:
                summary["rejected"] += 1
                continue
            state["trial"][method_key].append({
                "term": phrase,
                "trials_remaining": TRIAL_DEFAULT_TRIALS,
                "tier_a_produced": 0,
                "added_round": round_id,
                "haiku_category": haiku_cat,
                "sonnet_rating": rating,
                "guardian_hits": hits,
            })
            summary["accepted_trial"] += 1
            summary["by_method"].setdefault(method_key, {"strong": 0, "trial": 0})
            summary["by_method"][method_key]["trial"] += 1
            logger.info("  TRIAL  %s: %s (rating=%d, hits=%d)", method_key, phrase, rating, hits)
        else:
            summary["rejected"] += 1

    state["last_run_round"] = round_id
    save_state(state)
    logger.info("vocab_discovery: proposed=%d strong=%d trial=%d rejected=%d",
                summary["proposed"], summary["accepted_strong"],
                summary["accepted_trial"], summary["rejected"])
    return summary


# ── Trial graduation / attrition (called from round_runner after scoring) ───

def update_trial_results(round_id: int,
                         target_method: str,
                         terms_used: List[str],
                         tier_a_gained: int) -> dict:
    """Decrement trials_remaining for used trial terms; graduate / drop.

    - tier_a_gained > 0 on a round where this trial term was used → graduate
      immediately to strong pool.
    - trials_remaining reaches 0 with no Tier A → drop.
    """
    state = load_state()
    result = {"graduated": [], "dropped": [], "updated": 0}
    if target_method not in METHODS:
        return result

    trial_list = state["trial"].get(target_method, [])
    kept: List[dict] = []
    used_lc = {t.lower() for t in terms_used}

    for entry in trial_list:
        if entry["term"].lower() not in used_lc:
            kept.append(entry)
            continue
        entry["trials_remaining"] = int(entry.get("trials_remaining", 0)) - 1
        entry["tier_a_produced"] = int(entry.get("tier_a_produced", 0)) + int(tier_a_gained)
        result["updated"] += 1

        if entry["tier_a_produced"] > 0:
            # Graduate
            if entry["term"] not in state["strong"][target_method] and \
               len(state["strong"][target_method]) < MAX_STRONG_PER_METHOD:
                state["strong"][target_method].append(entry["term"])
            result["graduated"].append(entry["term"])
            logger.info("  GRADUATED %s: %s", target_method, entry["term"])
        elif entry["trials_remaining"] <= 0:
            # Drop
            state["dropped"][target_method].append(entry["term"])
            result["dropped"].append(entry["term"])
            logger.info("  DROPPED   %s: %s", target_method, entry["term"])
        else:
            kept.append(entry)

    state["trial"][target_method] = kept
    save_state(state)
    return result


# ── GOLD CSV mining (v3c Fix 2) ─────────────────────────────────────────────

# Map of GOLD csv filename → method. Only "all-high" annotated files are mined;
# scored files (case studies.csv, gut.csv, prepost 2.csv, rct 2.csv,
# quantitative.csv, Training_cases.csv) are skipped per v3c spec.
_GOLD_METHOD_CSVS = {
    "casestudy.csv": "case_study",
    "expert_qual.csv": "expert_qual",
    "expert_secondary_quant.csv": "expert_secondary",
    "gut_decision.csv": "gut",
    "prepost.csv": "prepost",
    "rct.csv": "rct",
}


def _read_gold_titles(path: Path) -> List[str]:
    """Read titles from a GOLD CSV, handling oversized fields."""
    import csv
    csv.field_size_limit(2**20)  # raise from default 131072 for long article bodies
    titles: List[str] = []
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                t = (row.get("title") or "").strip()
                if t:
                    titles.append(t)
    except Exception as e:
        logger.warning("gold_mine: failed reading %s: %s", path.name, e)
    return titles


def mine_gold_csvs(round_id: int = 0,
                   max_phrases_per_method: int = 12) -> dict:
    """One-time: extract candidate phrases from hand-annotated GOLD CSVs.

    For each method's GOLD csv:
      1. Read titles (raising csv field size limit for long bodies).
      2. Extract ngram candidates (reusing extract_candidate_phrases).
      3. Dual-LLM classify via Haiku → Sonnet.
      4. Apply the same AND/OR gates as run_discovery (rating>=6 strong,
         rating>=5 trial), plus Guardian preflight (20 <= hits <= 5000).
      5. Persist to DISCOVERED_TERMS.json.

    Returns a summary dict per method.
    """
    from query_builder import METHOD_TERMS

    data_dir = _PROJECT_ROOT / "data"
    state = load_state()

    # Build existing-term set to avoid duplicates
    existing: set = set()
    for m in METHODS:
        for t in METHOD_TERMS.get(m, []):
            existing.add(t.lower())
        for t in state["strong"].get(m, []):
            existing.add(t.lower())
        for entry in state["trial"].get(m, []):
            existing.add(entry["term"].lower())
        for t in state["dropped"].get(m, []):
            existing.add(t.lower())

    summary: Dict[str, Any] = {
        "round_id": round_id,
        "source": "gold_csvs",
        "by_method": {},
        "total_strong": 0,
        "total_trial": 0,
        "total_rejected": 0,
    }

    for fname, method in _GOLD_METHOD_CSVS.items():
        path = data_dir / fname
        if not path.exists():
            logger.warning("gold_mine: missing %s", fname)
            continue
        titles = _read_gold_titles(path)
        if len(titles) < 3:
            logger.info("gold_mine: %s has only %d titles, skipping", fname, len(titles))
            continue

        phrases = extract_candidate_phrases(titles, existing, min_count=2)
        if not phrases:
            # Fall back to min_count=1 for tiny GOLD files
            phrases = extract_candidate_phrases(titles, existing, min_count=1)
        phrases = phrases[:max_phrases_per_method]
        if not phrases:
            continue

        logger.info("gold_mine: %s → %d candidate phrases from %d titles",
                    fname, len(phrases), len(titles))

        haiku_out = haiku_classify_batch(phrases)
        # Force haiku category to match the GOLD file's method (GOLD is ground truth)
        forced = [(p, f"method_{method}") for p in phrases
                  if haiku_out.get(p, "noise") != "noise"]
        if not forced:
            # If haiku marked everything noise, still push through with forced method
            forced = [(p, f"method_{method}") for p in phrases]

        sonnet_out = sonnet_validate_batch(forced)

        m_strong = 0
        m_trial = 0
        m_rej = 0
        for phrase, _cat in forced:
            rating, _sonnet_cat = sonnet_out.get(phrase, (0, f"method_{method}"))
            hits = _preflight_term(phrase)
            if hits < 0:
                hits = 500
            if hits < 20 or hits > 5000:
                m_rej += 1
                continue

            if rating >= 6:
                if len(state["strong"][method]) >= MAX_STRONG_PER_METHOD:
                    m_rej += 1
                    continue
                if phrase not in state["strong"][method]:
                    state["strong"][method].append(phrase)
                    existing.add(phrase.lower())
                    m_strong += 1
                    logger.info("  STRONG %s: %s (rating=%d, hits=%d) [gold]",
                                method, phrase, rating, hits)
            elif rating >= 5:
                if len(state["trial"][method]) >= MAX_TRIAL_PER_METHOD:
                    m_rej += 1
                    continue
                state["trial"][method].append({
                    "term": phrase,
                    "trials_remaining": TRIAL_DEFAULT_TRIALS,
                    "tier_a_produced": 0,
                    "added_round": round_id,
                    "haiku_category": f"method_{method}",
                    "sonnet_rating": rating,
                    "guardian_hits": hits,
                    "source": "gold_mine",
                })
                existing.add(phrase.lower())
                m_trial += 1
                logger.info("  TRIAL  %s: %s (rating=%d, hits=%d) [gold]",
                            method, phrase, rating, hits)
            else:
                m_rej += 1

        summary["by_method"][method] = {
            "strong": m_strong, "trial": m_trial, "rejected": m_rej,
            "titles": len(titles), "phrases": len(phrases),
        }
        summary["total_strong"] += m_strong
        summary["total_trial"] += m_trial
        summary["total_rejected"] += m_rej

    save_state(state)
    return summary


if __name__ == "__main__":
    # Smoke test
    logging.basicConfig(level=logging.INFO)
    from query_builder import METHOD_TERMS
    s = run_discovery(round_id=0, existing_method_terms=METHOD_TERMS)
    print(json.dumps(s, indent=2))
