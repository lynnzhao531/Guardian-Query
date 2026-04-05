"""Query construction system — REVISED_ARCHITECTURE.md §4.

Key change: max 2 AND clauses — (METHOD_UNIQUE[k]) AND (DECISION_COMMON).
Guardian section filter handles garbage exclusion.
Bandit picks width k from {3, 5, 7, 10} controlling how many method terms.
"""
from __future__ import annotations
import hashlib
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── §4.2 Method terms (ranked specific → broad) ─────────────────────────────
# First terms are most specific; bandit picks top-k.
METHOD_TERMS: Dict[str, List[str]] = {
    "rct": [
        "randomised controlled trial", "randomized controlled trial", "RCT",
        "control group", "randomly assigned", "field experiment",
        "clinical trial", "controlled study", "randomisation",
        "trial participants",
    ],
    "prepost": [
        "before and after", "pre-post", "baseline", "follow-up evaluation",
        "post-implementation", "pre-intervention", "pilot evaluation",
        "trial results showed", "follow up study", "outcome measurement",
    ],
    "case_study": [
        "case study", "case studies", "pilot site", "lessons learned",
        "implementation story", "rollout experience", "one council",
        "one city", "pilot area", "local authority pilot",
    ],
    "expert_qual": [
        "expert panel", "independent review", "public consultation",
        "call for evidence", "advisory group", "citizens jury",
        "inquiry recommended", "commission recommended", "taskforce",
        "evidence review",
    ],
    "expert_secondary": [
        "administrative data", "observational study", "regression analysis",
        "econometric", "difference-in-differences", "quasi-experimental",
        "natural experiment", "cohort study", "linked data",
        "statistical analysis",
    ],
    "gut": [
        "without evidence", "no evidence", "despite evidence",
        "ignored evidence", "overruled", "political decision",
        "decided without", "gut feeling", "no data", "despite warnings",
    ],
}

# ── §4.3 Decision terms (common across all methods) ─────────────────────────
DECISION_TERMS: List[str] = [
    "pilot", "rolled out", "rollout", "implemented", "introduced",
    "trialled", "launched", "expanded", "scale up", "approved",
    "scrapped", "mandated", "decided to", "plans to", "will introduce",
    "evaluation",
]

# ── §4.4 Full-method NOT terms (when progress > 0.80) ───────────────────────
# Terms from other methods to exclude when one method is nearly complete.
# Applied as: AND NOT (other_method_unique_terms)
def get_full_method_not(target_method: str, method_progress: Dict[str, float]) -> List[str]:
    """Return NOT terms from other high-progress methods (§4.4)."""
    not_terms = []
    for method, progress in method_progress.items():
        if method == target_method:
            continue
        if progress > 0.80:
            # Add first 3 unique terms from that method
            not_terms.extend(METHOD_TERMS.get(method, [])[:3])
    return not_terms


# ── §4.4b Static NOT terms (Fix 3, always applied) ──────────────────────────
# These are applied UNCONDITIONALLY to every query, regardless of pool
# progress. Previously the full-method NOT only fired at progress>0.80, which
# meant every round since the start had NO NOT filtering. This reintroduces
# the per-round garbage exclusion that CLAUDE.md mandates.
GLOBAL_EXCLUDE = [
    # Court/crime noise
    "sentencing", "prosecution", "defendant", "verdict",
    "convicted", "charged", "magistrate", "crown court",
    "jury", "appeal",
    # Sport noise (belt-and-braces with section filter)
    "tournament", "championship", "premier league",
    "goal", "coach", "match",
    # Culture noise
    "album", "film", "theatre", "premiere", "box office",
]

METHOD_SPECIFIC_EXCLUDE: Dict[str, List[str]] = {
    "expert_secondary": [
        # Exclude RCT terms — expert_secondary is the non-RCT analysis bucket
        "randomized", "randomised", "trial", "rct", "placebo", "control group",
    ],
    "gut": [
        # Exclude any evidence-based framing — gut decisions are the absence
        "trial", "randomized", "randomised", "study", "evaluation",
        "pilot", "impact assessment", "review",
    ],
}


def get_static_not(target_method: str) -> List[str]:
    """Fix 3: static NOT terms that are ALWAYS applied to every query.

    Returns GLOBAL_EXCLUDE + any method-specific excludes for this target.
    """
    terms = list(GLOBAL_EXCLUDE)
    if target_method in METHOD_SPECIFIC_EXCLUDE:
        terms.extend(METHOD_SPECIFIC_EXCLUDE[target_method])
    return terms

# ── §8 Guardian section filter (replaces old garbage exclude terms) ──────────
# These sections are excluded via the Guardian API's section parameter
EXCLUDED_SECTIONS = [
    "sport", "football", "cricket", "rugby-union", "tennis",
    "entertainment", "music", "film", "tv-and-radio", "games",
    "fashion", "food", "travel", "lifeandstyle",
]
SECTION_FILTER = " AND NOT ".join(f"section/{s}" for s in EXCLUDED_SECTIONS)

# Valid order-by values for Guardian API
ORDER_BY_OPTIONS = ["relevance", "newest"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _or_clause(terms: List[str]) -> str:
    """Build parenthesised OR clause, quoting multi-word terms."""
    parts = []
    for t in terms:
        t = t.strip().strip('"')
        if " " in t or "/" in t or "-" in t:
            parts.append(f'"{t}"')
        else:
            parts.append(t)
    return "(" + " OR ".join(parts) + ")"


def _query_hash(q: str) -> str:
    """Short hash for dedup."""
    return hashlib.md5(q.encode()).hexdigest()[:12]


def _jaccard(terms_a: List[str], terms_b: List[str]) -> float:
    """Jaccard similarity between two term lists."""
    a, b = set(t.lower() for t in terms_a), set(t.lower() for t in terms_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ── §4.1 Core query builder (2 AND clauses) ─────────────────────────────────

def build_query(method: str, width_k: int = 5,
                decision_terms: Optional[List[str]] = None,
                not_terms: Optional[List[str]] = None) -> str:
    """Build query: (METHOD_UNIQUE[k]) AND (DECISION_COMMON) [AND NOT (...)]

    Args:
        method: one of the 6 method keys
        width_k: how many method terms to use (3, 5, 7, or 10)
        decision_terms: override decision terms (default: DECISION_TERMS)
        not_terms: additional NOT terms (from §4.4 full-method NOT)
    """
    m_terms = METHOD_TERMS.get(method, [])[:width_k]
    d_terms = decision_terms or DECISION_TERMS

    q = f"{_or_clause(m_terms)} AND {_or_clause(d_terms)}"

    if not_terms:
        q += f" AND NOT {_or_clause(not_terms)}"

    return q


# ── v3c Fix 3: Decision-only query (no method clause) ──────────────────────

def build_decision_only_query(decision_terms: Optional[List[str]] = None,
                              not_terms: Optional[List[str]] = None) -> str:
    """Build a decision-only query: (DECISION_COMMON) AND NOT (...)

    This 7th query type ignores method terms entirely. It is used to surface
    articles where a policy decision/implementation is discussed but no
    method vocabulary from any of our 6 methods appears in the title/body.
    The post-hoc scoring stage is expected to classify the method.
    """
    d_terms = decision_terms or DECISION_TERMS
    q = _or_clause(d_terms)
    if not_terms:
        q += f" AND NOT {_or_clause(not_terms)}"
    return q


# ── §4.6 / §5 Candidate generation with feature vectors ────────────────────

def generate_candidates(
    target_method: str,
    n: int = 30,
    method_progress: Optional[Dict[str, float]] = None,
    discovered_terms: Optional[Dict[str, List[str]]] = None,
    previous_queries: Optional[List[dict]] = None,
    round_num: int = 1,
    method_saturation: Optional[Dict[str, float]] = None,
    trial_terms: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """Produce up to n candidate query dicts for target_method.

    Each dict includes:
        method, width_k, decision_terms, method_terms, query,
        guardian_order_by, features (11-vector for bandit), query_hash.
    """
    progress = method_progress or {}
    saturation = method_saturation or {}
    _target_sat = float(saturation.get(target_method, 0.0))
    # Fix 3: static excludes always on, then add dynamic full-method NOT.
    not_terms = get_static_not(target_method)
    not_terms.extend(get_full_method_not(target_method, progress))
    prev_hashes = set()
    if previous_queries:
        for pq in previous_queries:
            h = pq.get("query_hash") or _query_hash(pq.get("query", ""))
            prev_hashes.add(h)

    candidates: List[Dict[str, Any]] = []
    widths = [3, 5, 7, 10]

    # v3c Fix 3: add ONE decision-only candidate (7th query type). It has no
    # method clause, so width_k=0 and method_terms=[]. The bandit may still
    # pick it via context features; round_runner treats is_decision_only=True
    # by scoring fewer articles (30) and prioritising titles that contain any
    # method term, so the post-hoc consensus can classify the method.
    do_not = list(not_terms)
    do_query = build_decision_only_query(not_terms=do_not or None)
    do_hash = _query_hash(do_query + "relevance" + "DO")
    if do_hash not in prev_hashes:
        target_prog = progress.get(target_method, 0.0)
        overall_prog = sum(progress.values()) / max(len(progress), 1) if progress else 0.0
        phase = 1 if round_num <= 20 else (2 if round_num <= 60 else 3)
        candidates.append({
            "method": target_method,
            "width_k": 0,
            "decision_terms": list(DECISION_TERMS),
            "method_terms": [],
            "not_terms": do_not,
            "query": do_query,
            "guardian_order_by": "relevance",
            "is_decision_only": True,
            "query_type": "decision_only",
            "features": {
                "target_progress": target_prog,
                "overall_progress": overall_prog,
                "round_phase": phase,
                "recent_unique_rate": 0.5,
                "recent_tier_b_rate": 0.1,
                "method_width": 0.0,
                "decision_width": len(DECISION_TERMS) / 20.0,
                "has_full_not": 1.0 if do_not else 0.0,
                "est_log_available": 0.0,
                "method_is_fresh": 1.0 if target_prog < 0.1 else 0.0,
                "query_novelty": 1.0,
                "method_saturation": _target_sat,
            },
            "query_hash": do_hash,
        })

    # Extra discovered terms to mix in (strong + trial pools)
    extra_m_terms = list((discovered_terms or {}).get(target_method, []))
    extra_trial_terms = list((trial_terms or {}).get(target_method, []))

    for width_k in widths:
        for order_by in ORDER_BY_OPTIONS:
            # Standard query
            q = build_query(target_method, width_k, not_terms=not_terms or None)
            qh = _query_hash(q + order_by)

            # §4.7 Duplicate prevention (Jaccard > 0.80)
            m_terms_used = METHOD_TERMS.get(target_method, [])[:width_k]
            is_dup = qh in prev_hashes
            if not is_dup and previous_queries:
                for pq in previous_queries:
                    pm = pq.get("method_terms", [])
                    if _jaccard(m_terms_used, pm) > 0.80:
                        is_dup = True
                        break

            if is_dup:
                continue

            # §5.2 Feature vector (11 features)
            target_prog = progress.get(target_method, 0.0)
            overall_prog = sum(progress.values()) / max(len(progress), 1) if progress else 0.0
            if round_num <= 20:
                phase = 1
            elif round_num <= 60:
                phase = 2
            else:
                phase = 3

            features = {
                "target_progress": target_prog,
                "overall_progress": overall_prog,
                "round_phase": phase,
                "recent_unique_rate": 0.5,     # updated by bandit from history
                "recent_tier_b_rate": 0.1,     # updated by bandit from history
                "method_width": width_k / 10.0,
                "decision_width": len(DECISION_TERMS) / 20.0,
                "has_full_not": 1.0 if not_terms else 0.0,
                "est_log_available": 0.0,      # filled by preflight
                "method_is_fresh": 1.0 if target_prog < 0.1 else 0.0,
                "query_novelty": 0.0 if is_dup else 1.0,
                "method_saturation": _target_sat,
            }

            candidates.append({
                "method": target_method,
                "width_k": width_k,
                "decision_terms": list(DECISION_TERMS),
                "method_terms": m_terms_used,
                "not_terms": not_terms or [],
                "query": q,
                "guardian_order_by": order_by,
                "features": features,
                "query_hash": qh,
            })

            if len(candidates) >= n:
                return candidates

    # Fill remaining with random variations (add discovered terms, vary decision)
    while len(candidates) < n:
        width_k = random.choice(widths)
        order_by = random.choice(ORDER_BY_OPTIONS)
        m_base = METHOD_TERMS.get(target_method, [])[:width_k]

        # Mix in discovered terms occasionally; trial terms used at 20%
        trial_used: List[str] = []
        if extra_trial_terms and random.random() < 0.20:
            trial_used = random.sample(extra_trial_terms, min(2, len(extra_trial_terms)))
            m_terms_used = m_base + trial_used
        elif extra_m_terms and random.random() < 0.3:
            m_terms_used = m_base + random.sample(extra_m_terms, min(2, len(extra_m_terms)))
        else:
            m_terms_used = m_base

        # Vary decision terms
        d_terms = list(DECISION_TERMS)
        if random.random() < 0.3:
            d_terms = random.sample(DECISION_TERMS, max(8, len(DECISION_TERMS) - 4))

        q = f"{_or_clause(m_terms_used)} AND {_or_clause(d_terms)}"
        if not_terms:
            q += f" AND NOT {_or_clause(not_terms)}"

        qh = _query_hash(q + order_by)
        if qh in prev_hashes:
            continue

        target_prog = progress.get(target_method, 0.0)
        overall_prog = sum(progress.values()) / max(len(progress), 1) if progress else 0.0
        phase = 1 if round_num <= 20 else (2 if round_num <= 60 else 3)

        candidates.append({
            "method": target_method,
            "width_k": width_k,
            "decision_terms": d_terms,
            "method_terms": m_terms_used,
            "not_terms": not_terms or [],
            "query": q,
            "guardian_order_by": order_by,
            "features": {
                "target_progress": target_prog,
                "overall_progress": overall_prog,
                "round_phase": phase,
                "recent_unique_rate": 0.5,
                "recent_tier_b_rate": 0.1,
                "method_width": len(m_terms_used) / 10.0,
                "decision_width": len(d_terms) / 20.0,
                "has_full_not": 1.0 if not_terms else 0.0,
                "est_log_available": 0.0,
                "method_is_fresh": 1.0 if target_prog < 0.1 else 0.0,
                "query_novelty": 1.0,
                "method_saturation": _target_sat,
            },
            "query_hash": qh,
            "trial_terms_used": list(trial_used),
        })

    return candidates


# ── §4.6 Preflight with adaptive k adjustment ───────────────────────────────

def apply_preflight(
    candidate: Dict[str, Any],
    guardian_client: Any,
    *,
    min_supply: int = 30,
    max_supply: int = 2500,
) -> Tuple[Dict[str, Any], int, List[dict]]:
    """Run preflight check and adaptively adjust width k.

    Returns (final_candidate, total_available, trace).
    """
    trace: List[dict] = []
    method = candidate["method"]
    width_k = candidate["width_k"]
    order_by = candidate.get("guardian_order_by", "relevance")

    def _pf(q: str, label: str) -> int:
        total = guardian_client.preflight(q)
        trace.append({"level": label, "query": q, "total": total})
        logger.info("preflight [%s] total=%d", label, total)
        return total

    # Initial check
    total = _pf(candidate["query"], "BASE")

    # Broaden: increase k if too few results
    if total < min_supply and width_k < 10:
        for new_k in [k for k in [5, 7, 10] if k > width_k]:
            q = build_query(method, new_k,
                            candidate.get("decision_terms"),
                            candidate.get("not_terms") or None)
            total = _pf(q, f"BROADEN_k{new_k}")
            if total >= min_supply:
                candidate = {**candidate, "query": q, "width_k": new_k,
                             "method_terms": METHOD_TERMS.get(method, [])[:new_k]}
                break

    # Narrow: decrease k if too many results
    if total > max_supply:
        for new_k in [k for k in [7, 5, 3] if k < width_k]:
            q = build_query(method, new_k,
                            candidate.get("decision_terms"),
                            candidate.get("not_terms") or None)
            total = _pf(q, f"NARROW_k{new_k}")
            if total <= max_supply:
                candidate = {**candidate, "query": q, "width_k": new_k,
                             "method_terms": METHOD_TERMS.get(method, [])[:new_k]}
                break

    # Still too wide? Try k=3 with fewer decision terms
    if total > max_supply:
        strict_d = DECISION_TERMS[:8]  # fewer decision terms
        q = build_query(method, 3, decision_terms=strict_d,
                        not_terms=candidate.get("not_terms") or None)
        total = _pf(q, "NARROW_strict")
        if total <= max_supply:
            candidate = {**candidate, "query": q, "width_k": 3,
                         "decision_terms": strict_d,
                         "method_terms": METHOD_TERMS.get(method, [])[:3]}

    # Update estimated log available in features
    import math
    if "features" in candidate:
        candidate["features"]["est_log_available"] = math.log1p(total) / 10.0

    return candidate, total, trace


# ── §4.1 Guardian section filter string ──────────────────────────────────────

def get_section_filter() -> str:
    """Return the section exclusion string for Guardian API calls.
    Use as: section=-sport,-football,-cricket,...
    """
    return ",".join(f"-{s}" for s in EXCLUDED_SECTIONS)


# ── §4.5 Term discovery ─────────────────────────────────────────────────────

def merge_discovered_terms(
    method: str,
    new_terms: List[str],
    max_per_method: int = 5,
) -> List[str]:
    """Add newly discovered terms to the method's pool (max 5 extra)."""
    current = list(METHOD_TERMS.get(method, []))
    added = []
    for t in new_terms:
        if t.lower() not in {c.lower() for c in current} and len(added) < max_per_method:
            current.append(t)
            added.append(t)
    if added:
        METHOD_TERMS[method] = current
        logger.info("Added %d discovered terms to %s: %s", len(added), method, added)
    return added


# ── Test helper ──────────────────────────────────────────────────────────────

def test_queries(n: int = 6) -> None:
    """Print sample queries for manual inspection."""
    for method in METHOD_TERMS:
        for k in [3, 5, 7]:
            q = build_query(method, k)
            print(f"\n[{method}] k={k}:")
            print(f"  {q}")
            print(f"  Length: {len(q)} chars")
            if n <= 0:
                break
        n -= 1
        if n <= 0:
            break


if __name__ == "__main__":
    test_queries()
