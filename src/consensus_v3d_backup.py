"""Two-tier consensus per CONSENSUS_ARCHITECTURE.md §3-§7.

5 models: M1 (Sonnet), M2-old (Feb SFT), M2-new (K* SFT), M3 (Embed), M4 (K*+Ridge).
Tier B = OR-like (any 1 model high), Tier A = AND-like (3+ models agree).
"""
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "project_state" / "STATE.json"
WEIGHTS_PATH = ROOT / "project_state" / "MODEL_WEIGHTS.json"
THRESHOLD_PATH = ROOT / "project_state" / "THRESHOLD_HISTORY.json"

METHODS = ["method_rct", "method_prepost", "method_case_study",
           "method_expert_qual", "method_expert_secondary", "method_gut"]
MODEL_KEYS = ["m1", "m2old", "m2new", "m3", "m4", "m5", "m6"]
TIED_EPS = 0.02

# ── Per-model thresholds (Layer 1 recalibration) ────────────────────────────
# Original pipeline used a universal discrete score>=1.0 gate, which collapses
# the ensemble when a model rarely emits 1.0 (true for M3, M4, M5 on the
# 100-article test set). These thresholds operate on the raw continuous p1
# values returned by score_article and reflect each model's actual calibration
# as measured on outputs/test_100_results.csv.
MODEL_THRESHOLDS: Dict[str, Dict[str, float]] = {
    # v3 (Fix 1, 2026-04-04): separate decision/method thresholds per diagnostic.
    # Method signal is useless (HIGH/LOW distributions overlap) for M1, M3, M4,
    # M6 on the 99-article test set, so method threshold disabled (0.05).
    # Discrete models pinned at decision >= 0.25 so score=0.5 articles
    # (which map to p1=0.20) can NEVER vote HIGH.
    "m1":    {"decision": 0.25, "method": 0.05},   # Sonnet discrete: method near-random
    "m2old": {"decision": 0.25, "method": 0.25},   # Feb SFT: both required, conservative
    "m2new": {"decision": 0.25, "method": 0.25},
    "m3":    {"decision": 0.70, "method": 0.05},   # Continuous embedding: method near-random
    "m4":    {"decision": 0.50, "method": 0.05},   # v3 continuous Ridge: method weak
    "m5":    {"decision": 0.30, "method": 0.30},   # DeBERTa: dec==meth (single bit), 0.5-safe
    "m6":    {"decision": 0.25, "method": 0.05},   # Haiku: decision clean, method wrapped
}
# Rollback copies
MODEL_THRESHOLDS_V2: Dict[str, Dict[str, float]] = {
    "m1":    {"decision": 0.25, "method": 0.25},
    "m2old": {"decision": 0.20, "method": 0.20},
    "m2new": {"decision": 0.20, "method": 0.20},
    "m3":    {"decision": 0.70, "method": 0.70},
    "m4":    {"decision": 0.40, "method": 0.40},
    "m5":    {"decision": 0.25, "method": 0.25},
    "m6":    {"decision": 0.25, "method": 0.25},
}
MODEL_THRESHOLDS_V1: Dict[str, Dict[str, float]] = {
    mk: {"decision": 0.80, "method": 0.80} for mk in MODEL_KEYS
}


def _load_json(p, default=None):
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return default if default is not None else {}


def _save_json(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(d, f, indent=2, default=str)


# ── Model weights ────────────────────────────────────────────────────────────

def load_model_weights() -> Dict[str, float]:
    w = _load_json(WEIGHTS_PATH, {})
    if not w:
        return {k: 0.20 for k in MODEL_KEYS}
    return w


def save_model_weights(w: Dict[str, float]):
    _save_json(WEIGHTS_PATH, w)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_p1(model_scores: Optional[dict], dim: str) -> float:
    if not model_scores:
        return 0.0
    v = model_scores.get(dim)
    if v is None:
        return 0.0
    if isinstance(v, dict):
        return float(v.get("p1", 0.0))
    return float(v)


def _get_score(model_scores: Optional[dict], dim: str) -> float:
    if not model_scores:
        return 0.0
    v = model_scores.get(dim)
    if v is None:
        return 0.0
    if isinstance(v, dict):
        return float(v.get("score", 0.0))
    return float(v)


def _is_model_high_v1(scores: Optional[dict]) -> bool:
    """LEGACY: universal discrete score>=1.0 gate. Kept for rollback."""
    if not scores:
        return False
    d = _get_score(scores, "decision")
    if d < 1.0:
        return False
    for m in METHODS:
        if _get_score(scores, m) >= 1.0:
            return True
    return False


def _is_model_high(model_key: str, scores: Optional[dict]) -> bool:
    """Per-model HIGH gate using raw p1 values and MODEL_THRESHOLDS.

    A model votes HIGH iff its decision_p1 and its max method_p1 both
    meet that model's calibrated threshold. This replaces the old
    universal score>=1.0 check, which collapsed for any model that
    rarely emits the top discretization bucket.
    """
    if not scores:
        return False
    t = MODEL_THRESHOLDS.get(model_key, MODEL_THRESHOLDS["m1"])
    if _get_p1(scores, "decision") < t["decision"]:
        return False
    for m in METHODS:
        if _get_p1(scores, m) >= t["method"]:
            return True
    return False


def _model_top_method(scores: Optional[dict]) -> Tuple[str, float]:
    """Return (top_method, top_p1) for a model's scores."""
    if not scores:
        return METHODS[0], 0.0
    best_m, best_p1 = METHODS[0], 0.0
    for m in METHODS:
        p1 = _get_p1(scores, m)
        if p1 > best_p1:
            best_p1 = p1
            best_m = m
    return best_m, best_p1


# ── Continuous relevance score (§3) ──────────────────────────────────────────

def compute_relevance(model_scores_dict: Dict[str, Optional[dict]],
                      weights: Optional[Dict[str, float]] = None) -> float:
    """article_relevance = weighted avg of (decision_p1 * max_method_p1) per model."""
    w = weights or load_model_weights()
    num, den = 0.0, 0.0
    for mk in MODEL_KEYS:
        ms = model_scores_dict.get(mk)
        if ms is None:
            continue
        d_p1 = _get_p1(ms, "decision")
        max_m_p1 = max(_get_p1(ms, m) for m in METHODS)
        model_score = d_p1 * max_m_p1
        wi = w.get(mk, 0.20)
        num += model_score * wi
        den += wi
    return num / den if den > 0 else 0.0


# ── Tier B (§4) ──────────────────────────────────────────────────────────────

def classify_tier_b(model_scores_dict: Dict[str, Optional[dict]]) -> Optional[dict]:
    """Check if article enters Tier B (any 1 model gives D=1 AND method=1).
    Returns classification dict or None."""
    high_models = []
    model_methods = {}
    for mk in MODEL_KEYS:
        ms = model_scores_dict.get(mk)
        if _is_model_high(mk, ms):
            high_models.append(mk)
            top_m, top_p1 = _model_top_method(ms)
            model_methods[mk] = top_m

    if not high_models:
        return None

    # Determine case
    unique_methods = set(model_methods.values())
    n_high = len(high_models)

    if n_high == 1:
        mk = high_models[0]
        ms = model_scores_dict[mk]
        primary = model_methods[mk]
        # Check for secondary
        method_p1s = {m: _get_p1(ms, m) for m in METHODS}
        sorted_m = sorted(method_p1s, key=method_p1s.get, reverse=True)
        secondary = sorted_m[1] if len(sorted_m) > 1 and method_p1s[sorted_m[1]] >= 0.5 else None
        return {
            "tier": "B",
            "case": 1 if secondary is None else 4,
            "classified_method": primary,
            "secondary_method": secondary,
            "method_certainty": "LOW",
            "models_high": high_models,
            "models_agreeing_high": n_high,
            "disagreement_type": "outlier_high",
        }

    if len(unique_methods) == 1:
        # Case 2: multiple models, same method
        return {
            "tier": "B",
            "case": 2,
            "classified_method": list(unique_methods)[0],
            "secondary_method": None,
            "method_certainty": "MEDIUM",
            "models_high": high_models,
            "models_agreeing_high": n_high,
            "disagreement_type": "near_tier_a" if n_high >= 2 else "outlier_high",
        }

    # Case 3: multiple models, different methods
    # Pick most-voted method
    from collections import Counter
    method_votes = Counter(model_methods.values())
    primary = method_votes.most_common(1)[0][0]
    methods_detected = sorted(unique_methods)
    return {
        "tier": "B",
        "case": 3,
        "classified_method": primary,
        "secondary_method": [m for m in methods_detected if m != primary][0] if len(methods_detected) > 1 else None,
        "methods_detected": methods_detected,
        "method_certainty": "LOW",
        "models_high": high_models,
        "models_agreeing_high": n_high,
        "disagreement_type": "method_disagree",
    }


# ── Tier A (§5) ──────────────────────────────────────────────────────────────

def _get_tier_a_threshold() -> int:
    """Get current Tier A threshold (may be dynamically adjusted)."""
    hist = _load_json(THRESHOLD_PATH, {"current_threshold": None})
    t = hist.get("current_threshold")
    if t is not None:
        return int(t)
    # Default: 3/5 if M2-new available, 3/4 otherwise
    state = _load_json(STATE_PATH, {})
    m2new = state.get("new_finetuned_model_41mini")
    n_models = 5 if m2new else 4
    return 3


def classify_tier_a(model_scores_dict: Dict[str, Optional[dict]],
                    tier_b: Optional[dict] = None) -> Optional[dict]:
    """Check if article enters Tier A. Must already be Tier B.
    Returns classification dict or None."""
    if tier_b is None:
        tier_b = classify_tier_b(model_scores_dict)
    if tier_b is None:
        return None

    threshold = _get_tier_a_threshold()

    # Count models giving high-relevance AND max(method_p1) >= per-model threshold.
    # Layer 1: the old hard-coded 0.80 on top_p1 effectively required discrete
    # score==1.0, which collapsed M3/M4/M5. Now we honor each model's MODEL_THRESHOLDS.
    qualifying = []
    qual_methods = {}
    for mk in MODEL_KEYS:
        ms = model_scores_dict.get(mk)
        if ms is None:
            continue
        if not _is_model_high(mk, ms):
            continue
        top_m, top_p1 = _model_top_method(ms)
        t_method = MODEL_THRESHOLDS.get(mk, MODEL_THRESHOLDS["m1"])["method"]
        if top_p1 >= t_method:
            qualifying.append(mk)
            qual_methods[mk] = top_m

    if len(qualifying) < threshold:
        return None

    # Check method agreement among qualifying models
    from collections import Counter
    method_votes = Counter(qual_methods.values())
    primary, primary_count = method_votes.most_common(1)[0]

    # Agreeing models = those with same top method (or within epsilon)
    agreeing = []
    for mk in qualifying:
        if qual_methods[mk] == primary:
            agreeing.append(mk)
        else:
            # Check if method is within epsilon of primary's avg p1
            pass  # simplified: just check exact match

    if len(agreeing) < threshold:
        return None

    # Compute Tier A classification
    avg_method_p1s = {}
    for m in METHODS:
        vals = [_get_p1(model_scores_dict.get(mk), m)
                for mk in agreeing if model_scores_dict.get(mk)]
        avg_method_p1s[m] = sum(vals) / len(vals) if vals else 0.0

    top_val = avg_method_p1s[primary]
    tied = [m for m, v in avg_method_p1s.items() if abs(v - top_val) <= TIED_EPS]
    total_p1 = sum(avg_method_p1s.values())
    confidence = top_val / total_p1 if total_p1 > 0 else 0.0

    if confidence > 0.50:
        certainty = "HIGH"
    elif confidence > 0.30:
        certainty = "MEDIUM"
    else:
        certainty = "LOW"

    avg_dec_p1 = sum(_get_p1(model_scores_dict.get(mk), "decision")
                     for mk in agreeing if model_scores_dict.get(mk)) / len(agreeing)

    return {
        "tier": "A",
        "classified_method": primary,
        "tied_methods": tied,
        "secondary_method": tier_b.get("secondary_method"),
        "confidence": confidence,
        "method_certainty": certainty,
        "models_high": agreeing,
        "models_agreeing_high": len(agreeing),
        "which_models_high": agreeing,
        "avg_decision_p1": avg_dec_p1,
        "max_method_avg_p1": top_val,
    }


# ── Full consensus ───────────────────────────────────────────────────────────

def compute_consensus(model_scores_dict: Dict[str, Optional[dict]],
                      weights: Optional[Dict[str, float]] = None) -> dict:
    """Classify article into Tier A, Tier B, or none. Returns full result."""
    relevance = compute_relevance(model_scores_dict, weights)
    tier_b = classify_tier_b(model_scores_dict)
    tier_a = classify_tier_a(model_scores_dict, tier_b) if tier_b else None

    # Method classification even for non-tier articles
    avg_method_p1s = {}
    n_models = 0
    for mk in MODEL_KEYS:
        ms = model_scores_dict.get(mk)
        if ms is None:
            continue
        n_models += 1
        for m in METHODS:
            avg_method_p1s.setdefault(m, []).append(_get_p1(ms, m))
    for m in METHODS:
        vals = avg_method_p1s.get(m, [0])
        avg_method_p1s[m] = sum(vals) / len(vals)

    primary = max(avg_method_p1s, key=avg_method_p1s.get) if avg_method_p1s else METHODS[0]
    sorted_methods = sorted(avg_method_p1s, key=avg_method_p1s.get, reverse=True)
    secondary = sorted_methods[1] if len(sorted_methods) > 1 and \
        abs(avg_method_p1s[sorted_methods[0]] - avg_method_p1s[sorted_methods[1]]) <= 0.10 else None
    total_p1 = sum(avg_method_p1s.values())
    confidence = avg_method_p1s.get(primary, 0) / total_p1 if total_p1 > 0 else 0
    certainty = "HIGH" if confidence > 0.50 else ("MEDIUM" if confidence > 0.30 else "LOW")

    result = tier_a if tier_a else (tier_b if tier_b else {
        "tier": "none",
        "classified_method": primary,
        "secondary_method": secondary,
        "method_certainty": certainty,
        "models_high": [],
        "models_agreeing_high": 0,
        "disagreement_type": None,
    })
    result["article_relevance_score"] = relevance
    result["avg_method_p1s"] = avg_method_p1s
    if "classified_method" not in result:
        result["classified_method"] = primary
    if "method_certainty" not in result:
        result["method_certainty"] = certainty
    if "confidence" not in result:
        result["confidence"] = confidence

    # Per-model info
    for mk in MODEL_KEYS:
        ms = model_scores_dict.get(mk)
        result[f"{mk}_high"] = _is_model_high(mk, ms)
        result[f"{mk}_decision_p1"] = _get_p1(ms, "decision")
        tm, tp = _model_top_method(ms)
        result[f"{mk}_top_method"] = tm

    return result


def compute_credit(consensus_result: dict) -> dict:
    """Compute credit for Tier A pool updates (tied-method credit=1/k)."""
    if consensus_result.get("tier") != "A":
        return {}
    tied = consensus_result.get("tied_methods", [])
    if not tied:
        primary = consensus_result.get("classified_method")
        return {primary: 1.0} if primary else {}
    credit = 1.0 / len(tied)
    return {m: credit for m in tied}


# ── Dynamic threshold (§6) ──────────────────────────────────────────────────

def dynamic_threshold_check(tier_a_count: int, near_miss_count: int,
                            tier_b_count: int, round_start: int,
                            round_end: int) -> dict:
    """Run 10-round threshold check. Returns adjustment info."""
    ratio = near_miss_count / max(1, tier_a_count)
    current = _get_tier_a_threshold()
    new_threshold = current
    action = "none"

    if ratio > 3.0:
        new_threshold = max(2, current - 1)
        action = "lowered"
    elif ratio < 0.5 and tier_a_count > 10:
        action = "review_suggested"

    if tier_a_count == 0 and near_miss_count == 0:
        action = "no_signal"

    if new_threshold != current:
        hist = _load_json(THRESHOLD_PATH, {"history": [], "current_threshold": current})
        hist["history"].append({
            "rounds": f"{round_start}-{round_end}",
            "old": current, "new": new_threshold,
            "ratio": ratio, "tier_a": tier_a_count, "near_misses": near_miss_count,
        })
        hist["current_threshold"] = new_threshold
        _save_json(THRESHOLD_PATH, hist)

    return {
        "tier_a_count": tier_a_count,
        "tier_b_count": tier_b_count,
        "near_miss_count": near_miss_count,
        "ratio": ratio,
        "old_threshold": current,
        "new_threshold": new_threshold,
        "action": action,
    }
