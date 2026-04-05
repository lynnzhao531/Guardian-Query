"""Round runner — REVISED_ARCHITECTURE.md compliant.

5-model scoring, two-tier consensus, intelligent sampling, goldmine extension.
Saves scored_results_full.csv every round with ALL raw per-model scores.
"""
from __future__ import annotations
import csv, glob, hashlib, json, logging, math, os, random, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "project_state"
OUTPUT_DIR = ROOT / "outputs"
POOLS_DIR = OUTPUT_DIR / "pools"
ROUNDS_DIR = OUTPUT_DIR / "rounds"
COST_LEDGER = STATE_DIR / "COST_LEDGER.json"
POOL_STATUS = STATE_DIR / "POOL_STATUS.json"
PERSISTENCE_PATH = STATE_DIR / "METHOD_PERSISTENCE.json"
RUNBOOK = ROOT / "RUNBOOK.md"

METHODS = ["rct", "prepost", "case_study", "expert_qual", "expert_secondary", "gut"]
MFULL = [f"method_{m}" for m in METHODS]
DIMS = ["decision"] + MFULL
MODEL_KEYS = ["m1", "m2old", "m2new", "m3", "m4", "m5", "m6"]
COST_PER_ARTICLE = 0.005

# §8 Filtering limits
MAX_BODY_CHARS = 15000   # was 6000
MIN_SCORED_FLOOR = 30    # hard floor


def _jload(p, default=None):
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return default if default is not None else {}


def _jsave(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(d, f, indent=2, default=str)


def _budget_remaining():
    ld = _jload(COST_LEDGER, {})
    return 60.0 - ld.get("scoring_spend_usd", 0.0)


def _update_cost(added):
    ld = _jload(COST_LEDGER, {})
    ld["scoring_spend_usd"] = ld.get("scoring_spend_usd", 0.0) + added
    ld["last_updated"] = datetime.now(timezone.utc).isoformat()
    _jsave(COST_LEDGER, ld)


def _url_canon(u):
    return u.strip().rstrip("/").lower()


def _is_live_blog(a):
    """§8: Live blog detection by URL only (not title)."""
    return "/live/" in a.get("url", "").lower()


def _pool_urls(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {_url_canon(r.get("url", r.get("url_canon", ""))) for r in csv.DictReader(f)}


def _append_csv(path, rows, fieldnames):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0].keys()) if rows else ["url", "title"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def rebuild_query_log() -> int:
    """Rebuild outputs/query_log.csv from all round_manifest.json files.

    query_log.csv is a DERIVED artifact: every row comes from a manifest, so
    it is always consistent with on-disk round state. Called automatically
    at the end of every round (in run_round's finally block) and by the
    backfill script.
    """
    from schemas import QUERY_LOG_COLUMNS, manifest_to_qlog_row

    rows = []
    for mpath in sorted(ROUNDS_DIR.glob("round_*/round_manifest.json"),
                        key=lambda p: int(p.parent.name.split("_")[-1])):
        try:
            with open(mpath) as f:
                m = json.load(f)
        except Exception:
            continue
        if m.get("status") == "no_data_recorded":
            # Keep a one-line stub so dashboards can see the gap honestly
            stub = {col: "" for col in QUERY_LOG_COLUMNS}
            stub["round_id"] = m.get("round_id", "")
            stub["status"] = "no_data_recorded"
            rows.append(stub)
            continue
        rows.append(manifest_to_qlog_row(m))

    out = OUTPUT_DIR / "query_log.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QUERY_LOG_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in QUERY_LOG_COLUMNS})
    import os as _os
    _os.replace(tmp, out)
    return len(rows)


# ── Seen URLs tracker (dedup across rounds) ─────────────────────────────────

def _load_seen_urls():
    p = STATE_DIR / "SEEN_URLS.json"
    return set(_jload(p, []))


def _save_seen_urls(urls):
    p = STATE_DIR / "SEEN_URLS.json"
    _jsave(p, list(urls))


# ── Method persistence ──────────────────────────────────────────────────────

def _load_persistence():
    return _jload(PERSISTENCE_PATH, {
        "current_method": None, "round_count": 0,
        "start_round": None, "zero_tier_a_streak": 0
    })


def _save_persistence(p):
    _jsave(PERSISTENCE_PATH, p)


def _select_target(ps, round_id):
    """Method persistence: stick with lowest-progress method for up to 10 rounds.
    Every 5th round: explore a different method."""
    pers = _load_persistence()

    cm = pers.get("current_method")
    if cm and ps.get(f"is_full_{cm}", False):
        cm = None
        pers["current_method"] = None
        pers["round_count"] = 0

    if cm is None or pers.get("round_count", 0) >= 10:
        best_m, best_prog = None, 2.0
        for m in METHODS:
            prog = max(ps.get(f"overall_credit_{m}", 0) / 35,
                       ps.get(f"llm_count_{m}", 0) / 100)
            if prog < 1.0 and prog < best_prog:
                best_prog = prog
                best_m = m
        if best_m is None:
            return random.choice(METHODS), pers
        pers["current_method"] = best_m
        pers["round_count"] = 0
        pers["start_round"] = round_id
        pers["zero_tier_a_streak"] = 0
        logger.info("METHOD PERSISTENCE: targeting %s from round %d", best_m, round_id)

    rc = pers.get("round_count", 0)
    if rc > 0 and rc % 5 == 0:
        others = [m for m in METHODS if m != pers["current_method"]
                  and not ps.get(f"is_full_{m}", False)]
        if others:
            target = random.choice(others)
            pers["round_count"] = rc + 1
            _save_persistence(pers)
            return target, pers

    pers["round_count"] = rc + 1
    _save_persistence(pers)
    return pers["current_method"], pers


# ── Score one article with all models ────────────────────────────────────────

def _score_article(title, body, m1, m2old, m2new, m3, m4, m5, m6, dry_run=False):
    """Score article with all available models. Returns dict of model_key->scores."""
    result = {}
    b = body[:800]

    for mk, fn, skip in [("m1", m1.score_article, dry_run),
                          ("m3", m3.score_article, dry_run),
                          ("m4", m4.score_article, dry_run)]:
        try:
            result[mk] = None if skip else fn(title, b)
        except Exception as e:
            logger.error("%s: %s", mk, e)
            result[mk] = None

    try:
        result["m2old"] = None if dry_run else m2old.score_article(title, b)
    except Exception as e:
        logger.error("m2old: %s", e)
        result["m2old"] = None

    m2new_ok = m2new is not None and m2new.is_available() and not dry_run
    try:
        result["m2new"] = m2new.score_article(title, b) if m2new_ok else None
    except Exception as e:
        logger.error("m2new: %s", e)
        result["m2new"] = None

    # M5 — local DeBERTa, no API cost
    try:
        result["m5"] = None if (m5 is None or dry_run) else m5.score_article(title, b)
    except Exception as e:
        logger.error("m5: %s", e)
        result["m5"] = None

    # M6 — Claude Haiku 4.5
    try:
        result["m6"] = None if (m6 is None or dry_run) else m6.score_article(title, b)
    except Exception as e:
        logger.error("m6: %s", e)
        result["m6"] = None

    return result


# ── §7 Intelligent sampling ─────────────────────────────────────────────────

def _intelligent_sample(articles: List[dict], max_n: int = 100,
                        target_method: str = "rct") -> List[dict]:
    """§7: 40 highest title_score + 30 highest uncertainty + 20 random + 10 rare sections."""
    if len(articles) <= max_n:
        return list(articles)

    # Title score: simple keyword match count
    method_kw = {
        "rct": ["trial", "randomis", "randomiz", "experiment", "control group"],
        "prepost": ["before", "after", "baseline", "follow-up", "evaluation"],
        "case_study": ["case study", "pilot", "lessons", "rollout"],
        "expert_qual": ["review", "consultation", "panel", "inquiry"],
        "expert_secondary": ["data", "analysis", "regression", "observational"],
        "gut": ["without evidence", "no evidence", "despite", "ignored"],
    }
    kws = method_kw.get(target_method, method_kw["rct"])

    for a in articles:
        title_lower = a.get("title", "").lower()
        a["_title_score"] = sum(1 for kw in kws if kw in title_lower)
        # Uncertainty proxy: body length (medium articles = more uncertain)
        body_len = len(a.get("body_text", ""))
        a["_uncertainty"] = 1.0 - abs(body_len - 5000) / 10000.0
        a["_section"] = a.get("sectionId", a.get("section", "unknown"))

    selected = set()
    result = []

    def _add(arts, n):
        added = 0
        for a in arts:
            if id(a) not in selected and added < n:
                selected.add(id(a))
                result.append(a)
                added += 1

    # 40 highest title_score
    by_title = sorted(articles, key=lambda a: a.get("_title_score", 0), reverse=True)
    _add(by_title, 40)

    # 30 highest uncertainty
    by_unc = sorted(articles, key=lambda a: a.get("_uncertainty", 0), reverse=True)
    _add(by_unc, 30)

    # 10 rare sections (least common sections)
    from collections import Counter
    section_counts = Counter(a.get("_section", "unknown") for a in articles)
    rare_sorted = sorted(articles, key=lambda a: section_counts.get(a.get("_section", ""), 999))
    _add(rare_sorted, 10)

    # 20 random
    remaining = [a for a in articles if id(a) not in selected]
    random.shuffle(remaining)
    _add(remaining, max_n - len(result))

    return result


# ── §3.5 scored_results_full.csv fields ─────────────────────────────────────

FULL_RESULTS_FIELDS = [
    "url", "title", "round_id", "tier", "classified_method",
    "article_relevance_score", "models_agreeing_high",
    "disagreement_type", "method_certainty",
]
# Add per-model raw scores
for _mk in MODEL_KEYS:
    for _d in DIMS:
        FULL_RESULTS_FIELDS.append(f"{_mk}_{_d}_score")
        FULL_RESULTS_FIELDS.append(f"{_mk}_{_d}_p1")


def _flatten_result(art_url, art_title, round_id, scores, consensus):
    """Flatten one result into a row for scored_results_full.csv."""
    row = {
        "url": art_url,
        "title": art_title,
        "round_id": round_id,
        "tier": consensus.get("tier", "none"),
        "classified_method": consensus.get("classified_method", ""),
        "article_relevance_score": consensus.get("article_relevance_score", 0),
        "models_agreeing_high": consensus.get("models_agreeing_high", 0),
        "disagreement_type": consensus.get("disagreement_type", ""),
        "method_certainty": consensus.get("method_certainty", ""),
    }
    for mk in MODEL_KEYS:
        ms = scores.get(mk)
        for d in DIMS:
            if ms and isinstance(ms.get(d), dict):
                row[f"{mk}_{d}_score"] = ms[d].get("score", 0)
                row[f"{mk}_{d}_p1"] = ms[d].get("p1", 0)
            else:
                row[f"{mk}_{d}_score"] = 0
                row[f"{mk}_{d}_p1"] = 0
    return row


# ── Query log fields (§11) ──────────────────────────────────────────────────

QLOG_FIELDS = [
    "round_id", "timestamp", "target_method", "base_query", "final_query",
    "query_width_k", "decision_width", "guardian_order_by",
    "total_available", "candidates_retrieved", "scored_count",
    "unique_scored_count", "duplicate_scored_count", "unique_rate",
    "tier_a_count", "tier_b_count", "near_miss_count",
    "goldmine_triggered", "goldmine_pages_extended",
    "phase", "method_persistence_round", "epsilon_override",
    "terms_discovered_this_round",
    "reward_V", "reward_tier_a", "reward_tier_b", "reward_method_credit",
    "reward_unique", "reward_dup", "reward_goldmine", "reward_R",
    "guardian_keys_used", "backoff_level",
]


# ── Pool headers ─────────────────────────────────────────────────────────────

POOL_A_HDR = [
    "url", "title", "classified_method", "tied_methods",
    "confidence", "method_certainty", "credit", "round_id", "source",
    "article_relevance_score", "models_agreeing_high",
]
# §3.2: Tier B includes per-model scores
POOL_B_HDR = [
    "url", "title", "classified_method", "secondary_method",
    "method_certainty", "article_relevance_score",
    "models_agreeing_high", "disagreement_type", "round_id",
] + [f"{mk}_decision_p1" for mk in MODEL_KEYS]


# ── Main orchestrator ────────────────────────────────────────────────────────

def _write_manifest(round_id: int, manifest: dict) -> Path:
    """Atomic write of round_manifest.json: .tmp then rename."""
    rd = ROUNDS_DIR / f"round_{round_id}"
    rd.mkdir(parents=True, exist_ok=True)
    target = rd / "round_manifest.json"
    tmp = rd / "round_manifest.json.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    import os as _os
    _os.replace(tmp, target)
    return target


def run_round(round_id: int, *, dry_run: bool = False) -> dict:
    """Execute one round with 5-model scoring and two-tier consensus.

    Wrapped in try/finally: a round_manifest.json is ALWAYS written, even if
    the round crashes. The manifest's `status` and `failure_stage` fields let
    the dashboard (and any downstream tooling) tell the difference between
    'completed', 'crashed at fetch', 'crashed at score', etc.
    """
    from schemas import new_manifest

    manifest = new_manifest(round_id)
    manifest["pipeline_version"] = "v3"
    manifest["started_at"] = datetime.now(timezone.utc).isoformat()
    manifest["failure_stage"] = "init"

    try:
        import model1_llm_judge as m1
        import model2_old as m2old
        import model3_embedding_classifier as m3
        import model4_v3 as m4           # Layer 6: continuous Ridge (was model4_hypothesis_classifier)
        try:
            import model2_new as m2new
            if not m2new.is_available():
                m2new = None
        except Exception:
            m2new = None
        # M5 — local DeBERTa classifier (Layer 2 continuous)
        try:
            import model5_deberta as m5
        except Exception as _e:
            logger.warning("m5 import failed: %s", _e)
            m5 = None
        # M6 — Claude Haiku 4.5 (Layer 3)
        try:
            import model6_haiku as m6
        except Exception as _e:
            logger.warning("m6 import failed: %s", _e)
            m6 = None

        from consensus import (compute_consensus, compute_credit, load_model_weights,
                               dynamic_threshold_check, _get_tier_a_threshold,
                               MODEL_THRESHOLDS)
        from guardian_client import GuardianClient
        from query_builder import (generate_candidates, apply_preflight,
                                   get_section_filter, METHOD_TERMS,
                                   GLOBAL_EXCLUDE)
        from bandit import Bandit
        import framework_contract as fc
        import numpy as np

        # FIX B: config snapshot for reproducibility / forensic review
        _active_models = ["m1", "m2old", "m3", "m4"]
        if 'm2new' in dir() and m2new is not None:
            _active_models.append("m2new")
        if 'm5' in dir() and m5 is not None:
            _active_models.append("m5")
        if 'm6' in dir() and m6 is not None:
            _active_models.append("m6")
        try:
            _tier_a_min = int(_get_tier_a_threshold())
        except Exception:
            _tier_a_min = 3
        manifest["config_snapshot"] = {
            "models_active": _active_models,
            "thresholds": {mk: dict(MODEL_THRESHOLDS[mk]) for mk in MODEL_THRESHOLDS},
            "tier_a_min_agree": _tier_a_min,
            "global_exclude_terms": len(GLOBAL_EXCLUDE),
        }

        ts = datetime.now(timezone.utc).isoformat()
        rd = ROUNDS_DIR / f"round_{round_id}"
        rd.mkdir(parents=True, exist_ok=True)
        S = {"round_id": round_id, "timestamp": ts}

        # A) Pre-round checks
        manifest["failure_stage"] = "pre_round_checks"
        fc.validate_repo_structure()
        fc.validate_models_exist()
        fc.validate_pool_files_exist_or_create()
        if _budget_remaining() <= 0:
            manifest["status"] = "skipped"
            manifest["failure_stage"] = "budget_exhausted"
            raise RuntimeError("Budget exhausted")

        # B) Target method (persistence)
        manifest["failure_stage"] = "method_selection"
        ps = _jload(POOL_STATUS, {})
        target, pers = _select_target(ps, round_id)

        # Forced method rotation: 1 round per method × 6 methods after baseline
        # is first captured. Baseline is persisted so it survives process
        # restarts; each new round advances the offset by exactly 1.
        # Order is chosen to hit non-rct supply first (rct is SEEN_URL-saturated).
        # Bandit takes over from offset 6+.
        _METHOD_CYCLE = ["prepost", "case_study", "expert_qual",
                         "expert_secondary", "gut", "rct"]
        _ROTATION_FILE = STATE_DIR / "METHOD_ROTATION.json"
        _rot = _jload(_ROTATION_FILE, {})
        if "baseline_round_id" not in _rot:
            _rot["baseline_round_id"] = int(round_id)
            _jsave(_ROTATION_FILE, _rot)
        _idx = int(round_id) - int(_rot["baseline_round_id"])
        if 0 <= _idx < 6:
            forced = _METHOD_CYCLE[_idx]
            logger.info("method rotation: offset %d → forcing target=%s (was %s)",
                        _idx, forced, target)
            target = forced
            pers["current_method"] = forced
        else:
            # STUCK DETECTOR (post-rotation only): if the current target scored 0
            # unique articles in its last 3 attempts, switch to the least-explored
            # method from the recent window. Protects against SEEN_URLS saturation.
            try:
                _recent = sorted(
                    glob.glob(str(ROUNDS_DIR / "round_*" / "round_manifest.json")),
                    key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]),
                )
                _recent_manifests = []
                for _p in _recent[-20:]:
                    try:
                        _recent_manifests.append(json.load(open(_p)))
                    except Exception:
                        pass
                # Count consecutive zero-scored rounds for target (walking backward)
                _zero_streak = 0
                for _m in reversed(_recent_manifests):
                    if _m.get("target_method") == target:
                        if int(_m.get("scored_count", 0) or 0) == 0:
                            _zero_streak += 1
                        else:
                            break
                    # allow gaps (other methods) — only break on a successful attempt
                if _zero_streak >= 3:
                    _counts = {meth: 0 for meth in METHODS}
                    for _m in _recent_manifests:
                        _mt = _m.get("target_method")
                        if _mt in _counts:
                            _counts[_mt] += 1
                    _least = min(METHODS, key=lambda meth: _counts.get(meth, 0))
                    logger.warning(
                        "STUCK DETECTOR: %s had %d consecutive zero-scored rounds; "
                        "switching to least-explored method %s",
                        target, _zero_streak, _least,
                    )
                    manifest["stuck_detector_fired"] = True
                    manifest["stuck_detector_from"] = target
                    manifest["stuck_detector_to"] = _least
                    target = _least
                    pers["current_method"] = _least
            except Exception as _e:
                logger.warning("stuck detector error (non-fatal): %s", _e)

        S["target_method"] = target
        S["persistence_round"] = pers.get("round_count", 0)
        manifest["target_method"] = target

        # C) Generate candidates (new 2-AND-clause system)
        manifest["failure_stage"] = "query_build"
        method_progress = {}
        for m in METHODS:
            method_progress[m] = max(ps.get(f"overall_credit_{m}", 0) / 35,
                                      ps.get(f"llm_count_{m}", 0) / 100)
        cands = generate_candidates(target, n=30, method_progress=method_progress,
                                     round_num=round_id)

        # D) Bandit selects query
        bandit = Bandit()
        bandit.load_state()
        chosen = bandit.select_query(cands, round_num=round_id)
        S["selection_mode"] = chosen.get("_selection_mode", "unknown")
        manifest["base_query"] = chosen.get("query", "")
        manifest["final_query"] = chosen.get("query", "")
        manifest["query_width_k"] = int(chosen.get("width_k", 5))
        manifest["decision_width"] = len(chosen.get("decision_terms", []))

        # E) Preflight
        manifest["failure_stage"] = "preflight"
        guardian = GuardianClient()
        chosen, total, pf_trace = apply_preflight(chosen, guardian)
        if total > 2500:
            logger.warning("total=%d > 2500 after preflight, proceeding with first 5 pages", total)
            # Don't crash — just limit to 5 pages, the results will be from the narrowest query tried
        S.update(base_query=chosen["query"], final_query=chosen["query"],
                 query_width_k=chosen.get("width_k", 5),
                 total_available=total)
        manifest["base_query"] = chosen.get("query", "")
        manifest["final_query"] = chosen.get("query", "")
        manifest["total_available"] = int(total)

        # F) Guardian retrieval with section filter
        manifest["failure_stage"] = "fetch"
        section_filter = get_section_filter()
        order_by = chosen.get("guardian_order_by", "relevance")
        raw = guardian.fetch_pages(chosen["query"], pages=5, page_size=50,
                                   total_available=total,
                                   section=section_filter,
                                   order_by=order_by) if total else []
        _write_csv(rd / "candidates_raw.csv", raw, ["url", "title", "body_text"])
        manifest["files_written"].append(f"outputs/rounds/round_{round_id}/candidates_raw.csv")
        S["candidates_retrieved"] = len(raw)
        manifest["candidates_retrieved"] = len(raw)
        manifest["guardian_order_by"] = order_by

        # G) Filtering (§8)
        manifest["failure_stage"] = "filter"
        sl = slo = sm = 0
        filt = []
        for a in raw:
            if _is_live_blog(a):
                sl += 1
                continue
            body = a.get("body_text", "")
            if not body or not body.strip():
                sm += 1
                continue
            if len(body) > MAX_BODY_CHARS:
                slo += 1
                continue
            filt.append(a)

        # §8: If <30 survive, relax body filter
        if len(filt) < MIN_SCORED_FLOOR:
            relaxed = [a for a in raw if not _is_live_blog(a) and a.get("body_text", "").strip()]
            if len(relaxed) > len(filt):
                logger.info("Relaxing body filter: %d → %d articles", len(filt), len(relaxed))
                filt = relaxed
                slo = 0  # reset long count since we relaxed

        S.update(skipped_live=sl, skipped_long=slo, skipped_missing=sm)
        manifest["candidates_filtered"] = len(filt)

        # H) Dedup against seen URLs
        manifest["failure_stage"] = "dedup"
        seen_urls = _load_seen_urls()
        unique_articles = []
        dup_count = 0
        for a in filt:
            uc = _url_canon(a.get("url", ""))
            if uc in seen_urls:
                dup_count += 1
            else:
                unique_articles.append(a)
                seen_urls.add(uc)
        _save_seen_urls(seen_urls)
        S["duplicate_count"] = dup_count
        S["unique_count"] = len(unique_articles)
        unique_rate = len(unique_articles) / max(len(filt), 1)
        dup_rate = dup_count / max(len(filt), 1)
        manifest["unique_scored"] = len(unique_articles)
        manifest["duplicate_scored"] = dup_count
        manifest["unique_rate"] = float(unique_rate)

        # I) §7 Intelligent sampling
        manifest["failure_stage"] = "sample"
        scored_set = _intelligent_sample(unique_articles, max_n=100, target_method=target)
        S["scored_count"] = len(scored_set)

        # J) Scoring with all models
        manifest["failure_stage"] = "score"
        weights = load_model_weights()
        results = []
        full_rows = []  # §3.5: scored_results_full.csv
        for art in scored_set:
            t, b = art["title"], art.get("body_text", "")
            scores = _score_article(t, b, m1, m2old, m2new, m3, m4, m5, m6, dry_run=dry_run)
            cr = compute_consensus(scores, weights) if not dry_run else {"tier": "none", "article_relevance_score": 0}
            results.append({"url": art["url"], "title": t, "scores": scores, "consensus": cr})
            full_rows.append(_flatten_result(art["url"], t, round_id, scores, cr))
        _update_cost(len(results) * COST_PER_ARTICLE)
        manifest["scored_count"] = len(results)
        manifest["cost_this_round"] = round(len(results) * COST_PER_ARTICLE, 4)

        # §3.5 MANDATORY: Save scored_results_full.csv
        _write_csv(rd / "scored_results_full.csv", full_rows, FULL_RESULTS_FIELDS)
        manifest["files_written"].append(f"outputs/rounds/round_{round_id}/scored_results_full.csv")
        # Also append to global full results
        _append_csv(OUTPUT_DIR / "scored_results_full.csv", full_rows, FULL_RESULTS_FIELDS)

        # K) Per-model high relevance
        active_models = ["m1", "m2old", "m3", "m4"] + (["m2new"] if m2new else []) \
                        + (["m5"] if m5 else []) + (["m6"] if m6 else [])
        for mk in active_models:
            highs = []
            for row in results:
                ms = row["scores"].get(mk)
                if ms and row["consensus"].get(f"{mk}_high", False):
                    highs.append({"url": row["url"], "title": row["title"]})
            _write_csv(rd / f"round_{round_id}_{mk}_papers.csv", highs, ["url", "title"])
            S[f"high_rel_{mk}"] = len(highs)

        # L) Tier A consensus
        manifest["failure_stage"] = "consensus"
        tier_a_hits = [r for r in results if r["consensus"].get("tier") == "A"]
        tier_a_rows = []
        for r in tier_a_hits:
            c = r["consensus"]
            tier_a_rows.append({
                "url": r["url"], "title": r["title"],
                "classified_method": c.get("classified_method", ""),
                "tied_methods": json.dumps(c.get("tied_methods", [])),
                "confidence": c.get("confidence", 0),
                "method_certainty": c.get("method_certainty", ""),
                "article_relevance_score": c.get("article_relevance_score", 0),
                "models_agreeing_high": c.get("models_agreeing_high", 0),
                "which_models_high": json.dumps(c.get("which_models_high", [])),
            })
        _write_csv(rd / f"round_{round_id}_tier_a_papers.csv", tier_a_rows)
        S["tier_a_count"] = len(tier_a_hits)
        manifest["tier_a_count"] = len(tier_a_hits)

        # L2) Tier B candidates
        tier_b_only = [r for r in results if r["consensus"].get("tier") == "B"]
        tier_b_rows = []
        for r in tier_b_only:
            c = r["consensus"]
            row = {
                "url": r["url"], "title": r["title"],
                "classified_method": c.get("classified_method", ""),
                "secondary_method": c.get("secondary_method", ""),
                "method_certainty": c.get("method_certainty", ""),
                "article_relevance_score": c.get("article_relevance_score", 0),
                "models_agreeing_high": c.get("models_agreeing_high", 0),
                "disagreement_type": c.get("disagreement_type", ""),
            }
            # Per-model scores for Tier B pool
            for mk in MODEL_KEYS:
                row[f"{mk}_decision_p1"] = c.get(f"{mk}_decision_p1", 0)
            tier_b_rows.append(row)
        _write_csv(rd / f"round_{round_id}_tier_b_papers.csv", tier_b_rows)
        S["tier_b_count"] = len(tier_b_only)
        manifest["tier_b_count"] = len(tier_b_only)

        # Near-miss count
        threshold = _get_tier_a_threshold()
        near_misses = [r for r in tier_b_only
                       if r["consensus"].get("models_agreeing_high", 0) == threshold - 1]
        S["near_miss_count"] = len(near_misses)
        manifest["near_miss_count"] = len(near_misses)

        # M) Pool updates — Tier A
        manifest["failure_stage"] = "pool_update"
        nct = 0.0
        pmc = {m: 0.0 for m in METHODS}
        for r in tier_a_hits:
            c = r["consensus"]
            credits = compute_credit(c)
            url = _url_canon(r["url"])
            for tm, cv in credits.items():
                sh = tm.replace("method_", "")
                pp = POOLS_DIR / f"pool_{sh}_overall.csv"
                if url not in _pool_urls(pp):
                    _append_csv(pp, [{
                        "url": r["url"], "title": r["title"],
                        "classified_method": c.get("classified_method", ""),
                        "tied_methods": json.dumps(c.get("tied_methods", [])),
                        "confidence": c.get("confidence", 0),
                        "method_certainty": c.get("method_certainty", ""),
                        "credit": cv, "round_id": round_id, "source": "consensus",
                        "article_relevance_score": c.get("article_relevance_score", 0),
                        "models_agreeing_high": c.get("models_agreeing_high", 0),
                    }], POOL_A_HDR)
                    pmc[sh] = pmc.get(sh, 0) + cv
                    nct += cv

        # M2) Pool updates — Tier B candidates (with per-model scores)
        tier_b_new = 0
        for r in tier_b_only:
            c = r["consensus"]
            cm = c.get("classified_method", "")
            sh = cm.replace("method_", "") if cm.startswith("method_") else cm
            if sh not in METHODS:
                continue
            pp = POOLS_DIR / f"pool_{sh}_candidates.csv"
            url = _url_canon(r["url"])
            if url not in _pool_urls(pp):
                brow = {
                    "url": r["url"], "title": r["title"],
                    "classified_method": cm,
                    "secondary_method": c.get("secondary_method", ""),
                    "method_certainty": c.get("method_certainty", ""),
                    "article_relevance_score": c.get("article_relevance_score", 0),
                    "models_agreeing_high": c.get("models_agreeing_high", 0),
                    "disagreement_type": c.get("disagreement_type", ""),
                    "round_id": round_id,
                }
                for mk in MODEL_KEYS:
                    brow[f"{mk}_decision_p1"] = c.get(f"{mk}_decision_p1", 0)
                _append_csv(pp, [brow], POOL_B_HDR)
                tier_b_new += 1

        # Update pool status
        ps = _jload(POOL_STATUS, {})
        for m in METHODS:
            ps[f"overall_credit_{m}"] = ps.get(f"overall_credit_{m}", 0) + pmc.get(m, 0)
            ps[f"is_full_{m}"] = (ps.get(f"overall_credit_{m}", 0) >= 35 or
                                   ps.get(f"llm_count_{m}", 0) >= 100)
        _jsave(POOL_STATUS, ps)
        S.update(new_tier_a_credit=nct, tier_b_new=tier_b_new)

        # N) §6 Goldmine check
        goldmine = False
        goldmine_pages = 0
        if len(scored_set) > 0:
            tb_rate = len(tier_b_only) / len(scored_set)
            ta_rate = len(tier_a_hits) / len(scored_set)
            if tb_rate > 0.25 or ta_rate > 0.10:
                goldmine = True
                logger.info("GOLDMINE: tier_b_rate=%.2f, tier_a_rate=%.2f", tb_rate, ta_rate)
                # TODO: extend pages (up to 20), stop when rate<0.05 or cost>$5
                goldmine_pages = 0  # placeholder for extension
        S["goldmine_triggered"] = goldmine
        S["goldmine_pages_extended"] = goldmine_pages
        manifest["goldmine_triggered"] = bool(goldmine)

        # Update persistence
        pers = _load_persistence()
        if len(tier_a_hits) == 0:
            pers["zero_tier_a_streak"] = pers.get("zero_tier_a_streak", 0) + 1
        else:
            pers["zero_tier_a_streak"] = 0
        if pers.get("zero_tier_a_streak", 0) >= 10:
            pers["current_method"] = None
            pers["round_count"] = 0
            logger.info("METHOD PERSISTENCE: 10 rounds with 0 Tier A, switching")
        _save_persistence(pers)

        # O) §5.3 Reward computation
        manifest["failure_stage"] = "reward"
        reward_results = {
            "unique_scored": len(scored_set),
            "tier_a_count": len(tier_a_hits),
            "tier_b_count": len(tier_b_only),
            "unique_rate": unique_rate,
            "duplicate_rate": dup_rate,
            "goldmine_triggered": goldmine,
            "per_method_credit": pmc,
            "per_method_progress": method_progress,
        }
        R = Bandit.compute_reward(reward_results)
        V = min(1.0, len(scored_set) / 50.0)
        manifest["reward_R"] = float(R)
        manifest["reward_V"] = float(V)

        # Q) Bandit update
        feats = bandit.extract_features(chosen)
        bandit.update(feats, R)
        bandit.save_state()
        S["reward_R"] = R

        # R) Audit
        manifest["failure_stage"] = "audit"
        audit = {**S, "budget_remaining": _budget_remaining(), "pool_status": _jload(POOL_STATUS, {})}
        audit_dir = ROOT / "audits" / f"round_{round_id}"
        audit_dir.mkdir(parents=True, exist_ok=True)
        _jsave(audit_dir / f"round_{round_id}_audit.json", audit)

        # S) Runbook
        phase = bandit.get_phase(round_id)
        manifest["phase"] = str(phase)
        manifest["budget_remaining"] = float(_budget_remaining())
        RUNBOOK.parent.mkdir(parents=True, exist_ok=True)
        with open(RUNBOOK, "a") as f:
            f.write(f"\n## Round {round_id} ({ts})\n"
                    f"- Target: {target} (persistence {pers.get('round_count', 0)}/10)\n"
                    f"- Phase: {phase} | Width k: {chosen.get('width_k', 5)} | Order: {order_by}\n"
                    f"- Scored: {len(results)} | Unique: {len(unique_articles)} | Dup: {dup_count}\n"
                    f"- Tier A: {len(tier_a_hits)} | Tier B: {len(tier_b_only)} | Near-miss: {len(near_misses)}\n"
                    f"- Goldmine: {goldmine} | Reward: {R:.4f}\n"
                    f"- Budget: ${_budget_remaining():.2f}\n")

        # Print round summary
        print(f"\n{'═' * 50}")
        print(f"  ROUND {round_id} COMPLETE — Phase {phase}")
        print(f"{'═' * 50}")
        print(f"Target: {target} (persistence {pers.get('round_count', 0)}/10)")
        print(f"Query k={chosen.get('width_k', 5)}, order={order_by}, available={total}")
        print(f"Scored: {len(results)} | Unique: {len(unique_articles)} | Dup: {dup_count} ({dup_rate:.0%})")
        print(f"Tier A: {len(tier_a_hits)} | Tier B: {len(tier_b_only)} | Near-miss: {len(near_misses)}")
        print(f"Goldmine: {'YES' if goldmine else 'no'} | Reward: {R:.4f}")
        pool_str = "  ".join(f"{m}={ps.get(f'overall_credit_{m}', 0):.1f}/35" for m in METHODS)
        print(f"Pool: {pool_str}")
        model_highs = " ".join(f"{mk}={S.get(f'high_rel_{mk}', 0)}" for mk in active_models)
        print(f"Model highs: {model_highs}")
        print(f"Budget: ${_budget_remaining():.2f} of $60")
        print(f"{'═' * 50}")

        logger.info("Round %d done: scored=%d, tier_a=%d, tier_b=%d, R=%.4f",
                    round_id, len(results), len(tier_a_hits), len(tier_b_only), R)

        # ALL STEPS PASSED
        manifest["status"] = "completed"
        manifest["failure_stage"] = None
        return S

    except Exception as e:
        # Don't re-raise — leave an honest manifest and let the driver continue
        import traceback
        manifest["status"] = manifest.get("status") or "crashed"
        if manifest["status"] == "completed":
            manifest["status"] = "crashed"
        manifest["failure_message"] = str(e)[:500]
        print(f"ROUND {round_id} CRASHED at stage={manifest.get('failure_stage')!r}: {e}")
        traceback.print_exc()
        return {"round_id": round_id, "status": manifest["status"], "error": str(e)}

    finally:
        # ALWAYS write the manifest, even on crash, and rebuild the derived qlog.
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            _write_manifest(round_id, manifest)
        except Exception as werr:
            print(f"WARN: failed to write manifest for round {round_id}: {werr}")
        try:
            rebuild_query_log()
        except Exception as werr:
            print(f"WARN: rebuild_query_log failed: {werr}")
