"""Backfill round_manifest.json for every round directory that exists.

For each round:
  1. If an audit JSON exists: extract fields from it (this covers rounds 1-18, 23)
  2. If only round files exist: count rows → counts without query text
  3. If the directory is empty: status='no_data_recorded'
  4. If candidates_raw.csv is the only file: status='crashed', failure_stage='score'

Run with: python3 src/backfill_manifests.py

After this finishes, rebuild_query_log() can reconstruct query_log.csv so the
dashboard has a single, consistent source of truth.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schemas import new_manifest  # noqa: E402

ROUNDS = ROOT / "outputs" / "rounds"
AUDITS = ROOT / "audits"


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with open(path) as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)
    except Exception:
        return 0


def _version_for(round_id: int) -> str:
    if round_id <= 10:
        return "v1"
    if round_id <= 30:
        return "v2"
    return "v3"


def _load_audit(round_id: int) -> dict | None:
    p = AUDITS / f"round_{round_id}" / f"round_{round_id}_audit.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _atomic_write(target: Path, data: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    import os
    os.replace(tmp, target)


def backfill_round(round_id: int, round_dir: Path) -> dict:
    """Create a manifest for one round and write it to disk."""
    m = new_manifest(round_id)
    m["pipeline_version"] = _version_for(round_id)
    m["reconstructed"] = True

    files = sorted(f.name for f in round_dir.iterdir()) if round_dir.exists() else []
    m["files_written"] = [f"outputs/rounds/round_{round_id}/{f}" for f in files]

    audit = _load_audit(round_id)

    # Case A: empty round dir AND no audit → never ran
    if not files and not audit:
        m["status"] = "no_data_recorded"
        m["failure_stage"] = None
        m["reconstructed_from"] = "empty"
        _atomic_write(round_dir / "round_manifest.json", m)
        # Make sure the directory exists so the manifest has a home
        round_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(round_dir / "round_manifest.json", m)
        return m

    # Case B: audit JSON present — richest source
    if audit:
        m["reconstructed_from"] = "audit_json"
        m["started_at"] = str(audit.get("timestamp", ""))
        m["finished_at"] = str(audit.get("timestamp", ""))
        m["target_method"] = str(audit.get("target_method", "") or "")
        m["base_query"] = str(audit.get("base_query", "") or "")
        m["final_query"] = str(audit.get("final_query", "") or audit.get("base_query", "") or "")
        m["query_width_k"] = int(audit.get("query_width_k", 0) or 0)
        m["decision_width"] = int(audit.get("decision_width", 0) or 0)
        m["guardian_order_by"] = str(audit.get("guardian_order_by", "") or "")
        m["total_available"] = int(audit.get("total_available", 0) or 0)
        m["candidates_retrieved"] = int(audit.get("candidates_retrieved", 0) or 0)
        m["scored_count"] = int(audit.get("scored_count", 0) or 0)
        m["unique_scored"] = int(audit.get("unique_count", audit.get("unique_scored_count", 0)) or 0)
        m["duplicate_scored"] = int(audit.get("duplicate_count", audit.get("duplicate_scored_count", 0)) or 0)
        try:
            m["unique_rate"] = float(audit.get("unique_rate", 0) or 0)
        except Exception:
            m["unique_rate"] = 0.0
        m["tier_a_count"] = int(audit.get("tier_a_count", 0) or 0)
        m["tier_b_count"] = int(audit.get("tier_b_count", 0) or 0)
        m["near_miss_count"] = int(audit.get("near_miss_count", 0) or 0)
        try:
            m["reward_R"] = float(audit.get("reward_R", 0) or 0)
        except Exception:
            m["reward_R"] = 0.0
        m["goldmine_triggered"] = bool(audit.get("goldmine_triggered", False))
        m["phase"] = str(audit.get("phase", "") or "")
        # Audits only exist for rounds that ran to the audit step — treat as completed
        m["status"] = "completed"
        m["failure_stage"] = None

    # Case C: round has only candidates_raw.csv → crashed before scoring
    elif files == ["candidates_raw.csv"] or (
        "candidates_raw.csv" in files
        and not any("scored" in f or "tier_" in f or "papers" in f for f in files)
    ):
        m["reconstructed_from"] = "candidates_only"
        m["status"] = "crashed"
        m["failure_stage"] = "score"
        m["failure_message"] = "reconstructed: only candidates_raw.csv present"
        m["candidates_retrieved"] = _count_rows(round_dir / "candidates_raw.csv")
        m["total_available"] = m["candidates_retrieved"]

    # Case D: partial files but no audit — patch together from file counts
    else:
        m["reconstructed_from"] = "round_dir_files"
        m["status"] = "completed"  # tentative — they have outputs
        m["failure_stage"] = None
        m["candidates_retrieved"] = _count_rows(round_dir / "candidates_raw.csv")
        ta = round_dir / f"round_{round_id}_tier_a_papers.csv"
        tb = round_dir / f"round_{round_id}_tier_b_papers.csv"
        if ta.exists():
            m["tier_a_count"] = _count_rows(ta)
        if tb.exists():
            m["tier_b_count"] = _count_rows(tb)
        scored = round_dir / "scored_results_full.csv"
        if scored.exists():
            m["scored_count"] = _count_rows(scored)
        # Try to guess target method from per-model file presence
        for method in ("rct", "prepost", "case_study", "expert_qual", "expert_secondary", "gut"):
            if any(method in f for f in files):
                m["target_method"] = method
                break

    _atomic_write(round_dir / "round_manifest.json", m)
    return m


def main() -> None:
    # Include any round that has either a dir OR an audit
    round_ids: set[int] = set()
    for d in ROUNDS.glob("round_*"):
        mo = re.search(r"round_(\d+)", d.name)
        if mo:
            round_ids.add(int(mo.group(1)))
    for d in AUDITS.glob("round_*"):
        mo = re.search(r"round_(\d+)", d.name)
        if mo:
            round_ids.add(int(mo.group(1)))

    statuses: list[str] = []
    for rid in sorted(round_ids):
        round_dir = ROUNDS / f"round_{rid}"
        round_dir.mkdir(parents=True, exist_ok=True)
        manifest = backfill_round(rid, round_dir)
        statuses.append(manifest["status"])
        print(f"Round {rid:>2}: status={manifest['status']:<18} "
              f"scored={manifest['scored_count']:>4} "
              f"A={manifest['tier_a_count']:>2} B={manifest['tier_b_count']:>2} "
              f"src={manifest['reconstructed_from']}")

    counts = Counter(statuses)
    print()
    print(f"Backfilled {len(statuses)} manifests. Status distribution: {dict(counts)}")

    # Rebuild query_log.csv from the new manifests
    try:
        from round_runner import rebuild_query_log
        n = rebuild_query_log()
        print(f"Rebuilt query_log.csv with {n} rows from manifests.")
    except Exception as e:
        print(f"WARN: rebuild_query_log failed: {e}")


if __name__ == "__main__":
    main()
