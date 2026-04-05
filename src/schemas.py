"""Centralized schema definitions for pipeline artifacts.

Every output file that the dashboard or downstream tools reads should have its
columns defined here, so there's exactly one place to update when a field is
added. Today this module defines the round manifest — the atomic per-round
bookkeeping record introduced to fix the "14/33 rounds produced no data"
reliability problem.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


# ── Round manifest ──────────────────────────────────────────────────────────
# Written at the end of every round (success OR crash) at
#   outputs/rounds/round_<id>/round_manifest.json
# via an atomic rename (.tmp → final).
#
# The dashboard reads manifests as its primary source of per-round state.
# query_log.csv is derived from manifests by rebuild_query_log().

MANIFEST_FIELDS: dict[str, type] = {
    "round_id": int,
    "pipeline_version": str,     # 'v1' | 'v2' | 'v3'
    "started_at": str,           # ISO timestamp
    "finished_at": str,          # ISO timestamp (set in finally block)
    "status": str,               # 'completed' | 'crashed' | 'skipped' | 'no_articles' | 'no_data_recorded'
    "failure_stage": object,     # str or None — which step tripped if crashed
    "failure_message": object,   # str or None — first 500 chars of exception
    "target_method": str,
    "base_query": str,
    "final_query": str,
    "query_width_k": int,
    "decision_width": int,
    "guardian_order_by": str,
    "total_available": int,
    "candidates_retrieved": int,
    "candidates_filtered": int,
    "scored_count": int,
    "unique_scored": int,
    "duplicate_scored": int,
    "unique_rate": float,
    "tier_a_count": int,
    "tier_b_count": int,
    "near_miss_count": int,
    "reward_R": float,
    "reward_V": float,
    "goldmine_triggered": bool,
    "phase": str,
    "cost_this_round": float,
    "budget_remaining": float,
    "files_written": list,       # list[str] of relative paths
    "reconstructed": bool,       # True if created by backfill, not a live run
    "reconstructed_from": object,  # str or None — source used during backfill
}

# Defaults applied when a field isn't set yet (used by crashed-round manifests
# so every manifest has every key, even if most are None/0).
MANIFEST_DEFAULTS: dict[str, Any] = {
    "pipeline_version": "",
    "started_at": "",
    "finished_at": "",
    "status": "crashed",
    "failure_stage": None,
    "failure_message": None,
    "target_method": "",
    "base_query": "",
    "final_query": "",
    "query_width_k": 0,
    "decision_width": 0,
    "guardian_order_by": "",
    "total_available": 0,
    "candidates_retrieved": 0,
    "candidates_filtered": 0,
    "scored_count": 0,
    "unique_scored": 0,
    "duplicate_scored": 0,
    "unique_rate": 0.0,
    "tier_a_count": 0,
    "tier_b_count": 0,
    "near_miss_count": 0,
    "reward_R": 0.0,
    "reward_V": 0.0,
    "goldmine_triggered": False,
    "phase": "",
    "cost_this_round": 0.0,
    "budget_remaining": 0.0,
    "files_written": [],
    "reconstructed": False,
    "reconstructed_from": None,
}


def new_manifest(round_id: int) -> dict:
    """Return a manifest dict pre-populated with defaults and the given round_id.

    A round begins with status='crashed' — it's the default that gets overwritten
    to 'completed' only if every stage passes. That way a crash anywhere in the
    round leaves behind an honest manifest rather than nothing.
    """
    m: dict[str, Any] = {"round_id": int(round_id)}
    for k, v in MANIFEST_DEFAULTS.items():
        # Copy mutable defaults to avoid aliasing
        m[k] = list(v) if isinstance(v, list) else v
    return m


# Column order for query_log.csv when it is rebuilt from manifests.
# Keeping this in one place means dashboards, analyses, and exports all agree.
QUERY_LOG_COLUMNS: list[str] = [
    "round_id",
    "pipeline_version",
    "started_at",
    "status",
    "target_method",
    "base_query",
    "final_query",
    "query_width_k",
    "decision_width",
    "guardian_order_by",
    "total_available",
    "candidates_retrieved",
    "candidates_filtered",
    "scored_count",
    "unique_scored",
    "duplicate_scored",
    "unique_rate",
    "tier_a_count",
    "tier_b_count",
    "near_miss_count",
    "reward_R",
    "reward_V",
    "goldmine_triggered",
    "phase",
    "failure_stage",
    "failure_message",
]


def manifest_to_qlog_row(manifest: dict) -> dict:
    """Project a manifest dict onto the query_log column set."""
    return {col: manifest.get(col, "") for col in QUERY_LOG_COLUMNS}
