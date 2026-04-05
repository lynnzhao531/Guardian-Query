"""CLI wrapper: run multiple rounds per MASTER_PLAN_v3.md."""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from round_runner import (
    run_round, _budget_remaining, _jload,
    METHODS, COST_PER_ARTICLE, COST_LEDGER, ROUNDS_DIR, POOL_STATUS,
)

def _load_pool_status():
    return _jload(POOL_STATUS, {})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUERY_LOG = PROJECT_ROOT / "outputs" / "query_log.csv"


def _last_completed_round() -> int:
    """Detect last completed round from outputs/rounds/."""
    if not ROUNDS_DIR.exists():
        return 0
    rounds = []
    for d in ROUNDS_DIR.iterdir():
        if d.is_dir() and d.name.startswith("round_"):
            try:
                rounds.append(int(d.name.split("_")[1]))
            except (ValueError, IndexError):
                pass
    return max(rounds) if rounds else 0


def _all_full(ps: dict) -> bool:
    return all(ps.get(f"is_full_{m}", False) for m in METHODS)


def main():
    parser = argparse.ArgumentParser(description="Experiment-aversion query loop")
    parser.add_argument("--max_rounds", type=int, default=20)
    parser.add_argument("--budget_usd", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from last completed round")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run 1 round with minimal scoring")
    args = parser.parse_args()

    start_round = (_last_completed_round() + 1) if args.resume else 1
    end_round = start_round + (1 if args.dry_run else args.max_rounds)

    logger.info("Starting loop: rounds %d-%d, budget $%.2f, dry_run=%s",
                start_round, end_round - 1, args.budget_usd, args.dry_run)

    total_scored = 0
    total_consensus = 0
    rounds_completed = 0

    # FIX C: Cost guard — hard cap on cumulative cost for rounds >= 44 (v3 era).
    # Reads per-round cost from manifests and stops cleanly if budget exceeded.
    COST_GUARD_LIMIT = float(args.budget_usd)
    COST_GUARD_START_ROUND = 44

    def _v3_cost_spent() -> float:
        # Primary: sum per-round cost from manifests (if populated).
        total = 0.0
        for mp in glob.glob(str(ROUNDS_DIR / "round_*" / "round_manifest.json")):
            try:
                rid_i = int(os.path.basename(os.path.dirname(mp)).split("_")[1])
            except Exception:
                continue
            if rid_i < COST_GUARD_START_ROUND:
                continue
            try:
                m = json.load(open(mp))
            except Exception:
                continue
            total += float(m.get("cost_this_round_usd", 0) or 0)
        if total > 0:
            return total
        # Fallback: COST_LEDGER.scoring_spend_usd (cumulative project-wide).
        try:
            ledger = json.load(open(COST_LEDGER))
            return float(ledger.get("scoring_spend_usd", 0) or 0)
        except Exception:
            return 0.0

    for rid in range(start_round, end_round):
        # FIX C: cost guard check (in addition to budget_remaining)
        spent = _v3_cost_spent()
        if spent >= COST_GUARD_LIMIT:
            logger.warning("COST GUARD: $%.2f spent since round %d, limit $%.2f. Stopping.",
                           spent, COST_GUARD_START_ROUND, COST_GUARD_LIMIT)
            break

        remaining = _budget_remaining()
        est_cost = 100 * COST_PER_ARTICLE
        if remaining < est_cost:
            logger.warning("Budget low ($%.2f < $%.2f est). Stopping.", remaining, est_cost)
            break

        ps = _load_pool_status()
        if _all_full(ps):
            logger.info("All method pools FULL. Stopping.")
            break

        logger.info("=== Round %d ===", rid)
        try:
            summary = run_round(rid, dry_run=args.dry_run)
            total_scored += summary.get("scored_count", 0)
            total_consensus += summary.get("tier_a_count", 0)
            rounds_completed += 1
        except Exception:
            logger.exception("Round %d failed", rid)
            if args.dry_run:
                break
            continue

        # Auto-push to GitHub every 5 rounds (→ Streamlit Cloud redeploy).
        # Failure never crashes the pipeline.
        if rounds_completed > 0 and rounds_completed % 5 == 0 and not args.dry_run:
            try:
                result = subprocess.run(
                    ["bash", "scripts/push_to_github.sh"],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(PROJECT_ROOT),
                )
                if result.returncode == 0:
                    logger.info("  GitHub push: OK")
                else:
                    logger.warning("  GitHub push: FAILED — %s",
                                   (result.stderr or "")[:200])
            except Exception as e:
                logger.warning("  GitHub push: ERROR — %s", e)

    # ── Summary ──
    ps = _load_pool_status()
    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(f"Rounds completed : {rounds_completed}")
    print(f"Articles scored  : {total_scored}")
    print(f"Consensus found  : {total_consensus}")
    print(f"Budget remaining : ${_budget_remaining():.2f}")
    print("\nPool status:")
    for m in METHODS:
        credit = ps.get(f"overall_credit_{m}", 0.0)
        llm = ps.get(f"llm_count_{m}", 0)
        full = ps.get(f"is_full_{m}", False)
        print(f"  {m:20s}  credit={credit:6.1f}/35  llm={llm:3d}/100"
              f"  {'FULL' if full else ''}")
    print("=" * 60)


if __name__ == "__main__":
    main()
