#!/bin/bash
# Auto-push dashboard data to GitHub → triggers Streamlit Cloud redeploy
# Called every 5 rounds by run_query_loop.py

set -e
cd "$(dirname "$0")/.."

echo "$(date): Pushing dashboard data to GitHub..."

# Stage only dashboard-relevant files
git add -f dashboard.py
git add -f requirements.txt
git add -f .streamlit/config.toml
git add -f .gitignore
git add -f scripts/push_to_github.sh

# Output data (manifests, pools, query log)
git add -f outputs/rounds/round_*/round_manifest.json 2>/dev/null || true

# Per-round article files (Tier A, Tier B, scored results)
git add -f outputs/rounds/round_*/round_*_tier_a_papers.csv 2>/dev/null || true
git add -f outputs/rounds/round_*/round_*_tier_b_papers.csv 2>/dev/null || true
git add -f outputs/rounds/round_*/scored_results_full.csv 2>/dev/null || true

# Tier A/B pool files (dashboard reads these for Article page)
git add -f outputs/pools/pool_*_overall.csv 2>/dev/null || true
git add -f outputs/pools/pool_*_candidates.csv 2>/dev/null || true
git add -f outputs/pools/pool_*.csv 2>/dev/null || true
git add -f outputs/query_log.csv 2>/dev/null || true

# State files needed by dashboard
git add -f project_state/POOL_STATUS.json 2>/dev/null || true
git add -f project_state/COST_LEDGER.json 2>/dev/null || true
git add -f project_state/DISCOVERED_TERMS.json 2>/dev/null || true
git add -f project_state/THRESHOLD_HISTORY.json 2>/dev/null || true

# Knowledge base (for Dan's technical details)
git add -f knowledge_base/K_star.json 2>/dev/null || true

# Documentation (for Reproduce Files page)
git add -f CLAUDE.md 2>/dev/null || true
git add -f REVISED_ARCHITECTURE.md 2>/dev/null || true
git add -f DASHBOARD_SPEC.md 2>/dev/null || true
git add -f HANDOFF.md 2>/dev/null || true
git add -f MASTER_PLAN_v3.md 2>/dev/null || true
git add -f docs/versions/*.md 2>/dev/null || true

# Source code (for Reproduce Files page)
git add -f src/*.py 2>/dev/null || true

# Reports
git add -f reports/*.md 2>/dev/null || true

# Commit and push
ROUND_ID=$(ls -d outputs/rounds/round_* 2>/dev/null | sort -t_ -k2 -n | tail -1 | grep -o '[0-9]*$')
git commit -m "Dashboard update: round ${ROUND_ID:-unknown} $(date +%Y-%m-%d_%H:%M)" || {
    echo "Nothing to commit"
    exit 0
}

git push origin main || git push origin master

echo "$(date): Push complete. Streamlit Cloud will redeploy in ~1 minute."
