# Experiment Aversion Article Finder — Knowledge-Guided Pipeline

## READ IN ORDER:
1. This file → 2. MASTER_PLAN_v3.md §1-§10,§12-§13,§17,§19-§24
3. REVISED_ARCHITECTURE.md (overrides §11,§14-§16,§18)
4. DASHBOARD_SPEC.md → 5. HANDOFF.md

## What changed: first 10 rounds found 0 Tier A, 195 articles, 74% duplicates.
Queries too narrow, filtering too aggressive, models poorly calibrated.
REVISED_ARCHITECTURE.md redesigns everything.

## 5 Models
- M1: Sonnet (temp=0), K*-guided, aggressive → later: decomposed sub-questions
- M2-old: Feb SFT GPT, K*-FREE, conservative (valuable for rejection)
- M2-new: K*-guided SFT — DIAGNOSE before excluding (§2.1)
- M3: Embedding ONLY (K* REMOVED), class weights → later: contrastive fine-tuning
- M4: K* + keywords + Ridge, decision mapping FIXED → later: continuous features
- M5 (future): DeBERTa, K*-free, local, zero API cost

## Query: (METHOD) AND (DECISION) NOT (GLOBAL + METHOD_SPECIFIC + FULL_METHOD)
Plus Guardian API section filter, order-by alternation, 3-key rotation.

## Two tiers: A (3+ models agree, dynamic threshold) | B (1+ model high)

## Bandit: 3-phase curriculum, 11 features, 6-component reward

## NEVER
- Use K* features in Model 3
- Execute total_available > 2500 or omit NOT clause
- Score < 30 articles per round
- Skip saving scored_results_full.csv
- Exclude M2-new without 10+ rounds, 500+ articles of evidence
- Run DPO without reading DPO_GUARDS.md
- Overwrite model files without saving backup to _v2_backup/
- Make permanent decisions from < 200 scored articles
