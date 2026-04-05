# Experiment Aversion Article Finder — Knowledge-Guided Pipeline

## READ IN ORDER:
1. This file → 2. MASTER_PLAN_v3.md §1-§10,§12-§13,§17,§19-§24
3. REVISED_ARCHITECTURE.md (overrides §11,§14-§16,§18)
4. DASHBOARD_SPEC.md → 5. HANDOFF.md

## What changed: first 10 rounds found 0 Tier A, 195 articles, 74% duplicates.
Queries too narrow, filtering too aggressive, models poorly calibrated.
REVISED_ARCHITECTURE.md redesigns everything.

## 6 Active Models (v3 configuration)
- M1: Sonnet temp=0, K*-guided, decision-driven (method threshold disabled)
- M2-old: Feb SFT GPT, K*-FREE, conservative rejector
- M2-new: K*-guided SFT, under monitoring
- M3: Embedding + MLP, K*-FREE, continuous output, decision-driven
- M4: v3 continuous K* features + Ridge, best method classifier (11/15)
- M5: DistilBERT 3-class, continuous, local (zero cost), single relevance voter
- M6: Haiku with 3 K* + sub-questions, independent cheap LLM ($0.08/100 articles)

## Query: (METHOD) AND (DECISION) NOT (GLOBAL + METHOD_SPECIFIC + FULL_METHOD)
Plus Guardian API section filter, order-by alternation, 3-key rotation.
STATIC GLOBAL_EXCLUDE always applied unconditionally + method-specific + full-method at 80%.

## Two tiers: A (AND): ≥3 models agree on HIGH | B (1+ model high)

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
- Set discrete model threshold below 0.25 (admits score=0.5 articles)
- Lower Tier A threshold below 3 without evidence from 30+ rounds
