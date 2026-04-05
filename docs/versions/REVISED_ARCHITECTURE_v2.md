# REVISED ARCHITECTURE — Complete System Specification
## OVERRIDES MASTER_PLAN_v3.md §11 (bandit), §14-§16 (consensus), §18 (queries)
## MASTER_PLAN_v3.md §1-§10, §12-§13, §17 steps A-O, §19-§24 remain valid

# §1. WHY WE CHANGED

First 10 rounds: 0 Tier A, 195 articles, 74% duplicate rate.
Root causes: queries too narrow (4-5 AND clauses → 40 articles/round),
filtering too aggressive (6000-char limit removed best candidates),
9/10 rounds locked on RCT, 3 of 5 models produced zero signal.
Analysis of 60 prior query rounds: simple queries (≤2 AND) found 28x
more consensus articles than complex queries (>2 AND).

# §2. THE 5 MODELS

| ID | Engine | K*? | Status | Fix |
|---|---|---|---|---|
| M1 | Claude Sonnet (temp=0) | Yes | Active, aggressive | Decompose to 4 sub-questions (Step 2 of retraining) |
| M2-old | ft:gpt-4.1-mini (Feb) | No | Active, conservative | None — 100% LOW accuracy is valuable for rejection |
| M2-new | New fine-tuned gpt-4.1-mini | Yes | DIAGNOSE (see §2.1) | Run diagnostics before any exclusion |
| M3 | sentence-transformers + MLP | K*-FREE | Active, retrain | Remove K* features, add class weights, later contrastive fine-tune |
| M4 | K* + keyword + Ridge | Yes | Active, fix mapping | Decision mapping broken (see §2.3) |

## §2.1 M2-new Diagnostic Protocol

M2-new scored 0 on ~20 test articles. Too small a sample to exclude.
Possible causes: prompt mismatch, JSON parsing bug, temperature issue.

REQUIRED before exclusion:
1. Compare system prompt in training JSONL vs scoring code — must be IDENTICAL
2. Print RAW API response for 3 articles before JSON parsing
3. Test temperature=0 vs temperature=0.7
4. Score 10 articles M1 flagged HIGH — any non-zero scores?
5. If bug found: FIX and keep. If conservative: keep. If 0 after fixes + 500 articles + 10 rounds: then exclude with evidence.

Keep M2-new in scoring loop. Track m2new_diagnostic_status per round.

## §2.2 M3 Rebuild

Remove ALL K* hypothesis features (M3 and M4 agreed 100% when both used K*).
Use ONLY 6 cosine similarities to method-specific prototype centroids.
Retrain MLP regressors with class weights: 3x HIGH, 2x MID, 1x LOW.
LATER: contrastive fine-tuning (Step 3 of retraining).

## §2.3 M4 Decision Mapping Fix

Raw prediction in [0,1] was mapped using 0-5 scale logic. Fix:
  raw < 0.25 → discrete=0, p0=0.80, p05=0.15, p1=0.05
  0.25 ≤ raw < 0.75 → discrete=0.5, p0=0.20, p05=0.60, p1=0.20
  raw ≥ 0.75 → discrete=1, p0=0.05, p05=0.15, p1=0.80
Apply to ALL 7 dimensions.

## §2.4 Future Model 5 — DeBERTa (trained during retraining phase)

Fine-tuned DeBERTa/DistilBERT classifier. Completely K*-free and prompt-free.
Learns sequential token patterns. Detects RELEVANCE but not method type.
Independent from all other models. Runs locally, zero API cost.

## §2.5 DPO Policy

DPO is DEFERRED. If considered in future, read project_state/DPO_GUARDS.md
which contains 8 guards against over-optimization. Never run DPO without
following all 8 guards. SFT + K* in prompt provides sufficient regularization
per Wang et al. 2025 findings.

# §3. TWO-TIER CONSENSUS

## §3.1 Continuous Relevance Score

```
model_score_i = decision_p1_i × max(method_p1_across_6_methods_i)
article_relevance = weighted_avg(model_scores, weights=MODEL_WEIGHTS)
```

Weights start equal. Every 10 rounds recompute:
  w_i = 1 / (1 + max_correlation_with_any_other_model)
  Normalize to sum=1. Save to MODEL_WEIGHTS.json.

## §3.2 Tier B (OR): ≥1 model gives D=1 AND any method=1

Classification cases:
- CASE 1: 1 model, 1 method → certainty=LOW
- CASE 2: multiple models, same method → certainty=MEDIUM, flag "near_tier_a"
- CASE 3: models disagree on method → AMBIGUOUS, store in ALL relevant method pools
- CASE 4: 1 model, multiple methods → primary=highest, note secondary
- CASE 5: decision disagreement → flag "decision_split"

Tier B pool files (6): outputs/pools/pool_METHOD_candidates.csv
Columns:
```
url_canon, title, date, section, body_excerpt,
decision_avg_p1, max_method_avg_p1, article_relevance_score,
classified_method, secondary_method, method_certainty (HIGH/MEDIUM/LOW),
m1_high, m2old_high, m2new_high, m3_high, m4_high,
m1_decision_p1, m2old_decision_p1, m2new_decision_p1, m3_decision_p1, m4_decision_p1,
m1_top_method, m2old_top_method, m2new_top_method, m3_top_method, m4_top_method,
models_agreeing_high, disagreement_type, round_id, promoted_to_tier_a
```

## §3.3 Tier A (AND): ≥3 of active models agree (dynamic)

Entry: in Tier B + ≥3 models say HIGH + agree on method (epsilon=0.02).
Starting threshold: 3/5 (or 3/4 if M2-new excluded).
Computes: classified_method, tied_methods, confidence, method_certainty.
Credit: 1/k for tied methods (k=number of tied methods, Option A from old doc).

Tier A pools (12): pool_METHOD_overall.csv + pool_METHOD_LLM.csv

## §3.4 Dynamic Threshold (every 10 rounds)

near_miss = Tier B articles with (threshold - 1) agreeing models.
ratio = near_miss / max(1, tier_a_count).
If ratio > 3.0: LOWER threshold by 1. If ratio < 0.5 and tier_a > 10: FLAG review.

## §3.5 Tier Promotion (B → A)

When an article already in Tier B is re-scored in a later round:
- Combine old scores + new model scores (keep highest per model)
- Re-evaluate: does it now meet Tier A threshold?
- If yes: add to Tier A pools, mark promoted_to_tier_a=true in Tier B pool
- This captures articles that build consensus across rounds

Implementation: before scoring each round, check if any candidates_raw articles
exist in Tier B pools (match by url_canon). If yes, carry forward their existing
model scores and supplement with new scores.

## §3.6 Per-Round Output Files

```
outputs/rounds/round_x/
  candidates_raw.csv
  candidates_filtered.csv
  scored_set.csv
  scored_results_full.csv         ← RAW per-model scores, MANDATORY
  round_x_m1_papers.csv
  round_x_m2old_papers.csv
  round_x_m2new_papers.csv        (skip if M2-new excluded)
  round_x_m3_papers.csv
  round_x_m4_papers.csv
  round_x_m5_papers.csv           (when M5 active)
  round_x_tier_a_papers.csv
  round_x_tier_b_papers.csv
  agreement_report.json
  audits/round_x_audit.md
  audits/round_x_contract.json
```

scored_results_full.csv columns (EVERY scored article):
```
url_canon, title, date, section, body_length,
m1_decision_score, m1_decision_p0, m1_decision_p05, m1_decision_p1,
m1_rct_p1, m1_prepost_p1, m1_case_study_p1, m1_expert_qual_p1,
m1_expert_secondary_p1, m1_gut_p1, m1_max_method_p1, m1_top_method, m1_high,
m2old_decision_score, m2old_decision_p1, m2old_max_method_p1, m2old_top_method, m2old_high,
m2new_decision_score, m2new_decision_p1, m2new_max_method_p1, m2new_top_method, m2new_high,
m3_decision_score, m3_decision_p1, m3_max_method_p1, m3_top_method, m3_high,
m4_decision_score, m4_decision_p1, m4_max_method_p1, m4_top_method, m4_high,
m5_decision_score, m5_decision_p1, m5_max_method_p1, m5_top_method, m5_high,
article_relevance_score, tier, classified_method, secondary_method, method_certainty,
models_agreeing_high, which_models_high, disagreement_type, round_id
```

18 cumulative pools: 12 Tier A + 6 Tier B. No duplicate url_canon within any pool.

# §4. QUERY SYSTEM

## §4.1 Query Structure: 2 AND + 1 NOT

```
(METHOD_UNIQUE terms, width k)
AND (DECISION_COMMON terms)
NOT (GLOBAL_EXCLUDE + METHOD_SPECIFIC_EXCLUDE + FULL_METHOD_TERMS)
```

Guardian API section filter: -football,-sport,-film,-music,-books,-games,-fashion
Guardian order-by: "relevance" (even rounds) / "newest" (odd rounds)
3 Guardian API keys in .env, rotate on 429 or quota errors.

## §4.2 Method Term Bundles (variable width k ∈ {3,5,7,10})

Bandit chooses k. Query uses first k terms from ranked list.

RCT: "randomised controlled trial", "randomized controlled trial", "RCT",
  "control group", "randomly assigned", "field experiment", "clinical trial",
  "controlled study", "randomisation"/"randomization", "trial participants"
NOTE: "trial" alone is COMPOUND ONLY — must appear with "randomised" OR
"randomized" OR "placebo" OR "control group". Never bare "trial" (court cases).

Pre-post: "before and after", "pre-post", "baseline", "follow-up evaluation",
  "post-implementation", "pre-intervention", "pilot evaluation",
  "trial results showed", "follow up study", "outcome measurement"

Case study: "case study", "case studies", "pilot site", "lessons learned",
  "implementation story", "rollout experience", "one council", "one city",
  "pilot area", "local authority pilot"

Expert qualitative: "expert panel", "independent review", "public consultation",
  "call for evidence", "advisory group", "citizens jury", "inquiry recommended",
  "commission recommended", "taskforce", "evidence review"

Expert secondary: "administrative data", "observational study", "regression analysis",
  "econometric", "difference-in-differences", "quasi-experimental",
  "natural experiment", "cohort study", "linked data", "statistical analysis"

Gut: "without evidence", "no evidence", "despite evidence", "ignored evidence",
  "overruled", "political decision", "decided without", "gut feeling",
  "no data", "despite warnings"
NOTE: For gut, the ABSENCE of rigorous evidence IS the signal. Articles about
decisions made without method are relevant TO this category.

## §4.3 Decision Terms (COMMON, shared)

"pilot" OR "rolled out" OR "rollout" OR "implemented" OR "introduced" OR
"trialled" OR "launched" OR "expanded" OR "scale up" OR "approved" OR
"scrapped" OR "mandated" OR "decided to" OR "plans to" OR "will introduce" OR "evaluation"

## §4.4 NOT Clause — THREE components

### Component 1: GLOBAL_EXCLUDE (always, ALL queries)
"sentencing" OR "prosecution" OR "defendant" OR "verdict" OR "convicted" OR
"charged" OR "magistrate" OR "crown court" OR "jury" OR "appeal" OR
"tournament" OR "championship" OR "premier league" OR "goal" OR "coach" OR "match" OR
"album" OR "film" OR "theatre" OR "premiere" OR "box office"

### Component 2: METHOD_SPECIFIC_EXCLUDE (per target method)
expert_secondary adds NOT: "randomized" OR "randomised" OR "trial" OR "rct" OR "placebo" OR "control group"
gut adds NOT: "trial" OR "randomized" OR "randomised" OR "study" OR "evaluation" OR "pilot" OR "impact assessment" OR "review"
Other methods: no method-specific excludes beyond global.

### Component 3: FULL_METHOD_EXCLUDE (when method progress > 0.80)
Add that method's unique terms to NOT for OTHER methods' queries.
Pushes results away from over-represented methods.

## §4.5 Dynamic Term Discovery (every 5 rounds)
Collect Tier B → extract novel ngrams → Haiku classifies METHOD/DECISION/NOISE → test → keep/drop.
Provenance="discovered_round_N". This implements Sources 2-4 from old doc.

## §4.6 Supply Band + Adaptive Width
POST-FILTER minimum: 50. If <30 survive: relax body filter, score all.
MAX_TOTAL_AVAILABLE: 2500. NEVER exceed.
If total < 50: increase k by 2. If total > 2500: decrease k by 2. Up to 3 adjustments.
If still out of band: emergency query (MASTER_PLAN_v3.md §18.5.I).
ALWAYS score at least 30 articles per round.

## §4.7 Duplicate Query Prevention
Jaccard similarity > 0.80 with any previous query → REJECT.
Exception: goldmine queries may repeat.

# §5. BANDIT

## §5.1 Algorithm: Contextual Linear Thompson Sampling, ρ=0.97, M=30

## §5.2 Feature Space (11 features, compressed for ≤120 rounds)

Context (observed, can't control):
1. target_method_progress: max(credit/35, llm/100)
2. overall_progress: avg progress across 6 methods
3. round_phase: 0.0-0.33 (exploration), 0.33-0.66 (mixed), 0.66-1.0 (exploitation)
4. recent_unique_rate: unique fraction in last 3 rounds
5. recent_tier_b_rate: Tier B fraction in last 3 rounds

Query (chosen by bandit):
6. method_width: k/10 normalized (0.3-1.0)
7. decision_width: decision terms / 16 normalized
8. has_full_method_not: 1 if NOT includes full-method exclusions
9. estimated_log_available: log10(preflight total) / 6
10. method_is_fresh: 1 if not targeted in last 5 rounds
11. query_novelty: 1 - max(Jaccard with previous queries)

## §5.3 Reward (6 components)

```
V = min(1, unique_scored / 50)

R = V × [ α·(tier_a_new/100)
        + β·(tier_b_new/100)
        + γ·Σ w_m·(U_m/100)
        + η·(unique_rate)
        - δ·(scored_duplicate_rate)
        + ζ·goldmine_bonus ]

Defaults: α=0.20, β=0.15, γ=0.35, η=0.20, δ=0.15, ζ=0.05, p=1.5
w_m = max(0, 1 - progress_m)^p
unique_rate = unique_scored / scored_count
goldmine_bonus = 1.0 if triggered, else 0.0
```

Examples:
- 100 new articles, 0 Tier A, 5 Tier B → R ≈ 0.28 (exploration rewarded)
- 30 duplicate articles, 0 anything → R ≈ -0.04 (waste penalized)
- 80 new, 2 Tier A, 10 Tier B → R ≈ 0.45 (strong round)

Store ALL 6 component values + V + R in query_log.

## §5.4 Curriculum (3 phases)

Phase 1 — EXPLORATION (Rounds 1-20):
  Method rotation: 2 rounds per method, cycle all 6.
  Order: rct, case_study, prepost, expert_qual, expert_secondary, gut. Repeat.
  Every 5th round: wild card (k=10, method with fewest unique articles).
  ε-greedy: 20% random query from different method, k=10.
  Width default: k=7. Bandit adjusts within [5, 10].
  Guardian order-by alternates relevance/newest.

Phase 2 — MIXED (Rounds 21-60):
  Bandit chooses method AND width via Thompson Sampling.
  Must try each method ≥1 per 12-round block.
  Method persistence: up to 10 consecutive rounds on same method.
  If 10 rounds with 0 Tier B: switch, log "METHOD_EXHAUSTED".
  ε-greedy: 10%.
  Warm-start: batch update bandit posterior at round 21 using Phase 1 data.

Phase 3 — EXPLOITATION (Rounds 61+):
  Full bandit control. ε: 5%.
  Full-method NOT active at progress > 0.80.

# §6. GOLDMINE

After scoring 100 articles:
  If tier_b_rate > 0.25 OR tier_a_rate > 0.10:
    GOLDMINE. Fetch 5 more pages, sample 100 more.
    M1 uses HAIKU for extensions (budget saving). Other models normal.
    Continue until rate < 0.05 OR pages ≥ 20 OR round cost > $5.
    Query EXEMPT from duplicate prevention next round.
    Bandit receives ζ=1.0 bonus.

# §7. INTELLIGENT SAMPLING

When filtered > 100 articles:
  Title score (FREE): +2 per decision term, +2 per method term,
    +1 if relevant section, -3 per exclude term.
  Embedding uncertainty (local, cheap): embed title, cosine_sim to centroids,
    uncertainty = 1 - |max_sim - 0.5| × 2.
  Sample 100: 40 exploit (title_score) + 30 active-learn (uncertainty) +
    20 random + 10 from least-explored sections.
When filtered ≤ 100: score all.

# §8. FILTERING

- Live blog: ONLY filter URLs with "/live/" as path segment
- Body length: 15000 chars max (raised from 6000)
- Section filter via Guardian API: -football,-sport,-film,-music,-books,-games,-fashion
- If <30 survive filtering: RELAX body-length, keep all non-live-blog
- A round scoring 30 noisy articles > a round scoring 0

# §9. CROSS-ROUND LEARNING

§9.1 Vocabulary discovery every 5 rounds (per §4.5)
§9.2 Agreement monitoring every round: pairwise agreement, kappa,
  per-dimension correlation. Flag pair > 0.90 or < 0.10, model outlier > 60%.
§9.3 Model weight adjustment every 10 rounds (per §3.1)
§9.4 Dynamic threshold adjustment every 10 rounds (per §3.4)

# §10. DASHBOARD

4 pages per DASHBOARD_SPEC.md. Each page: Feynman main content +
folded "Technical details (Dan)" with parameters, formulas, versions.
Page 1: About This Project. Page 2: The Articles (Tier A + Tier B).
Page 3: Query History. Page 4: Model Health.

# §11. QUERY LOG COLUMNS

All from MASTER_PLAN_v3.md §7.3 PLUS:
query_width_k, decision_width, guardian_order_by,
unique_scored_count, duplicate_scored_count, unique_rate,
tier_a_count, tier_b_count, near_miss_count,
goldmine_triggered, goldmine_pages_extended,
phase, method_persistence_round, epsilon_override,
terms_discovered_this_round, m2new_diagnostic_status,
reward_alpha_term, reward_beta_term, reward_gamma_term,
reward_eta_term, reward_delta_term, reward_zeta_term,
reward_V, reward_R

# §12. TRAINING DATA HIERARCHY

When training any model, use sample_weight reflecting data reliability:
  GOLD (weight=1.0): expert-verified CSVs (all-high files, Training_cases.csv)
  SILVER (weight=0.8): human-scored CSV files (Method/Decision columns)
  BRONZE (weight=0.5): Tier A articles from pipeline (consensus-validated)
  UNCERTAIN (weight=0.2): Tier B articles (1+ model said yes)
  LIKELY_LOW (weight=0.3): scored by pipeline, not in any tier
Pipeline data supplements but NEVER overrides expert-verified data.
