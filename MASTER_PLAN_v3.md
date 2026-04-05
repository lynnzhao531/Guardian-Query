# MASTER PLAN v3 — Experiment Aversion Article Finder
## Complete specification. Every detail. No ambiguity.
## Merges: Vibe Coding Instructions (ALL requirements verbatim) + Wang et al. 2025 (K*)
## Updated: April 2, 2026

---

# §1. FIXED GOAL (DO NOT REINTERPRET)

We are building a stable pipeline to query the Guardian, score retrieved articles with 4 judges, and iteratively refine search queries to discover HIGH-RELEVANCE policy-evidence articles.

High relevance = Decision is high AND at least one Method is high.
We store both discrete {0, 0.5, 1} and probabilities (p0, p05, p1).
We maintain per-method pools and a query log, and dynamically steer queries toward underrepresented methods until quotas are filled.

K* (validated hypothesis set from Wang et al. 2025) guides ALL 4 models and query generation, preventing reward hacking and keyword overfitting.

---

# §2. THE 7-DIMENSION SCORING SYSTEM

Every model outputs the full 7-vector:
1. decision
2. method_rct
3. method_prepost
4. method_case_study
5. method_expert_qual
6. method_expert_secondary (canonical name; expert_corr is alias only)
7. method_gut

For each of the 7 dimensions, store BOTH:
- discrete score ∈ {0, 0.5, 1}
- probabilities: p0, p05, p1 (must sum to ~1 within tolerance 0.02)

---

# §3. DATA INVENTORY

## §3.1 Detection rule (used everywhere — never hardcode filenames)

For each CSV file in data/:
1. Read header row
2. If file has BOTH "Method" AND "Decision" columns with numeric values (0, 0.5, 1):
   → SCORED file. "Method" column = score for ONE specific method type.
   Detect which method by scanning non-score column names:
   - Columns containing "randomized controlled trial", "randomised trial" → method_rct
   - Columns containing "before and after", "pre-post" → method_prepost
   - Columns containing "case study", "case studies" → method_case_study
   - Columns containing "administrative data", "observational study" → method_expert_secondary
   - Columns containing "has decided to", "minister decided" → method_gut
3. If file does NOT have both Method and Decision columns:
   → ALL-HIGH file. Every article is high relevance.
   Map to method type by filename pattern or Training_cases.csv method_category column.

## §3.2 All-High Files (no scores — every article is relevant)

| File | Articles | Method gold standard | Body column |
|---|---|---|---|
| rct.csv | 21 | method_rct=1, decision=1 | body |
| prepost.csv | 12 | method_prepost=1, decision=1 | article_body |
| casestudy.csv | 13 | method_case_study=1, decision=1 | article_body |
| expert_qual.csv | 10 | method_expert_qual=1, decision=1 | body_text |
| expert_secondary_quant.csv | 15 | method_expert_secondary=1, decision=1 | body_text |
| gut_decision.csv | 4 | method_gut=1, decision=1 | body_text |
| Training_cases.csv | 60 | method_category column maps each article | bodyText |
| **Total** | **~135 unique** | | |

## §3.3 Scored Files (Method and Decision columns, scored 0 / 0.5 / 1)

| File | Articles | "Method" means | Method dist (0/0.5/1) | Decision dist (0/0.5/1) |
|---|---|---|---|---|
| rct_2.csv | 916 | method_rct | 286/56/169 | 343/87/81 |
| prepost_2.csv | 94 | method_prepost | 58/16/18 | 58/21/13 |
| case_studies.csv | 1230 | method_case_study | 209/97/21 | 195/83/48 |
| quantitative.csv | 31 | method_expert_secondary | 24/4/3 | 25/4/2 |
| gut.csv | 99 | method_gut | 63/24/12 | 54/25/20 |
| **Total** | **~2370 unique** | | | |

No scored file exists for method_expert_qual (only all-high examples).
Files with numeric prefixes (e.g., 1775190284702_) are duplicates. Deduplicate by URL.

---

# §4. K* — KNOWLEDGE BASE (from Wang, Sudhir & Zhou 2025)

## §4.1 What K* is
A validated set of 5-10 natural-language hypotheses about WHY articles are relevant.
Each hypothesis tagged as primarily about: a specific method type, decision, or both.
Stored in knowledge_base/K_star.json.

## §4.2 How K* is built

### Abduction (mini-batch b=10, ~12 batches, use Claude Sonnet as reasoner)

Build 3 types of contrastive pairs from ALL data:
- CLEAR pairs (~50): all-high articles vs scored articles with Method=0, Decision=0
- METHOD-EDGE pairs (~25 per scored file): same file, Method>=0.5 vs Method=0
- DECISION-EDGE pairs (~25 pooled): Decision>=0.5 vs Decision=0 from any scored file

For each mini-batch of 10 pairs:
- Prompt Claude Sonnet with pairs (title + first 800 chars each)
- Label pair type so reasoner knows which dimension to focus on
- Collect hypotheses (## formatted lines)
- Novelty filter: keep only if cosine distance > 0.3 from existing pool
- Adaptive sampling: articles scored poorly under current K* are sampled more in next batch

Pool target: 80-150 unique hypotheses.

### Induction (validation on held-out articles)

For a candidate hypothesis set K (5-8 hypotheses):
- Embed K into scoring prompt for Claude Haiku
- Score 60 validation articles (30 HIGH + 30 LOW) on all 7 dimensions
- Predict 0/0.5/1 per dimension
- Weighted accuracy: exact match=1.0, off by 0.5=0.5, off by 1.0=0.0
- Utility s(K) = weighted accuracy averaged across all dimensions and articles
- For all-HIGH articles in validation: treat as Method=1, Decision=1 on their method

### Optimization (simulated annealing, 60-80 iterations)

- Start with random subset K of 6 hypotheses from pool
- Each iteration: propose K' by one of: add 1 hypothesis, remove 1, swap 1
- Evaluate K' via induction
- Accept if better; accept if worse with probability exp(Δ/T)
- Cool T from 1.0 toward 0.01 over iterations
- Track best K* seen
- Stagnation (no improvement for 10 iterations) → return to abduction with new pairs
- Output: knowledge_base/K_star.json with:
  {"hypotheses": [...], "hypothesis_tags": {"hyp1": "method_rct", ...},
   "validation_accuracy": float, "pool_size": int, "iterations_run": int, "timestamp": str}

## §4.3 How K* is used everywhere
- Model 1: K* hypotheses are the scoring rubric in the system prompt
- Model 2: K* hypotheses are in the system prompt during fine-tuning (knowledge-guided DPO)
- Model 3: K* hypotheses supplement embedding features
- Model 4: K* hypotheses define the feature dimensions for sklearn regressors
- Query generation: K* patterns + seed terms together form query clause libraries
- False-positive analysis: K*-abduction on FP vs TP pairs to explain gaps

---

# §5. THE 4 FIXED MODELS

Model families are fixed. Only parameters may change.

## §5.1 Model 1: K*-Guided LLM Judge

- Engine: Claude Haiku (claude-haiku-4-5-20251001)
- System prompt: K* hypotheses grouped by method type
- User prompt: article title + first 800 chars of body
- Output: full 7-vector, each dimension scored 0-5, mapped to {0,0.5,1}
  - 0-1 → discrete 0, p0=0.8 p05=0.15 p1=0.05 (approximate from score)
  - 2 → discrete 0.5, p0=0.2 p05=0.6 p1=0.2
  - 3-5 → discrete 1, p0=0.05 p05=0.15 p1=0.8
  (Refine these mappings based on calibration data)
- ONE API call per article
- Cost: ~$0.002/article
- Saved under: uses API, no local model file

## §5.2 Model 2: K*-Fine-Tuned GPT

- Engine: OpenAI gpt-4o-mini, fine-tuned on method-specific preference pairs
- K* embedded in system prompt during training (paper's knowledge-guided approach)
- Training data: ~2400 rows from method-specific preference pairs (§6)
- Output: full 7-vector with discrete scores and probabilities
- ONE API call per article
- Cost: ~$5-10 fine-tuning (one-time), ~$0.001/article scoring
- Model name stored in project_state/STATE.json as latest_finetuned_model_name
- Saved under: OpenAI hosted, name in STATE.json

## §5.3 Model 3: K*-Embedding Classifier

- Engine: sentence-transformers all-MiniLM-L6-v2 (local, free)
- 6 method-specific prototype centroids from all-high files
- Features: [6 cosine similarities to centroids] + [K* hypothesis features from Model 4]
- 7 sklearn MLPRegressors (hidden_layer_sizes=(32,16)), one per dimension
- Each regressor trained ONLY on articles with labels for that dimension
- Output: full 7-vector, continuous predictions mapped to {0,0.5,1} and p0/p05/p1
- Cost: free (runs locally)
- Saved under: models/model3/

## §5.4 Model 4: K*-Hypothesis Bottleneck Classifier

- Engine: Claude Haiku for feature extraction + sklearn Ridge regressors
- For each K* hypothesis, Haiku answers YES/NO → binary feature vector
- ONE Haiku call per article (all hypotheses scored in single prompt):
  "For each principle below, answer 1 if this article clearly exhibits it, 0 if not.
  Principles: [numbered list]. Answer as comma-separated 0s and 1s only."
- 7 Ridge regressors, each trained on articles with labels for that dimension:
  - method_rct_regressor trains on rct_2.csv + rct.csv articles
  - method_prepost_regressor trains on prepost_2.csv + prepost.csv
  - method_case_study_regressor trains on case_studies.csv + casestudy.csv
  - method_expert_secondary_regressor trains on quantitative.csv + expert_secondary_quant.csv
  - method_expert_qual_regressor trains on expert_qual.csv only (gold=1, random lows=0)
  - method_gut_regressor trains on gut.csv + gut_decision.csv
  - decision_regressor trains on ALL scored files pooled
- Output: full 7-vector, continuous predictions mapped to {0,0.5,1} and p0/p05/p1
- Cost: ~$0.001/article for feature extraction, free for sklearn
- Saved under: models/model4/

---

# §6. FINE-TUNING PLAN (Model 2)

## §6.1 Preference pairs per method dimension

For EACH scored file, generate pairs where articles differ on Method score:
- Winner: article with higher Method score
- Loser: article with lower Method score
- Also pair: all-high gold articles (winner) vs Method=0 articles (loser)

| Method | Source scored file | Source gold file | Est. pairs |
|---|---|---|---|
| method_rct | rct_2.csv | rct.csv | ~400 |
| method_prepost | prepost_2.csv | prepost.csv | ~80 |
| method_case_study | case_studies.csv | casestudy.csv | ~300 |
| method_expert_secondary | quantitative.csv | expert_secondary_quant.csv | ~40 |
| method_expert_qual | (none) | expert_qual.csv vs lows from other files | ~30 |
| method_gut | gut.csv | gut_decision.csv | ~80 |
| decision | ALL scored files pooled | ALL all-high files | ~300 |
| **Total** | | | **~1230 pairs → ~2460 training rows** |

## §6.2 Training format (JSONL)

Each training example:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You score articles for relevance to policy experimentation research on 7 dimensions. Use these validated principles:\n\nDECISION principles:\n[decision K* hypotheses]\n\nRCT METHOD principles:\n[rct K* hypotheses]\n\nPRE-POST METHOD principles:\n[prepost K* hypotheses]\n\n[...all 6 methods...]\n\nScore each dimension 0, 0.5, or 1. Use -1 for dimensions you cannot assess. Output JSON only."
    },
    {
      "role": "user",
      "content": "Title: {title}\nExcerpt: {first 800 chars}"
    },
    {
      "role": "assistant",
      "content": "{\"decision\": 0.5, \"method_rct\": 1, \"method_prepost\": -1, \"method_case_study\": -1, \"method_expert_qual\": -1, \"method_expert_secondary\": -1, \"method_gut\": -1, \"primary_method\": \"rct\", \"reasoning\": \"...\"}"
    }
  ]
}
```

- Scored dimension = actual score from CSV
- Other method dimensions = -1 (not evaluated)
- Decision = actual score (available in all scored files)
- For all-high articles: their specific method = 1, decision = 1, others = -1
- BOTH winner and loser from each pair become separate training examples
- Save to: knowledge_base/openai_training_7vec.jsonl (80% train)
- Save to: knowledge_base/openai_validation_7vec.jsonl (20% val)

## §6.3 Fine-tuning execution

- Platform: OpenAI API
- Model: gpt-4o-mini-2024-07-18
- Epochs: 3
- Estimated cost: $5-10
- Time: 10-30 minutes
- Poll status every 60 seconds until succeeded or failed
- Save fine-tuned model name to project_state/STATE.json
- If fine-tuning fails: continue with 3 models, log error, note in HANDOFF.md

---

# §7. UNCHANGEABLE PER-ROUND OUTPUTS

After each query round x you MUST create:

## §7.1 Per-model high-relevance CSVs (5 files):
```
outputs/rounds/round_x/round_x_model1_papers.csv     (K*-LLM Judge)
outputs/rounds/round_x/round_x_model2_papers.csv     (Fine-Tuned GPT)
outputs/rounds/round_x/round_x_model3_papers.csv     (Embedding Classifier)
outputs/rounds/round_x/round_x_model4_papers.csv     (Hypothesis Classifier)
outputs/rounds/round_x/round_x_high_relevant_papers.csv  (consensus)
```

## §7.2 Update 12 pool CSVs (6 methods × 2):
```
outputs/pools/pool_rct_overall.csv
outputs/pools/pool_rct_LLM.csv
outputs/pools/pool_prepost_overall.csv
outputs/pools/pool_prepost_LLM.csv
outputs/pools/pool_case_study_overall.csv
outputs/pools/pool_case_study_LLM.csv
outputs/pools/pool_expert_qual_overall.csv
outputs/pools/pool_expert_qual_LLM.csv
outputs/pools/pool_expert_secondary_overall.csv
outputs/pools/pool_expert_secondary_LLM.csv
outputs/pools/pool_gut_overall.csv
outputs/pools/pool_gut_LLM.csv
```

## §7.3 Append 1 row to outputs/query_log.csv with ALL these columns:
- round_id, timestamp
- base_query, final_query, template_id, target_method
- total_available
- pages_fetched (5), page_size (50)
- candidates_retrieved
- skipped_live_count, skipped_long_count, skipped_missing_text_count
- scored_count
- high_rel_model1_count, high_rel_model2_count, high_rel_model3_count, high_rel_model4_count
- consensus_high_rel_count
- new_consensus_credit_total
- new_model2_new_total
- duplicate_rate_consensus
- new_consensus_credit_rct, new_consensus_credit_prepost, new_consensus_credit_case_study, new_consensus_credit_expert_qual, new_consensus_credit_expert_secondary, new_consensus_credit_gut
- new_model2_rct, new_model2_prepost, new_model2_case_study, new_model2_expert_qual, new_model2_expert_secondary, new_model2_gut
- alpha_term, beta_term, gamma_term, delta_term, reward_R
- guardian_key_rotations_this_round, guardian_requests_this_round, guardian_keys_used
- backoff_level_used, preflight_totals_by_level_json

## §7.4 Write audit files:
```
audits/round_x/round_x_audit.md
audits/round_x/round_x_contract.json
```

If ANY of these are missing or schema-invalid: FATAL error. STOP.

---

# §8. SCORING SET RULES

- Retrieve FIRST K=5 pages, page-size=50 (≤250 candidates)
- Filter out (skip, do not score):
  a) live/blog formats (url contains /live/ or title contains "as it happened", etc.)
  b) long articles where len(body_text) > 6000 characters
  c) missing body text
- From remaining: take deterministic seeded RANDOM sample of up to 100
  (seed = hash(round_id + final_query))
- If fewer than 100 remain: score all
- Store counts of filtered items in query_log

---

# §9. BUDGET RULES

- Hard cap for query scoring spend = $60.00 USD
- Track Anthropic AND OpenAI usage tokens and compute cost per call
- Update project_state/COST_LEDGER.json after every API call
- BEFORE scoring a round, estimate worst-case cost for scoring N articles (N≤100) with all 4 models
- If it would exceed remaining budget: STOP and do not score further
- K* synthesis cost tracked separately (one-time, ~$5)
- Fine-tuning cost tracked separately (one-time, ~$5-10)
- Budget cap applies ONLY to per-round query scoring

---

# §10. GUARDIAN API CLIENT

- Support .env with GUARDIAN_API_KEY_1, GUARDIAN_API_KEY_2, GUARDIAN_API_KEY_3
- Automatically rotate keys on rate-limit/quota errors (HTTP 429, and 403/401 with quota/rate limit message)
- Persist rotation state in project_state/guardian_key_state.json
- Log all rotations (no secrets) in audits/guardian_api_rotation.log
- Per-key throttling: min_interval_sec default 1.0 (configurable)
- Never log key values; only safe fingerprints

---

# §11. CONTEXTUAL BANDIT

## §11.1 Algorithm
src/bandit.py: Contextual Linear Thompson Sampling with exponential decay for non-stationarity.

Parameters (configurable, stored in CONFIG.yaml):
- decay ρ = 0.97
- candidate set size M = 30
- exploration is inherent in TS; no epsilon needed

## §11.2 Context features
- per-method progress (overall_credit/35 and llm_count/100)
- is_full flags per method
- round index
- recent duplicate_rate trend

## §11.3 Query features
- template id one-hot
- target method one-hot
- counts of terms used from each space
- indicators for use of compound terms
- optional hashed term ids (keep bounded)

## §11.4 Reward function (EXACT — do not modify)
```
N = scored_count (≤100)
U_all = new consensus overall credit added this round (sum credits)
U_dpo = number of NEW Model2-high URLs added to LLM pools this round
U_m = new consensus credit added for method m this round
progress_m = max(overall_credit_m/35, llm_count_m/100)
w_m = max(0, 1 - progress_m) ^ p   with p=1.5
DUP = duplicate_rate_consensus (0 if no consensus hits)
V = min(1, N/100)

R = V * [ α*(U_all/100) + β*(U_dpo/100) + γ*Σ_m w_m*(U_m/100) - δ*DUP ]

Defaults: α=0.25, β=0.15, γ=0.60, δ=0.20, p=1.5
```

Store ALL components in query_log per round.

---

# §12. DYNAMIC PAYOFF / METHOD QUOTAS

Goal per method m:
- overall pool: reach 35 "credits" (fractional credit for tied methods), OR
- LLM pool: reach 100 unique articles
whichever happens first.

If method is "FULL" (quota met): stop using that method's method-specific query terms.
If all FULL: switch to "decision/common only" mode (no method-specific terms).

---

# §13. TIED METHODS (OPTION A REQUIRED)

If article's highest average method score is tied across k methods:
- Add to ALL tied pools
- In overall pools: credit = 1/k
- Quota check uses sum(credit), not row count
- tied_methods within epsilon=0.02

---

# §14. PER-MODEL HIGH RELEVANCE

Per-model high relevance extraction:
```
decision.score == 1 AND max(method.p1) >= 0.80
```

Create per round:
- round_x_model1_papers.csv
- round_x_model2_papers.csv (skip if Model 2 unavailable)
- round_x_model3_papers.csv
- round_x_model4_papers.csv

---

# §15. CONSENSUS

Include article in round_x_high_relevant_papers.csv if ALL of:
- avg decision p1 across available models > 0.80
- mean(top 3 avg method p1) > 0.70
- agreement: at least 3/4 models (or 2/3 if Model 2 unavailable) share a method in their top-2 and avg method p1 >= 0.70

Also compute:
- classified_method = argmax avg method p1
- tied_methods within epsilon=0.02
- confidence = highest_avg_method_p1 / sum(avg_method_p1 over 6)

---

# §16. POOL UPDATES

From consensus file:
- add to each tied pool_<m>_overall.csv with credit=1/k

From Model 2 (fine-tuned) file:
- add to each tied top method pool_<m>_LLM.csv (no credit needed)

No duplicates allowed. Use url_canon as primary key.
Update project_state/POOL_STATUS.json.

---

# §17. ROUND RUNNER ALGORITHM (steps A through O)

For each round x:

**A) Pre-round contract checks (FATAL if fail):**
- validate_repo_structure()
- validate_config_immutables()
- validate_models_exist() — check K_star.json exists, sklearn models exist, STATE.json has fine-tuned model name (or mark Model 2 unavailable)
- validate_pool_files_exist_or_create()
- check budget remaining in COST_LEDGER.json

**B) Decide target method:**
- Sample method proportional to underfill: weight = (1 - progress_m) among not FULL
- If all FULL: switch to "decision/common only" mode

**C) Generate candidate queries (M=30):**
- Use K*-guided query builder (§18) to produce candidates spanning templates
- Avoid candidates that exactly repeat last query unless flagged as goldmine

**D) Bandit selects query:**
- Compute feature vector x for each candidate
- Sample θ from posterior and pick argmax
- Run preflight + backoff ladder (§18.3) if total < 20 or total > 2500

**E) Guardian retrieval:**
- Fetch pages 1..5, page-size=50 via GuardianClient with key rotation
- Store raw candidate list to outputs/rounds/round_x/candidates_raw.csv
- Record total_available

**F) Filtering:**
- Mark and skip live/blog (url contains /live/ or title contains "as it happened")
- Mark and skip long if len(body_text) > 6000
- Mark and skip missing text
- Store filtered list to outputs/rounds/round_x/candidates_filtered.csv

**G) Sampling:**
- If filtered_count >= 100: seeded random sample of 100 (seed = hash(round_id + final_query))
- Else: score all remaining
- Store to outputs/rounds/round_x/scored_set.csv

**H) Scoring:**
- Run ALL available models (4, or 3 if Model 2 unavailable) on each sampled article
- Each model outputs 7 dimensions with p0/p05/p1 and discrete score
- Use caching by url_canon and model version to avoid re-scoring duplicates
- Store to outputs/rounds/round_x/scored_results_full.parquet
- Run model_agreement_monitor (§19) on results

**I) Per-model high relevance extraction:**
- decision.score==1 AND max(method.p1)>=0.80
- Create 4 per-model CSVs (or 3 if Model 2 unavailable)

**J) Consensus high relevance:**
- Apply thresholds from §15
- Create round_x_high_relevant_papers.csv

**K) Pool updates:**
- Option A tied handling per §13 and §16
- No duplicate url_canon in any pool
- Update POOL_STATUS.json

**L) Query log append:**
- ALL columns listed in §7.3
- Exactly one row per round

**M) Bandit update:**
- Compute feature vector x for chosen query
- Compute reward R per §11.4
- Update posterior with reward R using decay ρ

**N) Post-round audits (FATAL if fail):**
- validate_round_outputs(round_id)
- validate_pool_files()
- validate_query_log()
- validate_scored_rows() — all scored rows have outputs for ALL models and ALL 7 dimensions, probabilities sum to 1 within tolerance, scores valid
- Write audits/round_x/round_x_audit.md
- Write audits/round_x/round_x_contract.json

**O) Update RUNBOOK:**
- Append step completion, round summary, file paths

---

# §18. QUERY SYSTEM

## §18.1 Supply band (IMMUTABLE)
- MIN_TOTAL_AVAILABLE = 20
- MAX_TOTAL_AVAILABLE = 2500
- Preflight with page-size=1 before executing
- NEVER execute query with total > 2500 (FATAL)
- Guardian fetch: pages=5, page_size=50

## §18.2 Query construction

Every executed query MUST include:
(METHOD clause) AND (DECISION clause) AND (POLICY CONTEXT clause)
plus optional tightening clauses, and excludes.

### Fixed clauses (do not change):

POLICY_CONTEXT (P):
```
(policy OR programme OR program OR scheme OR service OR measure OR
 regulation OR initiative OR intervention OR council OR government OR
 NHS OR city OR "public health" OR education OR transport OR policing)
```

ACTOR (A):
```
(government OR minister OR department OR council OR mayor OR NHS OR
 regulator OR agency OR authority)
```

EVIDENCE (G):
```
(evaluation OR study OR analysis OR assessment OR review OR report OR
 findings OR data OR evidence)
```

### Seed terms (base set from vibe coding doc, ENHANCED by K*)

DECISION_STRICT_TERMS:
"pilot", "pilot scheme", "pilot programme", "pilot program",
"rolled out", "roll out", "rollout", "trialled", "trialed",
"to be trialled", "to be trialed", "expanded", "extend", "extended",
"scale up", "scaled up", "nationwide", "launched", "introduced",
"implemented", "implementation", "funded", "funding", "approved",
"mandated", "ban", "banned", "scrap", "scrapped", "pathfinder", "demonstrator"

DECISION_BROAD_ADDONS (weight=0.7):
"plans to", "set to", "will decide", "to decide whether",
"expected to", "is considering", "considering"

METHOD_RCT_INCLUDE:
"randomised controlled trial", "randomized controlled trial",
"randomised trial", "randomized trial", "randomly assigned",
"random assignment", "control group", "placebo", "trial arm",
"double blind", "blinded", "cluster randomised", "cluster randomized",
"A/B test", "AB test", "field experiment", "clinical trial"
COMPOUND ONLY: ("trial" AND ("randomised" OR "randomized" OR placebo OR "control group"))

METHOD_PREPOST_INCLUDE:
"before and after", "before-and-after", "pre and post", "pre-post",
"baseline", "follow-up", "follow up", "post-implementation",
"post implementation", "pre-intervention", "post-intervention"

METHOD_CASE_STUDY_INCLUDE:
"case study", "case studies", "case report", "case series",
"lessons learned", "pilot site", "case example"

METHOD_EXPERT_QUAL_INCLUDE:
"public consultation", "consultation", "call for evidence",
"expert panel", "advisory panel", "expert advisory panel",
"independent review", "inquiry", "commission", "taskforce",
"working group", "committee"

METHOD_EXPERT_SECONDARY_INCLUDE:
"administrative data", "linked data", "registry data",
"observational study", "retrospective study", "cohort study",
"econometric", "regression", "difference-in-differences", "DiD",
"regression discontinuity", "RDD", "propensity score matching", "PSM",
"panel data", "time series", "natural experiment", "quasi-experimental"

METHOD_GUT_INCLUDE:
"without evidence", "no evidence", "no data", "on a hunch",
"gut feeling", "intuitively", "despite lack of evidence"

GLOBAL_EXCLUDE_CORE:
legal: "sentencing", "prosecution", "defendant", "appeal", "court", "tribunal", "judge"
(NOTE: do NOT globally exclude "jury" — it breaks "citizens jury")
sports: "match", "tournament", "goal", "coach"
entertainment: "album", "film", "theatre", "premiere"

METHOD-SPECIFIC EXCLUDES:
- expert_secondary: NOT (randomized OR randomised OR trial OR rct OR placebo OR "control group")
- gut: NOT (trial OR randomized OR randomised OR study OR evaluation OR pilot OR "impact assessment" OR review)

EVIDENCE_TERMS:
"evaluation", "evaluate", "assess", "assessment", "impact evaluation",
"process evaluation", "evidence", "data", "analysis", "study found",
"review found", "results showed", "findings", "report"

### K* enhancement to seed terms

After K* is built, src/seed_builder.py extracts additional terms from K* hypotheses:
- Parse key phrases from each hypothesis
- Extract discriminating terms from high-relevance articles matching each hypothesis
- Add to the appropriate query space with provenance="K_star"
- Store in knowledge_base/query_spaces/*.json with version, hash, provenance per term
- Normalize terms: lowercase, strip, unify UK/US spelling

## §18.3 Templates T1-T10

Placeholders: {M}=method, {D}=decision, {P}=policy, {A}=actor, {G}=evidence, {X}=excludes

T1: ({M}) AND ({D}) AND ({P}) NOT ({X})
T2: ({M}) AND ({D}) AND ({P}) AND ({A}) NOT ({X})
T3: ({M}) AND ({D}) AND ({P}) AND ({G}) NOT ({X})
T4: ({M}) AND ({D}) AND ({P}) AND ({A}) AND ({G}) NOT ({X})
T5: ({M}) AND ({D}) AND (({P}) OR ({A})) NOT ({X})
T6: ({M}) AND ({D}) AND (({P}) OR ({A})) AND ({G}) NOT ({X})
T7: ({M}) AND ({D}) AND ({P}) AND ({G}) AND ("pilot" OR "rolled out" OR "implemented" OR "introduced") NOT ({X})
T8: ({M}) AND ({D}) AND ({P}) AND ("evaluation" OR "study" OR "analysis") NOT ({X})
T9: ({M}) AND ({D}) AND ({P}) AND ("government" OR "council" OR "nhs") NOT ({X})
T10: ({M}) AND ({D}) AND ({P}) AND ({A}) AND ("evaluation" OR "analysis") NOT ({X})

For gut: ONLY use T1, T2, T5, T9 (no evidence clause).

Candidate template order:
- If m != gut: [T4, T3, T2, T1, T6, T5, T10, T9, T8, T7]
- If m == gut: [T2, T1, T5, T9]

## §18.4 Clause construction rules

- {M}: OR bundle of AT LEAST 6 terms from method_m include_terms (first 6 deterministically)
- {D}: OR bundle of AT LEAST 6 STRICT decision terms (first 6 deterministically). Broad addons only in broadening step.
- {P}: always POLICY_CONTEXT fixed clause
- {A}: always ACTOR fixed clause
- {G}: always EVIDENCE fixed clause (except gut)
- {X}: OR bundle from global + method-specific excludes

## §18.5 Preflight/backoff/tighten algorithm (EXACT)

For each candidate template in order:

A) Build BASE query using STRICT decision terms only.
B) Preflight total_available (page_size=1).
C) If total_available == 0:
   → BROADEN: replace {D} with STRICT+BROADS (add first 6 broad addons). Preflight.
D) If still 0:
   → BROADEN 2: expand {M} to first 10 method terms. Preflight.
E) If still 0: mark DEAD, move to next template.

F) If total_available > 2500:
   → TIGHTEN 1: ensure {A} clause included. Preflight.
   → If still >2500 and m != gut: TIGHTEN 2: ensure {G} clause. Preflight.
   → If still >2500: TIGHTEN 3: restrict {D} to ULTRA_STRICT:
     ("pilot scheme" OR "rolled out" OR "trialled" OR "implemented" OR "introduced"). Preflight.
   → If still >2500: mark TOO_WIDE, move to next template.

G) If total in [20, 2500]: ACCEPT immediately.
   Record backoff/tighten trace in preflight_totals_by_level_json.

H) If total between 1 and 19: ACCEPT with low_supply_override flag in JSON field.

I) If all candidates DEAD or TOO_WIDE: use EMERGENCY query per method:
   - rct: ("randomised controlled trial" OR "randomized controlled trial" OR "field experiment") AND ("pilot" OR "rolled out" OR "implemented") AND (policy OR programme OR program OR scheme OR service OR government OR council OR nhs)
   - prepost: ("before and after" OR "pre-post") AND ("implemented" OR "rolled out") AND (policy OR programme OR program OR scheme OR service OR government OR council OR nhs)
   - case_study: ("case study" OR "pilot site") AND ("pilot" OR "implemented" OR "rolled out") AND (policy OR programme OR program OR scheme OR service OR government OR council OR nhs)
   - expert_qual: ("independent review" OR "expert panel" OR consultation) AND ("recommend" OR "recommendation" OR "plans to" OR "will decide") AND (government OR council OR nhs OR department)
   - expert_secondary: ("administrative data" OR "observational study" OR "difference-in-differences") AND ("implemented" OR "rolled out" OR "plans to") AND (policy OR programme OR program OR scheme OR service OR government) AND NOT (randomized OR randomised OR trial OR rct)
   - gut: ("without evidence" OR "no evidence" OR "no data") AND ("has decided to" OR "government decided to" OR "council decided") AND (policy OR programme OR program OR regulation OR measure OR government OR council) AND NOT (trial OR randomized OR randomised OR study OR evaluation OR pilot OR review)

   If emergency also returns 0: STOP and log error.

---

# §19. MODEL AGREEMENT MONITOR

After every scoring batch (step H in round runner), compute and log:

A) Pairwise agreement rates: % articles where each model pair agrees on high/low + Cohen's kappa
B) Dimension-level agreement: correlation per dimension per model pair. Flag if < 0.3.
C) Systematic disagreement: articles where 1 model says HIGH and 3 say LOW (outlier). Track which model is most frequent outlier.
D) Agreement trend: running average per round. WARNING if < 0.5, CRITICAL if < 0.3 (suggest K* refresh).
E) Save to: outputs/rounds/round_x/agreement_report.json + outputs/agreement_trend.csv
F) Alert conditions printed prominently.

---

# §20. FALSE-POSITIVE LEARNING (Step 6)

After each round:
- Identify false positives: predicted high by any model but NOT in consensus
- Compute frequent ngrams in FP vs consensus positives per method
- Add to excludes if appears in ≥3 FPs and lift_fp > threshold
- Store provenance "learned_exclude" with round_id
- Every 10 rounds: also run K*-abduction on (FP, TP) pairs → new hypotheses
- If stagnation (no consensus articles for 5+ rounds): trigger full K* refresh
- Report every 10 rounds: reports/exclude_terms_evolution.md

---

# §21. AUDIT GATES AND CONTRACT VALIDATION

src/framework_contract.py implements:
- validate_repo_structure()
- validate_config_immutables()
- validate_round_outputs(round_id)
- validate_pool_files()
- validate_query_log()
- validate_scored_rows()

Validates:
- Required files exist per round (§7)
- CSV schemas match exactly
- No duplicate url_canon inside any pool
- query_log appended exactly one row per round
- All scored rows have outputs for ALL available models and ALL 7 dimensions
- All probabilities sum to 1 within tolerance 0.02
- Scores valid {0, 0.5, 1}
- Query constraints: final_query contains method+decision+policy, total_available ≤ 2500, query contains "AND" at least twice

Run BEFORE and AFTER each round. If fail: FATAL STOP.

Query-specific validation:
- final_query contains at least one decision term
- final_query contains at least one method term for target method
- final_query contains at least one policy context token
- final_query is not a single quoted phrase (must contain "AND" at least twice)
- template_id is one of T1..T10

---

# §22. STRICT ENGINEERING PRACTICES

A) Config-driven: all parameters in project_state/CONFIG.yaml
   Immutable keys list in project_state/CONTRACT.json
   Changing immutable keys in RUN mode = FATAL

B) Hash tracking: SHA256 of CONTRACT.json, CONFIG.yaml, and key scripts:
   src/run_query_loop.py, src/round_runner.py, src/bandit.py,
   src/query_builder.py, src/framework_contract.py, src/knowledge_synthesizer.py
   Write to project_state/framework_hashes.json each round.
   If hashes changed: loud WARNING in RUNBOOK and audits.

C) Unit tests (pytest under tests/):
   - query_log columns exist
   - pool dedupe works
   - credit sums correct
   - consensus file schema correct
   - K_star.json schema correct
   Run at start of each run.

D) Never silently change:
   All heuristic changes (tie epsilon, thresholds) MUST be in CONFIG.yaml and logged in RUNBOOK.

E) Run mode: DEV while building. RUN after dry run + 1 real round pass all audits.
   Do NOT switch to RUN until all audit gates pass.

---

# §23. PERSISTENT STATE FILES

```
project_state/CONFIG.yaml
project_state/STATE.json                (latest_finetuned_model_name)
project_state/COST_LEDGER.json          (Anthropic + OpenAI costs)
project_state/POOL_STATUS.json
project_state/BANDIT_STATE.json
project_state/SEEN_URLS.json
project_state/RUNBOOK.md
project_state/ROUND_PROTOCOL.md
project_state/CONTRACT.json
project_state/guardian_key_state.json
project_state/framework_hashes.json
project_state/finetune_job.json
```

---

# §24. FILE STRUCTURE

```
experiment-aversion-finder/
├── .env
├── CLAUDE.md                           (short reference → points here)
├── MASTER_PLAN_v3.md                   (THIS FILE — full spec)
├── HANDOFF.md                          (session state)
├── requirements.txt
├── data/                               (READ ONLY — originals)
├── knowledge_base/
│   ├── K_star.json
│   ├── hypothesis_pool.json
│   ├── article_pairs.json
│   ├── preference_pairs.json
│   ├── openai_training_7vec.jsonl
│   ├── openai_validation_7vec.jsonl
│   ├── query_spaces/*.json
│   ├── term_registry/term_registry.jsonl
│   └── seed_terms_static.yaml          (vibe coding seed terms)
├── skills/
│   └── scoring_rubric.md
├── src/
│   ├── knowledge_synthesizer.py
│   ├── data_loader.py
│   ├── model1_llm_judge.py
│   ├── model2_finetuned.py
│   ├── model3_embedding_classifier.py
│   ├── model4_hypothesis_classifier.py
│   ├── consensus.py
│   ├── model_agreement_monitor.py
│   ├── guardian_client.py
│   ├── config_env.py
│   ├── query_builder.py
│   ├── seed_builder.py
│   ├── bandit.py
│   ├── round_runner.py
│   ├── run_query_loop.py
│   └── framework_contract.py
├── models/
│   ├── model3/
│   └── model4/
├── outputs/
│   ├── rounds/round_x/
│   ├── pools/
│   ├── query_log.csv
│   └── agreement_trend.csv
├── audits/
├── project_state/
├── tests/
├── reports/
└── dashboard.py
```

---

# §25. EXECUTION ORDER

Phase 1: K* Synthesis
Phase 2: Build 4 Models + Agreement Monitor
Phase 3: Pipeline Infrastructure (contract, guardian, query builder, bandit, round runner, FP learning)
Phase 4: Dry Run + Real Rounds
Phase 5: Update HANDOFF.md for morning review
