# DASHBOARD SPECIFICATION
## Two audiences: researchers (Feynman-style main) and tech experts like Dan (folded details)

## *** ABSOLUTE RULE: DATA PRESERVATION ***
Every round's articles and queries are PERMANENT data. They must NEVER 
be lost, even when pipeline code, models, or dashboard versions change.
Specifically:
- Tier A articles: round, title, link, scores, method, confidence — ALL rounds, ALWAYS shown
- Tier B articles: same columns — ALL rounds, ALWAYS shown
- Query log: round, exact query text, how many found, tier a/b counts, target method — ALL rounds
- If scored_results_full.csv or query_log.csv from early rounds have different 
  column formats: compute missing columns from available data, never drop the round
- Claude Code MUST search outputs/rounds_v1_archive/ AND outputs/rounds/ to find ALL round data
- Dashboard must load and display rounds 1-10 (v1), 11-30 (v2), 31+ (v3) together
- Lay people need: ARTICLES FOUND and QUERIES USED. They do NOT need model parameters or reward formulas.

## DESIGN PRIORITY ORDER:
1. Articles found (Tier A and B) — this is the ENTIRE POINT of the project
2. Queries used and their yield per round — documents our search process
3. Overall progress (pools filling up) — are we getting closer to the goal?
4. Technical details (Dan only, folded) — for reproduction, not for browsing

# STRUCTURE: 4 PAGES
Page 1: "About This Project" — what, why, how, progress
Page 2: "The Articles" — Tier A, Tier B, (future: Tier Meta) — MOST IMPORTANT PAGE
Page 3: "Query History" — every round's query and results — SECOND MOST IMPORTANT
Page 4: "Reproduction Files (Dan)" — all docs, code, configs to reproduce the project
Note: "Model Health" is NOT a separate page. It goes inside Dan's folded section on Page 1.

# PAGE 1: "About This Project"

## MAIN CONTENT

### What We're Looking For
"We search The Guardian for a specific type of article: ones where an organization
faces a policy choice, uses some method to figure out which option is better,
and decides what to do based on the results. Like a doctor checking evidence
before prescribing — we want articles about organizations that check evidence
before choosing policy. These articles are rare (fewer than 1 in 100), use
wildly different vocabulary, and even experts disagree on borderline cases."

### Why It's Hard
"The word 'trial' appears in thousands of Guardian articles about court cases,
not policy experiments. 'Pilot' could mean a TV show or a policy test. The
relevance isn't in the words — it's in the article's structure: does it
describe testing → evidence → decision? No keyword search can reliably
capture this."

### How The Pipeline Works
Show simple flow diagram:
Search Guardian → Filter → Score with AI models → Consensus (Tier A/B) → Collect by method
(queries learn      (remove        (each scores on    (strong vs        (6 method pools
 what works)         live blogs)     7 dimensions)      candidate)        fill to target)

### The 7 Dimensions
"Every article scored on: Did they USE A METHOD to evaluate? (6 types:
RCT, pre-post, case study, expert review, data analysis, gut decision)
And: Did they MAKE A DECISION based on it?"

### Current Progress
Round count, unique articles scored, Tier A found, Tier B found.
Per-method progress bars (credits/35 target).
Phase indicator (Exploration/Mixed/Exploitation). Budget spent/remaining.

### FOLDED: "Technical details (Dan)"

#### The 5 Models (table: name, approach, K*-dependent?, what makes it unique)
#### K* Knowledge Base: what K* is conceptually ("validated principles about what
makes articles relevant, discovered by showing AI pairs of relevant/irrelevant
articles"). List all hypotheses. Validation accuracy. Method tags.
#### Query System: structure (METHOD AND DECISION NOT GARBAGE), variable width,
Guardian section filter, order-by alternation, supply band [50-2500],
duplicate prevention, method-specific excludes.
#### Multi-Armed Bandit: Thompson Sampling, 11 features listed, 6-component reward
with formula and plain-English explanation of each term.
#### Training Data Hierarchy: GOLD/SILVER/BRONZE/UNCERTAIN/LIKELY_LOW with weights.
#### Version History:
"v1 (rounds 1-10): 4-5 AND clauses, 4 models, single threshold.
 Issue: 40 articles/round, 74% duplicates, 0 Tier A.
 v2 (rounds 11-30): 2 AND + 1 NOT, 5 models, two-tier, method rotation, 6-component reward.
 v3 (rounds 31+): M1 decomposed, M3 contrastive, M4 continuous, M5 DeBERTa."

# PAGE 2: "The Articles"

## MAIN CONTENT

### Tier A — Strong Finds
"Multiple AI models independently agreed these describe real policy evaluations
with decisions. Read these first — highest confidence."
Table columns IN THIS ORDER: round, title (clickable URL), method, confidence, 
models agreeing, method_certainty, article_relevance_score.
Filter: method, confidence, round range. Download CSV.
NOTE: This table shows articles from ALL rounds (1-10, 11-30, 31+). 
For early rounds with different column formats, compute missing fields 
from available data. NEVER omit a round.

### Tier B — Candidates
"At least one model flagged these. 'Near Tier A' = one vote short.
Disagreement types:
- outlier_high: one model sees something others don't
- method_disagree: relevant but unclear which method
- threshold_miss: scores moderate, just below cutoff
- decision_split: unclear if policy decision happened"
Table columns IN THIS ORDER: round, title (clickable URL), method, which_models_high, 
disagreement_type, relevance_score, near_tier_a.
Filter: disagreement_type, method, round range. Download CSV.
NOTE: Same preservation rule — ALL rounds, never omit.

### (Future) Tier Meta
"Coming soon — active learning meta-model combining all model insights."

### FOLDED: "Technical details (Dan)"
#### How articles enter tiers: step-by-step with thresholds
#### Continuous relevance score formula
#### Dynamic threshold: near-miss ratio, rule, current value, history
#### Method certainty: HIGH/MEDIUM/LOW definitions
#### Tied methods and credit
#### Tier promotion (B→A): how articles build consensus across rounds
#### Version history: threshold changes, when/why adjusted

# PAGE 3: "Query History"

## MAIN CONTENT — EVERY ROUND MUST BE SHOWN, no exceptions

### DATA RECOVERY: If rounds 1-10 data is in outputs/rounds_v1_archive/, 
load from there. If query_log.csv has mixed formats, parse both. If some 
columns are missing from old rounds, show "N/A" — NEVER drop the round.

### Summary Stats
Total rounds, avg scored/round, avg unique rate, best round, methods covered.

### Round-by-Round Table (searchable, sortable)
Default columns (what lay people need):
  Round, Method targeted, Query text (full, expandable), 
  Articles found (total_available), Articles scored, 
  Unique articles (not duplicates), Tier A found, Tier B found
Toggle "Show technical details":
  width k, decision width, order-by, reward R and components,
  Guardian stats, goldmine, M2-new status, phase

### Trend Charts
- Articles scored per round
- Unique rate (should stay >60%)
- Tier A + Tier B per round
- Reward R trend (should go up as bandit learns)
- Duplicate rate (should go down)
- Method targeted per round (colored)

# PAGE 4: "Reproduction Files (Dan)"

## MAIN CONTENT

"This page is for Dan or anyone who needs to reproduce or audit the project.
It provides all configuration files, code, and documentation needed to 
understand and re-run the entire pipeline from scratch."

### Project Documentation
Upload/link to: CLAUDE.md, MASTER_PLAN_v3.md, REVISED_ARCHITECTURE.md,
DASHBOARD_SPEC.md, K_star.json, DPO_GUARDS.md, CONFIG.yaml

### Source Code
List all src/*.py files with one-line descriptions.
Link to download each or download all as zip.

### Model Files
List models/model3/, models/model4/, models/model5/ contents.
Note which version is active and which are backups.

### Data Files
List data/*.csv with row counts and descriptions.
Note: original expert-verified data, NOT pipeline outputs.

### State Files
project_state/STATE.json, MODEL_WEIGHTS.json, BANDIT_STATE.json,
COST_LEDGER.json, SEEN_URLS.json

### How to Reproduce (Feynman-style, for a CS professor)
"1. Read CLAUDE.md to understand the project structure.
 2. Read REVISED_ARCHITECTURE.md for the complete system specification.
 3. The pipeline searches The Guardian for articles matching a specific 
    structural pattern (method + decision). It uses 5 independent AI models 
    to score articles, requires multi-model consensus for high-confidence 
    finds (Tier A), and keeps single-model candidates for review (Tier B).
 4. Queries are generated by a contextual bandit that learns which search 
    terms work best for each of 6 method types. The reward function balances 
    finding articles, exploring new territory, and avoiding duplicates.
 5. Run: python src/run_query_loop.py --max_rounds 30 --budget_usd 60
 6. Results appear in outputs/rounds/ and outputs/pools/."

### FOLDED: "Model Health (Dan)"
This section contains what was previously Page 4:
  Agreement heatmap (pairwise). Per-model stats. Model weight history.
  M2-new diagnostic tracking. Weight formula. Error correlations.
  Model families and independence. Version history of model changes.

# IMPLEMENTATION
- st.set_page_config(page_title="Experiment Aversion Finder", layout="wide")
- st.sidebar with st.radio for page navigation
- st.expander("Technical details in case you're interested (Dan)")
- Handle missing columns gracefully (old rounds → "N/A", NEVER drop the round)
- Handle empty data (show "Waiting for data..." not crash)
- Download CSV for Tier A and Tier B
- run_dashboard.sh: headless, no email prompt, port 8501
- "Refresh Data" button on each page

## CRITICAL DATA LOADING LOGIC:
- Load round data from BOTH outputs/rounds/ AND outputs/rounds_v1_archive/
- Load query_log from BOTH outputs/query_log.csv AND outputs/query_log_v1_archive.csv
- If old rounds lack columns (e.g., no tier_a_count), compute from available data:
  - Count per-model papers to estimate which_models_high
  - Count high_relevant papers to estimate tier_a/tier_b
  - Use consensus_high_rel_count as tier_a_count if tier_a_count missing
- MERGE all rounds into unified display sorted by round_id
- LABEL each round with its version: v1 (1-10), v2 (11-30), v3 (31+)
