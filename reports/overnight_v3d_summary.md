# v3d Overnight Summary — STARVED-METHOD FOCUS

## Context (entering v3d, after v3c rounds 111-130)

v3c delivered 62 Tier A + ~600 Tier B across 20 rounds, but yield was
concentrated in RCT:

| Method            | Pool credit / 35 |
|-------------------|------------------|
| rct               | 93.9 (FULL, 2.7×) |
| expert_secondary  | 10.2             |
| expert_qual       | 8.2              |
| gut               | 5.0              |
| prepost           | 4.9              |
| case_study        | 3.9              |

Root cause: **precision, not supply.** RCT vocabulary is dense and
distinctive; starved methods have noisy vocabulary that pulls in RCT
articles. The pipeline keeps re-scoring the same winners.

## The 7 fixes

**Fix 1 — RCT phrases → GLOBAL_EXCLUDE.** 8 explicit RCT phrases
("randomised controlled trial", "control group", "placebo", …) are now
always excluded from queries. Removes noise from non-RCT rounds and
stops RCT from consuming budget.

**Fix 2 — Bandit RCT override.** If the bandit selects `target == "rct"`,
redirect to the *lowest-credit* non-RCT method. RCT pool is already 2.7×
full; further mining is pure waste. Manifest logs
`rct_override_fired`, `rct_override_to`, `rct_override_credits`.

**Fix 3 — M5 zero-cost pre-filter.** Before the expensive 6-model
ensemble, score all unique articles with the DistilBERT (M5). Keep only
those with `decision_p1 ≥ 0.15`. Only active for non-RCT methods and
batches ≥20 articles. Safety net: if M5 rejects everything, fall back to
top-20 by p1. Manifest logs `m5_prefilter_input`, `m5_prefilter_passed`.

**Fix 4 — Compound intelligent sampling.** Title relevance now
combines method keywords (×2 weight) with a 30-term
`DECISION_INDICATORS` list (×1 weight: policy, scheme, rollout, pilot,
reform, launch, review, find, effective, …). Shifts the top-40 window
toward titles that actually describe decisions.

**Fix 5 — Near-miss reward.** The bandit reward now includes a
`near_miss_count` term (articles where exactly `threshold - 1` models
agreed on HIGH — i.e. "almost Tier A"). Weight: **0.10**. Rebalanced:
tier_b 0.15 → 0.10, unique_rate 0.20 → 0.15. Total weights preserved.
Provides gradient signal in rounds that produce no Tier A but several
near-hits.

**Fix 6 — Body mining + cross-method mining.**
- `_read_gold_titles` now extracts body snippets (first 3 sentences,
  up to 1500 chars) alongside titles, giving the ngram miner real
  sentence-level material instead of bare headlines.
- New `mine_cross_method_pools()` reads `pool_rct_overall.csv` and
  `pool_expert_secondary_overall.csv`, runs candidate ngrams through
  Haiku, and **only accepts phrases Haiku classifies to a *different*,
  starved method**. Harvests the starved-method language that's hiding
  in the most abundant pools. Wired into the every-5-rounds discovery
  block in round_runner.

**Fix 7 — 40% trial rate for starved methods.** In
`query_builder.generate_candidates`, the trial-term usage probability
is 0.40 when target method progress < 0.30 (roughly pool credit < 10),
otherwise 0.20. Speeds up trial-term evaluation for under-represented
methods so failing terms get dropped and winners graduate faster.

## Invariants preserved
- 6 models untouched
- Tier A threshold ≥3 (no change)
- Per-model thresholds unchanged
- Static NOT clause unchanged (only *additions*)
- try/finally, manifests, config snapshots, vocabulary discovery cycle,
  stuck detector, exploration floor, pagination, decision-only,
  skip-when-unique<5, cost guard, auto-push every 5 rounds — all intact
- v3d backups of `round_runner.py`, `query_builder.py`, `bandit.py`,
  `consensus.py`, `vocabulary_discovery.py` in `src/*_v3d_backup.py`

## Launch
```
nohup python3 src/run_query_loop.py --max_rounds 20 --budget_usd 60 --resume \
  > overnight_v3d.log 2>&1 & disown
```
Round range: **134 – 153**, budget **$60**.

## Results
_To be filled in after the run completes._
