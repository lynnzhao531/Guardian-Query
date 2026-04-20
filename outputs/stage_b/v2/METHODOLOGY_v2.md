# Stage B v2 — Methodology

Run: 2026-04-19T15:17:06, elapsed 3:23:06

## 1. Executive summary

- Selected preprocessing: **P2** (of 5 tested)
- Final vocab filter: **min_df=5, max_df=0.5**
- Final priors: **alpha=auto, eta=auto**
- Selected K: **[15, 5]**
- Topic-word Jaccard at final K: see K_sweep/sweep_results.csv
- Article companion stability (median): 0.262

## 2. Objective shift: filter → structural understanding

v1 used LDA to remove irrelevant topics. v2 frames LDA as a map of corpus themes; we care about document cluster stability more than word-label stability.

## 3. Dataset

Corpus: 1,937 Stage-A survivors (608 overlap with expert training data).

## 4. v1 baseline

See `01_v1_diagnostic.md`.

## 5. Preprocessing (Part 2)

5 variants tested: P1 (minimal), P2 (lemma), P2b (+corpus-specific stopwords), P3 (bigrams), P4 (NER). See `preprocessing/comparison.md`. Decision rule: prefer higher stability when c_v margin is narrow.

## 6. Vocabulary filter (Part 3)

12-cell grid tested at K=25 with 3 seeds per cell. See `vocab_sensitivity/grid_heatmap.png`. Decision logged in `vocab_sensitivity/decision.md`.

## 7. Hyperparameter priors (Part 4)

25-cell alpha × eta grid tested at K=25 with 3 seeds. See `prior_sensitivity/alpha_eta_stability_heatmap.png`. Decision in `prior_sensitivity/decision.md`.

## 8. K selection (Part 5)

9 K values × 5 seeds + perplexity. See `K_sweep/K_sweep_plots.png`, `K_sweep/K_decision.md`.

## 9. Section-stratified LDA (Part 6)

Top-10 sections run independently at K=3,5,8 × 3 seeds. See `section_stratified/summary.md`.

## 10. BERTopic triangulation (Part 7)

3 configurations (default, coarse, very_coarse) at the final K. See `bertopic/alignment_analysis.md` and `bertopic/visualizations/`.

## 11. Article-level stability (Part 8)

Companion stability across 5 seeds at final K. See `article_stability/article_stability_summary.md`.

## 12. Per-topic profiles (Part 9)

See `topic_profiles/INDEX.md` and topic_k.md files.

## 13. Assumptions verified

| # | assumption | check location |
|---|---|---|
| A1 | document length | Part 1 |
| A2 | bag-of-words | Part 2 (P3/P4 partial mitigation); Part 7 (BERTopic) |
| A3 | exchangeability | acknowledged limitation |
| A4 | Dirichlet priors | Part 4 (full grid tested) |
| A5 | K correctly specified | Part 5 (5 seeds + perplexity) |
| A6 | vocab informative | Part 3 (full grid tested) |
| A7 | topic interpretability | Part 9 + Part 7 BERTopic |

## 14. Limitations

- Small corpus (1,937 docs) vs 10k+-term vocab
- Soft-assignment: 57% articles have P_dominant < 0.5
- c_v critique (Hoyle et al. 2021) mitigated by c_npmi + companion stability + BERTopic
- Pre-2018 articles = 13% (temporal non-exchangeability)
- Training overlap uneven by topic

## 15. Follow-ups available

- Dynamic LDA (Blei & Lafferty 2006)
- Hierarchical LDA
- Author-topic model (if byline available)

## 16. Reproduction

Python 3.9. Seeds [42, 13, 99, 7, 123]. See `requirements_stage_b_v2.txt`. Run: `python3 scripts/stage_b_v2.py`.
