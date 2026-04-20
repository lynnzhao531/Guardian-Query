# Synthetic Benchmark — Analysis Summary

Articles: 46 (10 A, 10 B, 10 C, 10 D, 6 E)
Ground truth labels: HIGH=10, BORDERLINE=6, LOW=30

## Headline metrics

| model | n | AUC A-vs-D | AUC A-vs-BCD | P@5 | P@10 | P@20 | C-FPR | rank corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | 46 | 1.0 | 0.9 | 1.0 | 1.0 | 0.5 | 0.0 | 0.7098 |
| M2_old | 46 | 1.0 | 0.8833 | 1.0 | 1.0 | 0.5 | 0.2 | 0.6807 |
| M2_new | 46 | 1.0 | 0.9 | 1.0 | 1.0 | 0.5 | 0.1 | 0.7245 |
| M3 | 46 | 1.0 | 0.8567 | 0.4 | 0.4 | 0.4 | 0.6 | 0.5818 |
| M4_v3 | 46 | 0.96 | 0.7517 | 0.2 | 0.4 | 0.3 | 0.1 | 0.4702 |
| M5 | 46 | 1.0 | 0.9133 | 0.8 | 0.5 | 0.45 | 0.2 | 0.6591 |
| M6 | 46 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 0.1 | 0.8306 |

Best AUC A-vs-D: **M1** (1.0).
Lowest Category-C false-positive rate: **M1** (0.0).

## Ranking agreement (continuous models)

| A | B | n | Spearman | Kendall τ | top-10 overlap |
|---|---|---:|---:|---:|---:|
| M3 | M4_v3 | 46 | 0.7432 | 0.591 | 6 |
| M3 | M5 | 46 | 0.7728 | 0.5826 | 6 |
| M3 | M6 | 46 | 0.7713 | 0.5978 | 4 |
| M4_v3 | M5 | 46 | 0.7218 | 0.5378 | 5 |
| M4_v3 | M6 | 46 | 0.7044 | 0.5487 | 4 |
| M5 | M6 | 46 | 0.8687 | 0.7085 | 5 |

## False positive count per model (non-A scored p1 > 0.5)

- M1: 12
- M2_old: 12
- M2_new: 12
- M3: 22
- M4_v3: 13
- M5: 13
- M6: 5

## Trap subcategories — which model falls for which

| subcategory | M1 | M2_old | M2_new | M3 | M4_v3 | M5 | M6 |
|---|---|---|---|---|---|---|---|
| TV pilot episode | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| UK adopts based on foreign data | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| academic paper discussion no decision | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| academic research announcement no policy | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| administrative data business | 0 | 1 | 1 | 1 | 0 | 1 | 1 |
| court trial with 'controlled trial' language | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| decision based on others' research | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| decision on one-off event | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| evaluation after decision | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| evidence-based rhetoric no specifics | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| historical retrospective | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| international comparison as color | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| medical RCT no policy decision | 0 | 1 | 0 | 1 | 0 | 1 | 0 |
| method present decision implicit | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| multi-option policy discussion | 1 | 1 | 0 | 0 | 1 | 1 | 0 |
| opinion piece for policy | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| pilot unclear if test or rollout | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| policy already implemented reaction | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| policy debate coverage no data | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| politician announces policy no evaluation | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| rumored upcoming policy | 1 | 1 | 1 | 1 | 0 | 1 | 0 |
| weather | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| weight loss before and after | 0 | 0 | 0 | 1 | 0 | 0 | 0 |

## Files in this folder

- `raw_scores.csv`
- `per_model_metrics.csv`
- `ranking_agreement.csv`
- `score_distributions.csv`
- `category_breakdown.csv`
- `false_positive_analysis.csv`
- `cost.json`
- `run.log`
