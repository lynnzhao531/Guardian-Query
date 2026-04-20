# Part 8 — article-level companion stability

Across 5 seeds at K=15:

- Median companion stability: **0.262**
- P10 / P25 / P75 / P90: 0.108 / 0.163 / 0.377 / 0.464
- Median (in_training=yes): 0.320
- Median (in_training=no):  0.238

Interpretation: companion stability > topic-word Jaccard usually, because document clusters stay similar even when the top-20 words of their label drift across seeds.
