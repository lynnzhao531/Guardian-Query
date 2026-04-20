# Part 1 — v1 diagnostic baseline

Corpus: 1937 articles, avg training overlap 31.4%

## Assignment concentration
- Strong (P≥0.5): 43.0%  Medium (0.3≤P<0.5): 46.6%  Weak (P<0.3): 10.4%
- Median 2nd-highest topic prob: 0.225
- Median per-article entropy: 1.335
- Mean per-article entropy: 1.282 (max if uniform over K=25: 3.219)

## Per-topic profile
| T | n | %training | P50_dom_prob | top section | section % |
|---:|---:|---:|---:|---|---:|
| 0 | 54 | 9% | 0.500 | UK news | 31% |
| 1 | 94 | 44% | 0.475 | Education | 50% |
| 2 | 24 | 46% | 0.421 | Technology | 21% |
| 3 | 125 | 26% | 0.498 | Environment | 43% |
| 4 | 39 | 10% | 0.451 | Business | 31% |
| 5 | 103 | 20% | 0.456 | Australia news | 26% |
| 6 | 21 | 19% | 0.563 | Law | 24% |
| 7 | 41 | 68% | 0.493 | Society | 24% |
| 8 | 65 | 23% | 0.431 | World news | 35% |
| 9 | 150 | 81% | 0.546 | Society | 35% |
| 10 | 27 | 81% | 0.509 | Society | 52% |
| 11 | 96 | 21% | 0.405 | Society | 41% |
| 12 | 51 | 37% | 0.512 | Environment | 37% |
| 13 | 64 | 17% | 0.426 | US news | 69% |
| 14 | 30 | 10% | 0.448 | Australia news | 50% |
| 15 | 36 | 14% | 0.414 | Technology | 42% |
| 16 | 151 | 42% | 0.415 | Society | 12% |
| 17 | 117 | 22% | 0.461 | Money | 28% |
| 18 | 40 | 15% | 0.414 | Society | 25% |
| 19 | 64 | 20% | 0.418 | Society | 50% |
| 20 | 180 | 17% | 0.442 | Politics | 33% |
| 21 | 24 | 4% | 0.505 | Society | 54% |
| 22 | 30 | 17% | 0.814 | Society | 67% |
| 23 | 117 | 56% | 0.561 | World news | 39% |
| 24 | 194 | 18% | 0.479 | Australia news | 71% |

**Topics with training% deviating >20 pts from corpus avg 31.4%:** 8

- T0: training% = 9%
- T4: training% = 10%
- T7: training% = 68%
- T9: training% = 81%
- T10: training% = 81%
- T14: training% = 10%
- T21: training% = 4%
- T23: training% = 56%

## Section × Topic matrix (top-10 × top-10 by count)
|  | T24 | T20 | T16 | T9 | T3 | T17 | T23 | T5 | T11 | T1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Australia news | 138 | 14 | 13 | 12 | 28 | 14 | 14 | 27 | 28 | 12 | 
| Society | 2 | 7 | 18 | 52 | 1 | 6 | 23 | 21 | 39 | 4 | 
| World news | 6 | 39 | 14 | 20 | 5 | 8 | 46 | 15 | 2 | 2 | 
| UK news | 12 | 19 | 3 | 4 | 3 | 10 | 6 | 20 | 1 | 3 | 
| US news | 0 | 3 | 7 | 5 | 6 | 0 | 9 | 10 | 5 | 2 | 
| Business | 16 | 10 | 3 | 7 | 11 | 25 | 1 | 0 | 1 | 0 | 
| Politics | 5 | 60 | 10 | 0 | 3 | 8 | 4 | 3 | 1 | 4 | 
| Environment | 2 | 1 | 3 | 0 | 54 | 2 | 2 | 1 | 0 | 0 | 
| Education | 0 | 3 | 11 | 4 | 2 | 1 | 0 | 0 | 4 | 47 | 
| Science | 1 | 1 | 17 | 31 | 4 | 0 | 5 | 0 | 1 | 2 | 
