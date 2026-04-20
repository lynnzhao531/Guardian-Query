# Part 6 — section-stratified LDA

| section | n | best K | c_v | stability | sample words |
|---|---:|---:|---:|---:|---|
| Australia news | 361 | 8 | 0.315 | 0.279 | health, review, school, report, department |
| Society | 292 | 8 | 0.404 | 0.283 | child, care, council, home, government |
| World news | 242 | 3 | 0.415 | 0.532 | child, work, tell, report, country |
| UK news | 122 | 8 | 0.307 | 0.213 | labour, case, inquiry, time, death |
| US news | 115 | 5 | 0.380 | 0.270 | trump, president, republican, state, white |
| Business | 115 | 5 | 0.343 | 0.348 | drug, trial, company, local, people |
| Politics | 111 | 8 | 0.315 | 0.243 | minister, school, need, prime, lead |
| Environment | 108 | 5 | 0.347 | 0.254 | food, study, shark, find, water |
| Education | 82 | 8 | 0.289 | 0.272 | university, student, government, high, new |
| Science | 79 | 3 | 0.320 | 0.363 | drug, patient, cancer, treatment, work |

Mean within-section stability: 0.306 (whole-corpus at K=25 was 0.22)

**Interpretation**: within-section stability is also modest; instability is not purely a heterogeneity artifact — it's partly inherent in corpus size + thematic mixing within sections.
