# Part 2 — Preprocessing variants (K=20, 3 seeds)

| variant | vocab | c_v mean±std | c_npmi | stability (mean / median) |
|---|---:|---:|---:|---:|
| P1 | 13867 | 0.4495±0.0210 | 0.0130 | 0.208 / 0.205 |
| P2 | 10053 | 0.4571±0.0121 | 0.0274 | 0.234 / 0.227 |
| P2b | 10046 | 0.4603±0.0078 | 0.0271 | 0.228 / 0.234 |
| P3 | 10672 | 0.4351±0.0168 | 0.0184 | 0.188 / 0.181 |
| P4 | 9605 | 0.4563±0.0031 | 0.0269 | 0.229 / 0.220 |

**Selected: P2** — prefer higher stability (v2 objective)

P2b added corpus-specific stopwords:
year, people, time, new, work, include, need, use, find, come, know, way, day, help, good, think, look, want
