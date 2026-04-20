# Stage A Audit Report

## Row counts

| step | count |
|---|---:|
| raw seed rows (sum of 6 files) | (see merge log) |
| merged unique by URL | 3150 |
| articles in >1 method query | 23 |
| after G1 liveblog (−528) | 2622 |
| after G2 opinion (−309) | 2313 |
| after G3 sport (−92) | 2221 |
| after G4 culture (−164) | 2057 |
| after G5 lifestyle (−63) | 1994 |
| after G6 obituaries (−16) | 1978 |
| after G7 letters_archive (−9) | 1969 |
| after G8 shop (−14) | 1955 |
| after G9 briefings (−18) | 1937 |
| **survivors** | **1937** |

## Exclusion groups — detail

| group | rule | count | expected | OK | by-method |
|---|---|---:|---:|---|---|
| G1 | liveblog | 528 | 528±5 | ✓ | expert_qual=221, casestudy=92, gut=91, rct=81, prepost=40, expert_secondary=17 |
| G2 | opinion | 309 | 309±5 | ✓ | casestudy=134, expert_qual=67, gut=55, rct=50, prepost=2, expert_secondary=1 |
| G3 | sport | 92 | 92±5 | ✓ | expert_qual=30, casestudy=27, gut=23, rct=7, prepost=3, expert_secondary=2 |
| G4 | culture | 164 | 164±5 | ✓ | casestudy=122, expert_qual=24, gut=9, rct=7, prepost=2 |
| G5 | lifestyle | 63 | 63±5 | ✓ | rct=31, casestudy=17, gut=8, expert_qual=6, expert_secondary=1 |
| G6 | obituaries | 16 | 17±3 | ✓ | expert_qual=8, rct=6, casestudy=2 |
| G7 | letters_archive | 9 | 9±3 | ✓ | gut=6, rct=3 |
| G8 | shop | 14 | 14±3 | ✓ | casestudy=10, rct=4 |
| G9 | briefings | 18 | 17±3 | ✓ | expert_qual=9, casestudy=5, gut=3, rct=1 |

Cumulative excluded: 1213 (expected 1213±15)
Survivors: 1937 (expected 1937±15)

## Training-data overlap among survivors

- in_training = yes:  608
- in_training = no:   1329
- in_training = unknown: 0

## Survivors — section breakdown (top 20)

| section | count |
|---|---:|
| Australia news | 361 |
| Society | 292 |
| World news | 242 |
| UK news | 122 |
| US news | 115 |
| Business | 115 |
| Politics | 111 |
| Environment | 108 |
| Education | 82 |
| Science | 79 |
| Money | 54 |
| Media | 51 |
| Technology | 47 |
| Global development | 33 |
| Law | 30 |
| News | 19 |
| Healthcare Professionals Network | 11 |
| Public Leaders Network | 9 |
| Guardian Sustainable Business | 8 |
| Working in development | 6 |

(total distinct sections: 42)

## Survivors — word count distribution

- min: 71
- max: 9975
- mean: 1046.8
- p25: 640
- median: 827
- p75: 1146
- p90: 1686
- p99: 4565

## Discrepancies

None — every group within tolerance.
