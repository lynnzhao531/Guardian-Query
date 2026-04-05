"""Score the test set with M6 Haiku only. Output: outputs/test_100_m6.csv"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TEST_CSV = ROOT / "outputs" / "test_set_100.csv"
OUT_CSV  = ROOT / "outputs" / "test_100_m6.csv"

METHODS = ["method_rct", "method_prepost", "method_case_study",
           "method_expert_qual", "method_expert_secondary", "method_gut"]


def main():
    import model6_haiku as m6

    test = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(test)} test articles")

    rows = []
    t_start = time.time()
    for i, r in test.iterrows():
        title = str(r["title"])
        body = str(r["body_excerpt"])
        try:
            res = m6.score_article(title, body)
            d_p1 = float(res["decision"]["p1"])
            m_p1 = {m: float(res[m]["p1"]) for m in METHODS}
            top_m = max(m_p1, key=m_p1.get)
        except Exception as e:
            print(f"  fail @{i}: {e}")
            d_p1 = 0.0
            m_p1 = {m: 0.0 for m in METHODS}
            top_m = METHODS[0]

        rows.append({
            "url": str(r["url"]),
            "title": title[:100],
            "bucket": r["bucket"],
            "method_hint": r.get("method_hint", ""),
            "m6_decision": d_p1,
            "m6_max_method": m_p1[top_m],
            "m6_top_method": top_m,
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  {i+1}/{len(test)} ({elapsed:.0f}s, avg {elapsed/(i+1):.2f}s/article)")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
