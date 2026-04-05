"""Rescore the 99-article test set with the Layer-2 continuous models.

Only re-runs M3, M4-v3, M5 (the local non-LLM models whose discretization
was just removed). M1/M2 scores are reused from test_100_results.csv.

Output: outputs/test_100_continuous.csv with columns
  {url, title, bucket, method_hint,
   m3c_decision, m3c_max_method, m3c_top_method,
   m4vc_decision, m4vc_max_method, m4vc_top_method,
   m5c_decision, m5c_max_method, m5c_top_method}
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TEST_CSV = ROOT / "outputs" / "test_set_100.csv"
OUT_CSV  = ROOT / "outputs" / "test_100_continuous.csv"

METHODS = ["method_rct", "method_prepost", "method_case_study",
           "method_expert_qual", "method_expert_secondary", "method_gut"]


def flatten(prefix: str, result: dict, row: dict) -> None:
    d = float(result["decision"]["p1"])
    meth_p1 = {m: float(result[m]["p1"]) for m in METHODS}
    top_m = max(meth_p1, key=meth_p1.get)
    row[f"{prefix}_decision"]    = d
    row[f"{prefix}_max_method"]  = meth_p1[top_m]
    row[f"{prefix}_top_method"]  = top_m


def main():
    test = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(test)} test articles")

    import model3_embedding_classifier as m3
    import model4_v3 as m4v
    import model5_deberta as m5
    print("Models imported")

    rows = []
    t_start = time.time()
    for i, r in test.iterrows():
        title = str(r["title"])
        body = str(r["body_excerpt"])
        row = {
            "url": str(r["url"]),
            "title": title[:100],
            "bucket": r["bucket"],
            "method_hint": r.get("method_hint", ""),
        }
        try:
            flatten("m3c",  m3.score_article(title, body), row)
        except Exception as e:
            print(f"  M3 fail @{i}: {e}")
        try:
            flatten("m4vc", m4v.score_article(title, body), row)
        except Exception as e:
            print(f"  M4v3 fail @{i}: {e}")
        try:
            flatten("m5c",  m5.score_article(title, body), row)
        except Exception as e:
            print(f"  M5 fail @{i}: {e}")
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test)} ({time.time()-t_start:.0f}s)")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({time.time()-t_start:.0f}s total)")


if __name__ == "__main__":
    main()
