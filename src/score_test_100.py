"""Score the 99-article test set with all 9 model versions and write raw
results to outputs/test_100_results.csv.

Does NOT re-score articles already present in the CSV — safe to re-run.
"""
from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("score100")

ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT / "outputs" / "test_set_100.csv"
OUT_CSV = ROOT / "outputs" / "test_100_results.csv"

MODELS = [
    ("m1_old", "model1_llm_judge"),
    ("m1_v4",  "model1_llm_judge_v4"),
    ("m1_v5",  "model1_llm_judge_v5"),
    ("m2_old", "model2_old"),
    ("m3_old", "model3_embedding_classifier"),
    ("m3_v4",  "model3_v4"),
    ("m4_old", "model4_hypothesis_classifier"),
    ("m4_v3",  "model4_v3"),
    ("m5",     "model5_deberta"),
]

DIMENSIONS = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]


def flatten(key: str, result: dict | None, row: dict) -> None:
    if result is None:
        row[f"{key}_decision"] = None
        row[f"{key}_max_method"] = None
        row[f"{key}_top_method"] = None
        row[f"{key}_high"] = 0
        return
    d_p1 = float(result.get("decision", {}).get("p1", 0))
    m_p1 = {k: float(result.get(k, {}).get("p1", 0))
            for k in DIMENSIONS if k != "decision"}
    max_m = max(m_p1.values()) if m_p1 else 0.0
    top_m = max(m_p1, key=m_p1.get) if m_p1 else None
    row[f"{key}_decision"] = d_p1
    row[f"{key}_max_method"] = max_m
    row[f"{key}_top_method"] = top_m
    row[f"{key}_high"] = int(d_p1 >= 0.75 and max_m >= 0.75)


def main():
    test = pd.read_csv(TEST_CSV)
    logger.info("Loaded %d test articles", len(test))
    logger.info("Buckets: %s", test["bucket"].value_counts().to_dict())

    # Load existing results if present (resume-safe)
    done = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame()
    done_by_url = {str(r["url"]): dict(r) for _, r in done.iterrows()} if not done.empty else {}
    logger.info("Resuming with %d already-scored rows", len(done_by_url))

    # Import models
    loaded = {}
    for key, mod_name in MODELS:
        try:
            loaded[key] = importlib.import_module(mod_name)
            logger.info("Loaded %s", mod_name)
        except Exception as e:
            logger.error("FAILED %s: %s", mod_name, e)

    rows = []
    for i, r in test.iterrows():
        url = str(r["url"])
        title = str(r["title"])
        body = str(r["body_excerpt"])
        row = {
            "url": url,
            "title": title[:100],
            "bucket": r["bucket"],
            "method_hint": r.get("method_hint", ""),
            "source": r.get("source", ""),
        }
        prior = done_by_url.get(url, {})
        for key, mod in loaded.items():
            high_col = f"{key}_high"
            if high_col in prior and not pd.isna(prior.get(high_col)):
                # carry forward
                for c in (f"{key}_decision", f"{key}_max_method",
                          f"{key}_top_method", f"{key}_high"):
                    row[c] = prior.get(c)
                continue
            logger.info("[%s] %d/%d %s", key, i + 1, len(test), r["bucket"])
            t0 = time.time()
            try:
                result = mod.score_article(title, body)
            except Exception as e:
                logger.warning("  failed: %s", e)
                result = None
            flatten(key, result, row)
            row[f"{key}_time_s"] = round(time.time() - t0, 2)
        rows.append(row)

        # Save incrementally every 10 rows
        if (i + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            logger.info("Checkpoint saved at %d rows", i + 1)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    logger.info("Final → %s", OUT_CSV)


if __name__ == "__main__":
    main()
