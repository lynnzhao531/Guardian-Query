"""Model 1 v5 — same holistic 0-5 rubric as M1 v2, plus 3 calibration examples.

System prompt and output format are identical to model1_llm_judge (v2). The
only change is that the user message is prefixed with three few-shot
calibration examples drawn from data/Training_cases.csv (HIGH RCT / MID
expert_qual / LOW gut). Expert 'notes' are used as the human-written reasoning.

Same public API: `score_article(title, body) -> 7-vector`.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import anthropic
import pandas as pd
from dotenv import load_dotenv

from model1_llm_judge import (
    _build_system_prompt,
    _parse_7vector,
    _map_to_probabilities,
    KSTAR_PATH,
    DIMENSIONS,
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

ROOT = Path(__file__).resolve().parents[1]
TRAINING_CASES = ROOT / "data" / "Training_cases.csv"
MODEL = "claude-sonnet-4-6"
MAX_BODY_CHARS = 800
MAX_RETRIES = 3
INITIAL_BACKOFF = 5

_api_stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
_cached_examples: str | None = None


def _pick_calibration_examples() -> str:
    """Return a 3-shot calibration block from Training_cases.csv."""
    global _cached_examples
    if _cached_examples is not None:
        return _cached_examples
    df = pd.read_csv(TRAINING_CASES)
    high = df[(df["method_category"] == "RCT_Field_AB")
              & (df["rubric_score_0to5"] >= 4)].head(1)
    mid = df[(df["method_category"] == "Expert_Qualitative")
             & (df["rubric_score_0to5"].between(2, 3))].head(1)
    low = df[(df["method_category"] == "Gut_NoLabel")
             & (df["rubric_score_0to5"] <= 1)].head(1)

    def _fmt(row, label: str, score_line: str) -> str:
        r = row.iloc[0]
        return (
            f"EXAMPLE ({label}): '{str(r['title'])[:180]}'\n"
            f"Why: {str(r.get('notes','')).strip()}\n"
            f"Score: {score_line}"
        )

    parts = ["Here are calibration examples:\n"]
    parts.append(_fmt(high, "HIGH relevance",  "decision=1, method_rct=1"))
    parts.append(_fmt(mid,  "MEDIUM",          "decision=0.5, method_expert_qual=0.5"))
    parts.append(_fmt(low,  "LOW",             "decision=0, all methods=0"))
    _cached_examples = "\n\n".join(parts)
    return _cached_examples


def _build_user_prompt(title: str, body: str) -> str:
    snippet = (body or "")[:MAX_BODY_CHARS]
    examples = _pick_calibration_examples()
    return (
        f"{examples}\n\n"
        "Now score this article:\n"
        f"Title: {title or ''}\n"
        f"Excerpt: {snippet}"
    )


def score_article(title: str, body: str) -> dict:
    with open(KSTAR_PATH, encoding="utf-8") as f:
        kstar = json.load(f)
    hypotheses = kstar["hypotheses"]

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0)
    system_prompt = _build_system_prompt(hypotheses)
    user_prompt = _build_user_prompt(title, body)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=256,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            _api_stats["calls"] += 1
            _api_stats["input_tokens"] += response.usage.input_tokens
            _api_stats["output_tokens"] += response.usage.output_tokens
            raw = _parse_7vector(response.content[0].text)
            return _map_to_probabilities(raw)
        except anthropic.APIStatusError as e:
            last_error = e
            wait = 30 if e.status_code == 529 else INITIAL_BACKOFF * (2 ** attempt)
            time.sleep(wait)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_error = e
            time.sleep(INITIAL_BACKOFF * (2 ** attempt))

    raise RuntimeError(f"Model 1 v5 failed after {MAX_RETRIES} retries: {last_error}")


if __name__ == "__main__":
    r = score_article(
        "Stop and breathe: police staff offered meditation lessons",
        "Meditation lessons will be made available to all 200,000 police staff "
        "after a trial across five forces found the practice improved wellbeing.",
    )
    print(json.dumps(r, indent=2))
