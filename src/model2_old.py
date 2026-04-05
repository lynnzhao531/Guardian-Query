from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

MODEL_NAME = "ft:gpt-4.1-mini-2025-04-14:personal::DAMjnKOH"

SYSTEM_PROMPT = (
    "You score Guardian newspaper articles for relevance to policy experimentation "
    "research on 7 dimensions. Each 0, 0.5, or 1. Use -1 for dimensions you cannot "
    "assess. Output JSON only. Keys: decision, method_rct, method_prepost, "
    "method_case_study, method_expert_qual, method_expert_secondary, method_gut"
)

DIMENSIONS = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]

PROB_LOOKUP = {
    0:   {"p0": 0.80, "p05": 0.15, "p1": 0.05},
    0.5: {"p0": 0.20, "p05": 0.60, "p1": 0.20},
    1:   {"p0": 0.05, "p05": 0.15, "p1": 0.80},
}

ZERO_RESULT = {
    dim: {"score": 0.0, "p0": 0.80, "p05": 0.15, "p1": 0.05} for dim in DIMENSIONS
}


def _parse_json(text: str) -> dict | None:
    """Extract JSON from response, handling ```json fenced blocks."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _call_model(title: str, body_text: str) -> dict | None:
    """Send a single request to the fine-tuned model and parse the result."""
    client = OpenAI(timeout=60.0)
    user_msg = f"Title: {title}\n\nBody:\n{body_text[:6000]}"
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=300,
        temperature=0,
    )
    return _parse_json(resp.choices[0].message.content or "")


def _build_result(raw: dict) -> dict:
    """Convert flat scores into nested format with probabilities."""
    result = {}
    for dim in DIMENSIONS:
        val = raw.get(dim)
        if val is None or val == -1:
            score = 0.0
            probs = PROB_LOOKUP[0]
        else:
            score = float(val)
            probs = PROB_LOOKUP.get(val, PROB_LOOKUP[0])
        result[dim] = {"score": score, **probs}
    return result


def score_article(title: str, body_text: str) -> dict:
    """Score an article using the fine-tuned model (without K*).

    Returns nested format: {"decision": {"score": 1.0, "p0": ..., "p05": ..., "p1": ...}, ...}
    On failure after one retry, returns all zeros with low confidence.
    """
    for _ in range(2):
        parsed = _call_model(title, body_text)
        if parsed is not None:
            return _build_result(parsed)
    return dict(ZERO_RESULT)


def is_available() -> bool:
    """Return True -- this model is always available."""
    return True
