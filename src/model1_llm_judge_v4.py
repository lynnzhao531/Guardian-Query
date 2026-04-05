"""Model 1 v4 — continuous 0-10 sub-question rating.

Uses the same K* system prompt as the current Model 1 (v2), but replaces the
holistic 0-5 rubric with four continuous sub-questions. Mapping:
  decision        = q3 / 10
  method (q4)     = min(1.0, q1/10 + (q2/10)*0.2)
  q4='unclear'    → spread q1/30 across all methods (weak)

Same public API `score_article(title, body) -> 7-vector` as model1_llm_judge.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import anthropic
from dotenv import load_dotenv

# Reuse the K*-guided system prompt builder + few-shot loader from v2.
from model1_llm_judge import (
    _build_system_prompt,
    DIMENSIONS,
    KSTAR_PATH,
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

MODEL = "claude-sonnet-4-6"
MAX_BODY_CHARS = 800
MAX_RETRIES = 3
INITIAL_BACKOFF = 5

_VALID_METHODS = {
    "rct", "prepost", "case_study", "expert_qual", "expert_secondary", "gut",
}

_api_stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


def _build_user_prompt(title: str, body: str) -> str:
    snippet = (body or "")[:MAX_BODY_CHARS]
    return (
        "For this article, rate your confidence 0-10 on each question:\n"
        "Q1: How clearly does it describe a policy being TESTED or EVALUATED?\n"
        "    (0=not at all, 5=vaguely mentioned, 10=explicitly described)\n"
        "Q2: How clearly is there a COMPARISON of options or outcomes?\n"
        "    (0=none, 5=implied, 10=explicit A vs B or before/after)\n"
        "Q3: How clearly is a DECISION linked to evidence or evaluation?\n"
        "    (0=no link, 5=implied link, 10=decision explicitly follows from evidence)\n"
        "Q4: Which method? rct/prepost/case_study/expert_qual/expert_secondary/gut/none/unclear\n\n"
        f"Title: {title or ''}\n"
        f"Excerpt: {snippet}\n\n"
        "Answer JSON only: {\"q1\":N,\"q2\":N,\"q3\":N,\"q4\":\"...\"}"
    )


def _parse_response(text: str) -> Dict:
    t = text.strip()
    if "```" in t:
        s, e = t.find("{"), t.rfind("}") + 1
        t = t[s:e]
    raw = json.loads(t)
    return {
        "q1": float(raw.get("q1", 0)),
        "q2": float(raw.get("q2", 0)),
        "q3": float(raw.get("q3", 0)),
        "q4": str(raw.get("q4", "unclear")).strip().lower(),
    }


def _probs_for(score: float) -> dict:
    """Turn a continuous [0,1] score into the discrete {0,0.5,1} + probabilities."""
    s = max(0.0, min(1.0, float(score)))
    if s >= 0.75:
        return {"score": 1.0, "p0": 0.05, "p05": 0.15, "p1": 0.80}
    if s >= 0.25:
        return {"score": 0.5, "p0": 0.20, "p05": 0.60, "p1": 0.20}
    return {"score": 0.0, "p0": 0.80, "p05": 0.15, "p1": 0.05}


def _to_7vector(parsed: Dict) -> Dict[str, dict]:
    q1, q2, q3 = parsed["q1"], parsed["q2"], parsed["q3"]
    q4 = parsed["q4"]

    dec = q3 / 10.0
    methods = {f"method_{m}": 0.0 for m in _VALID_METHODS}

    if q4 in _VALID_METHODS:
        base = q1 / 10.0
        boost = (q2 / 10.0) * 0.2
        methods[f"method_{q4}"] = min(1.0, base + boost)
    elif q4 == "unclear" and q1 >= 3:
        weak = q1 / 30.0
        for k in methods:
            methods[k] = weak
    # 'none' → all zeros

    out: Dict[str, dict] = {"decision": _probs_for(dec)}
    for k, v in methods.items():
        out[k] = _probs_for(v)
    # Make sure every dimension is present in the canonical order
    return {d: out.get(d, _probs_for(0.0)) for d in DIMENSIONS}


def score_article(title: str, body: str) -> dict:
    """Score an article via v4 continuous sub-questions. Returns 7-vector dict."""
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
                max_tokens=128,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            _api_stats["calls"] += 1
            _api_stats["input_tokens"] += response.usage.input_tokens
            _api_stats["output_tokens"] += response.usage.output_tokens
            parsed = _parse_response(response.content[0].text)
            return _to_7vector(parsed)
        except anthropic.APIStatusError as e:
            last_error = e
            wait = 30 if e.status_code == 529 else INITIAL_BACKOFF * (2 ** attempt)
            time.sleep(wait)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_error = e
            time.sleep(INITIAL_BACKOFF * (2 ** attempt))

    raise RuntimeError(f"Model 1 v4 failed after {MAX_RETRIES} retries: {last_error}")


if __name__ == "__main__":
    r = score_article(
        "Stop and breathe: police staff offered meditation lessons",
        "Meditation lessons will be made available to all 200,000 police staff "
        "after a trial across five forces found the practice improved wellbeing.",
    )
    print(json.dumps(r, indent=2))
