"""Model 1 v3 — K*-Guided LLM Judge with DECOMPOSED prompt (4 sub-questions).

Per REVISED_ARCHITECTURE.md §2 + §12: Model 1 should decompose its judgement
into 4 orthogonal sub-questions rather than asking for a single 7-vector in
one shot. This gives better HIGH recall by making the LLM reason explicitly
about each axis before committing to a score.

Sub-questions:
  Q1: EVALUATION — Did somebody actually evaluate policy here?
  Q2: COMPARISON — Is there a comparison between options / before & after / with vs without?
  Q3: DECISION — Was a policy choice made, announced, scrapped, expanded, or rejected?
  Q4: METHOD — Which evaluation method (if any) was used? (rct/prepost/case_study/expert_qual/expert_secondary/gut)

Mapping to 7-vector:
  decision = f(Q3, Q1)  — a policy decision requires both a choice and an evaluation
  method_X = f(Q4 if X selected, Q1, Q2)  — method present requires eval + comparison

This file does NOT overwrite src/model1_llm_judge.py. It is side-by-side
for A/B testing per Task 7.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List

import anthropic
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

KSTAR_PATH = PROJECT_ROOT / "knowledge_base" / "K_star.json"
MODEL = "claude-sonnet-4-6"
MAX_BODY_CHARS = 1200
MAX_RETRIES = 3
INITIAL_BACKOFF = 5

_api_stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

DIMENSIONS = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]

METHOD_KEYS = ["rct", "prepost", "case_study", "expert_qual", "expert_secondary", "gut", "none"]

# Probability mapping for final {0, 0.5, 1} discretisation
_PMAP = {
    0.0: {"score": 0.0, "p0": 0.80, "p05": 0.15, "p1": 0.05},
    0.5: {"score": 0.5, "p0": 0.20, "p05": 0.60, "p1": 0.20},
    1.0: {"score": 1.0, "p0": 0.05, "p05": 0.15, "p1": 0.80},
}


def _build_system_prompt(hypotheses: List[str]) -> str:
    hyp_block = "\n".join(f"  H{i+1}: {h}" for i, h in enumerate(hypotheses))
    return f"""You analyse UK policy journalism for the experiment-aversion research project.

Your task: given a Guardian article, answer FOUR orthogonal sub-questions. Each
sub-question is scored 0-10 independently. Do NOT conflate them.

## K* — validated signals (use as reasoning aids, not filters)
{hyp_block}

## The 4 sub-questions

Q1. EVALUATION (0-10)
    How clearly does this article describe somebody evaluating a policy option?
    0  = no evaluation mentioned
    3  = passing reference to some assessment
    7  = evaluation is a substantial part of the article
    10 = evaluation is THE central topic (results, numbers, conclusions)

Q2. COMPARISON (0-10)
    Does the article contain a comparison — treatment vs control, before vs after,
    option A vs option B, with vs without, or against a counterfactual?
    0  = no comparison
    5  = implicit comparison (e.g., "worse than expected")
    10 = explicit quantitative comparison between groups / timepoints

Q3. DECISION (0-10)
    Was a concrete policy action taken, announced, scrapped, expanded, or rejected?
    0  = purely descriptive, no decision
    5  = a decision is discussed or recommended but not taken
    10 = clear, attributed decision by an identifiable body (council, government, agency)

Q4. METHOD (one of: rct, prepost, case_study, expert_qual, expert_secondary, gut, none)
    What evaluation METHOD (if any) does the article describe?
    - rct                : randomised controlled trial, control group, field experiment
    - prepost            : before/after, baseline/follow-up, pre/post-intervention
    - case_study         : single-site narrative ("one council", "pilot area")
    - expert_qual        : expert panel, citizens' jury, public consultation, inquiry
    - expert_secondary   : administrative data, regression, natural experiment, cohort
    - gut                : decision made despite or without evidence
    - none               : no evaluation method discussed

## Output format

Return ONLY a JSON object:
{{"q1_evaluation": <0-10>, "q2_comparison": <0-10>, "q3_decision": <0-10>, "q4_method": "<one of the 7 keys>", "reasoning": "<1 short sentence>"}}
"""


def _build_user_prompt(title: str, body_text: str) -> str:
    snippet = (body_text or "")[:MAX_BODY_CHARS]
    return f"Title: {title}\n\nBody (first {MAX_BODY_CHARS} chars):\n{snippet}"


# ── Parsing ──────────────────────────────────────────────────────────────────

def _parse_response(text: str) -> Dict:
    t = text.strip()
    if "```" in t:
        s = t.find("{"); e = t.rfind("}") + 1
        t = t[s:e]
    obj = json.loads(t)
    q1 = int(obj.get("q1_evaluation", 0))
    q2 = int(obj.get("q2_comparison", 0))
    q3 = int(obj.get("q3_decision", 0))
    q4 = str(obj.get("q4_method", "none")).lower().strip()
    q1 = max(0, min(10, q1))
    q2 = max(0, min(10, q2))
    q3 = max(0, min(10, q3))
    if q4 not in METHOD_KEYS:
        q4 = "none"
    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4}


# ── Mapping to 7-vector ─────────────────────────────────────────────────────

def _decision_score(q1: int, q3: int) -> float:
    """Decision dimension requires: a decision AND an evaluation backing it."""
    # Normalised combined signal
    combined = (0.6 * q3 + 0.4 * q1) / 10.0
    # Comparison boost: if both high → 1.0
    if q3 >= 7 and q1 >= 5:
        return 1.0
    if combined >= 0.65:
        return 1.0
    if combined >= 0.30:
        return 0.5
    return 0.0


def _method_score(q1: int, q2: int, method_flag: bool) -> float:
    """A method dimension fires iff q4 selected it AND there's evaluation evidence."""
    if not method_flag:
        return 0.0
    combined = (0.5 * q1 + 0.5 * q2) / 10.0
    if q1 >= 6 and q2 >= 4:
        return 1.0
    if combined >= 0.60:
        return 1.0
    if combined >= 0.25:
        return 0.5
    return 0.0


def _gut_score(q1: int, q2: int, q3: int, method_flag: bool) -> float:
    """Gut: decision made but LOW evaluation/comparison. ABSENCE is the signal."""
    if not method_flag:
        return 0.0
    # Need a decision, low evaluation and low comparison
    if q3 >= 6 and q1 <= 4 and q2 <= 4:
        return 1.0
    if q3 >= 4 and q1 <= 5:
        return 0.5
    return 0.0


def _to_7vector(parsed: Dict) -> Dict[str, dict]:
    q1, q2, q3, q4 = parsed["q1"], parsed["q2"], parsed["q3"], parsed["q4"]

    out: Dict[str, float] = {d: 0.0 for d in DIMENSIONS}
    out["decision"] = _decision_score(q1, q3)

    method_to_dim = {
        "rct": "method_rct",
        "prepost": "method_prepost",
        "case_study": "method_case_study",
        "expert_qual": "method_expert_qual",
        "expert_secondary": "method_expert_secondary",
    }
    if q4 in method_to_dim:
        out[method_to_dim[q4]] = _method_score(q1, q2, True)
    elif q4 == "gut":
        out["method_gut"] = _gut_score(q1, q2, q3, True)

    # Map to {score,p0,p05,p1}
    result = {}
    for d, s in out.items():
        result[d] = dict(_PMAP[s])
    return result


# ── Main scoring function ───────────────────────────────────────────────────

def score_article(title: str, body_text: str) -> dict:
    with open(KSTAR_PATH, encoding="utf-8") as f:
        hypotheses = json.load(f)["hypotheses"]

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0)
    system_prompt = _build_system_prompt(hypotheses)
    user_prompt = _build_user_prompt(title, body_text)

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=400,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            _api_stats["calls"] += 1
            _api_stats["input_tokens"] += resp.usage.input_tokens
            _api_stats["output_tokens"] += resp.usage.output_tokens

            parsed = _parse_response(resp.content[0].text)
            return _to_7vector(parsed)
        except anthropic.APIStatusError as e:
            last_err = e
            wait = 30 if e.status_code == 529 else INITIAL_BACKOFF * (2 ** attempt)
            time.sleep(wait)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_err = e
            time.sleep(INITIAL_BACKOFF * (2 ** attempt))

    raise RuntimeError(f"Model1 v3 failed after {MAX_RETRIES} retries: {last_err}")


def get_api_stats() -> dict:
    return dict(_api_stats)


def reset_api_stats():
    for k in _api_stats:
        _api_stats[k] = 0


if __name__ == "__main__":
    title = "Stop and breathe: police staff offered meditation lessons"
    body = ("Meditation lessons aimed at reducing stress will be made available "
            "to all 200,000 police staff in England and Wales after a trial "
            "across five forces found the practice improved average wellbeing, "
            "life satisfaction, resilience and work performance.")
    print(json.dumps(score_article(title, body), indent=2))
    print(get_api_stats())
