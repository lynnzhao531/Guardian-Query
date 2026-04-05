"""Model 6 — Claude Haiku 4.5, designed to be independent from M1 (Sonnet).

Design choices that make this independent from M1:

  1. Only 3 K* hypotheses (the ones covering the widest method span), not 11.
     M1 uses all 11, so Haiku has a different prior surface.
  2. Cognitive framing: 4 small 0-10 sub-questions instead of M1's holistic
     rubric. Different scoring format → different failure modes.
  3. Few-shot calibration biased toward THIN methods (expert_qual,
     expert_secondary, gut) — the cases M1 struggles with.
  4. Explicit gut-decision branch: absence of evidence is treated as its own
     signal rather than a deficit.

Returns the standard 7-vector shape: {dim: {score, p0, p05, p1}}.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
TIMEOUT = 30.0
TEMPERATURE = 0.0
MAX_BODY_CHARS = 800  # Haiku context is cheap but excerpt kept short on purpose

DIMENSIONS = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]
METHOD_KEYS = ["rct", "prepost", "case_study",
               "expert_qual", "expert_secondary", "gut"]

# Three K* hypotheses covering the widest span of methods.
# Chosen for method breadth, not validation accuracy (K_star.json does not
# expose per-hypothesis metrics).
K_STAR_3 = [
    # H1 — covers RCT / prepost / case_study via peer-review framing
    "HIGH articles treat the study design itself as newsworthy and "
    "explicitly signal peer review or journal publication as "
    "legitimising the findings.",
    # H2 — covers expert_qual and expert_secondary via naming the body
    "HIGH articles explicitly name the evaluative body (citizens' jury, "
    "royal commission, expert panel, independent review, ombudsman).",
    # H11 — covers gut: absence-of-evidence as positive signal
    "Evidence or reasoning is conspicuously absent or thin relative to "
    "the scale of the decision — the article notes what was decided "
    "without documenting a formal analytical process beforehand. "
    "(This marks GUT decisions, not LOW articles.)",
]

SYSTEM_PROMPT = (
    "You are a careful classifier for articles about policy evaluation. "
    "You rate each article on four short numeric questions and pick one method. "
    "Use these three priors when scoring:\n\n"
    + "\n".join(f"  {i+1}. {h}" for i, h in enumerate(K_STAR_3))
    + "\n\nImportant: for GUT decisions, the ABSENCE of rigorous evidence IS "
    "the signal. An article about a decision being made WITHOUT consulting "
    "evidence should score method_specificity LOW but q4='gut' with high "
    "confidence. Do not confuse 'gut' with 'low-quality LOW article'.\n\n"
    "Output JSON only. No prose."
)

# Six calibration examples, one per method, biased toward thin methods
FEW_SHOT = [
    ("rct", 4,
     "Voters to be asked for ID in trials of system to combat electoral fraud",
     "Pilots framed as evidence-based approach to decide on rollout.",
     {"q1": 8, "q2": 7, "q3": 9, "q4": "rct"}),
    ("prepost", 4,
     "Four-day week: major breakthrough as most UK firms in trial make change permanent",
     "Before/after + decision to make permanent.",
     {"q1": 8, "q2": 8, "q3": 7, "q4": "prepost"}),
    ("case_study", 4,
     "Sarah's law roll-out begins after test run 'saves 60 from abuse'",
     "Pilot + evaluation + expansion decision.",
     {"q1": 7, "q2": 7, "q3": 5, "q4": "case_study"}),
    ("expert_qual", 4,
     "Citizens' jury in England backs assisted dying for terminally ill",
     "Qual deliberation intended to inform legislation/policy.",
     {"q1": 5, "q2": 6, "q3": 4, "q4": "expert_qual"}),
    ("expert_secondary", 4,
     "How States Use Data to Inform Decisions",
     "Admin/analytics used for decisions; less A/B.",
     {"q1": 5, "q2": 5, "q3": 6, "q4": "expert_secondary"}),
    ("gut", 3,
     "Minister scraps safety scheme despite civil service advice",
     "Decision made without formal analysis; absence of evidence IS the signal.",
     {"q1": 2, "q2": 7, "q3": 1, "q4": "gut"}),
]


def _fewshot_block() -> str:
    out = ["Calibration examples:"]
    for meth, rubric, title, note, ans in FEW_SHOT:
        out.append(
            f"[{meth}, rubric={rubric}] '{title}' — Expert note: '{note}' → "
            f"{json.dumps(ans)}"
        )
    return "\n".join(out)


def _build_user_prompt(title: str, body: str) -> str:
    excerpt = (body or "")[:MAX_BODY_CHARS]
    return (
        _fewshot_block() + "\n\n"
        "Now rate this article:\n\n"
        f"Title: {title}\n"
        f"Excerpt: {excerpt}\n\n"
        "Return JSON only with keys q1, q2, q3, q4:\n"
        "  q1: how clearly does it describe a policy being TESTED or EVALUATED? (0-10)\n"
        "  q2: how clearly does evidence LEAD TO a policy decision? (0-10)\n"
        "  q3: how specific is the evaluation METHOD described? (0-10)\n"
        "  q4: which method — one of "
        "rct/prepost/case_study/expert_qual/expert_secondary/gut/none/unclear"
    )


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            timeout=TIMEOUT,
        )
    return _client


def _call_haiku(title: str, body: str) -> dict:
    client = _get_client()
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=200,
            temperature=TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(title, body)}],
        )
    except Exception as e:
        logger.warning("Haiku API call failed: %s", e)
        return {"q1": 0, "q2": 0, "q3": 0, "q4": "none"}

    text = resp.content[0].text if resp.content else ""
    # Extract first JSON object in the response
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return {"q1": 0, "q2": 0, "q3": 0, "q4": "none"}
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return {"q1": 0, "q2": 0, "q3": 0, "q4": "none"}

    return {
        "q1": float(parsed.get("q1", 0)),
        "q2": float(parsed.get("q2", 0)),
        "q3": float(parsed.get("q3", 0)),
        "q4": str(parsed.get("q4", "none")).strip().lower(),
    }


def _map_to_7vec(q: dict) -> dict:
    """Map q1..q4 → 7-vector.

    decision_p1 = q2/10                            (evidence → decision link)
    method base = q1/10                            (evaluation presence)
    method boost = (q3/10) * 0.2                   (method specificity)
    if q4 in valid methods: method_q4 = min(1, base + boost)
    elif q4 == 'unclear' and q1 >= 3: all methods = q1/30
    else: all methods = base * 0.3  (dim signal without commitment)
    """
    decision_p1 = max(0.0, min(1.0, q["q1"] / 10.0 * 0.5 + q["q2"] / 10.0 * 0.5))
    base = q["q1"] / 10.0
    boost = (q["q3"] / 10.0) * 0.2
    q4 = q["q4"]

    method_p1 = {f"method_{k}": 0.0 for k in METHOD_KEYS}
    if q4 in METHOD_KEYS:
        method_p1[f"method_{q4}"] = max(0.0, min(1.0, base + boost))
    elif q4 == "unclear" and q["q1"] >= 3:
        for k in METHOD_KEYS:
            method_p1[f"method_{k}"] = q["q1"] / 30.0
    else:
        # q4 == 'none' or unrecognised — leave methods at 0
        pass

    def wrap(p1: float) -> dict:
        p1 = max(0.0, min(1.0, p1))
        remaining = 1.0 - p1
        if p1 >= 0.5:
            p0 = remaining * 0.3
            p05 = remaining * 0.7
        else:
            p0 = remaining * 0.7
            p05 = remaining * 0.3
        return {"score": p1, "p0": p0, "p05": p05, "p1": p1}

    out = {"decision": wrap(decision_p1)}
    for m, v in method_p1.items():
        out[m] = wrap(v)
    return out


def score_article(title: str, body: str) -> dict:
    q = _call_haiku(title, body)
    return _map_to_7vec(q)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    t0 = time.time()
    r = score_article(
        "App for being a better parent works, Oxford University study finds",
        "A randomised trial of 300 mothers at Oxford found the parenting app "
        "improved responsiveness to children. The NHS will now pilot it in "
        "three health trusts.",
    )
    print(json.dumps(r, indent=2))
    print(f"elapsed: {time.time()-t0:.1f}s")
