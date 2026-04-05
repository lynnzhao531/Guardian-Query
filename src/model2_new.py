from __future__ import annotations
"""Model 2-new: K*-guided fine-tuned GPT-4.1-mini scorer.

Wrapper for the newer fine-tuned model (gpt-4.1-mini) that scores
Guardian articles on the 7-vector using K* hypotheses.
"""
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

KSTAR_PATH = ROOT / "knowledge_base" / "K_star.json"
STATE_PATH = ROOT / "project_state" / "STATE.json"
STATE_KEY = "new_finetuned_model_41mini"
MAX_BODY = 800

DIMS = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]

_PROB = {
    -1:  {"p0": 0.80, "p05": 0.15, "p1": 0.05},   # unscored → treat as LOW
    0:   {"p0": 0.80, "p05": 0.15, "p1": 0.05},
    0.5: {"p0": 0.20, "p05": 0.60, "p1": 0.20},
    1:   {"p0": 0.05, "p05": 0.15, "p1": 0.80},
}

_ZERO_VEC = {d: {"score": 0.0, "p0": 0.80, "p05": 0.15, "p1": 0.05} for d in DIMS}


def _load_kstar() -> list[str]:
    with open(KSTAR_PATH, encoding="utf-8") as f:
        return json.load(f)["hypotheses"]


def _load_state() -> dict:
    with open(STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _build_system_prompt(hypotheses: list[str]) -> str:
    """Build system prompt matching EXACT training format (flat H1-H11, -1 option)."""
    hyp_lines = [f"  H{i+1}: {h}" for i, h in enumerate(hypotheses)]
    hyp_block = "\n".join(hyp_lines)
    return (
        "You are an expert classifier for experiment-aversion research.\n"
        "Score the Guardian article on 7 dimensions using {0, 0.5, 1} or -1 if unscored.\n\n"
        f"## K* Hypotheses (validated knowledge)\n{hyp_block}\n\n"
        f"## Dimensions\n{', '.join(DIMS)}\n\n"
        "Return ONLY a JSON object with these 7 keys. Each value is 0, 0.5, 1, or -1.\n"
        "Use -1 for dimensions without evidence."
    )


def _parse_json(txt: str) -> dict:
    """Extract JSON from model response, handling ```json blocks."""
    txt = txt.strip()
    if "```" in txt:
        start = txt.find("{")
        end = txt.rfind("}") + 1
        if start >= 0 and end > start:
            txt = txt[start:end]
    return json.loads(txt)


def _unwrap_scores(raw: dict) -> dict:
    """Handle both training format {"scores": {...}, "reasoning": "..."} and flat format."""
    if "scores" in raw and isinstance(raw["scores"], dict):
        return raw["scores"]
    return raw


def _raw_to_vector(raw: dict) -> dict:
    scores = _unwrap_scores(raw)
    out = {}
    for dim in DIMS:
        v = float(scores.get(dim, 0))
        if v not in (-1, 0, 0.5, 1):
            v = 0
        # Map -1 (unscored) to 0 for scoring purposes
        score_val = 0.0 if v == -1 else v
        prob_key = v if v in _PROB else 0
        out[dim] = {"score": score_val, **_PROB[prob_key]}
    return out


def score_article(title: str, body_text: str) -> dict:
    """Score an article via the new fine-tuned model.

    Returns nested dict: {dim: {"score", "p0", "p05", "p1"}} for each of
    the 7 dimensions.
    """
    state = _load_state()
    model_name = state.get(STATE_KEY)
    if not model_name:
        raise RuntimeError(f"Model 2-new not available: {STATE_KEY} missing from STATE.json")

    client = OpenAI(timeout=60.0)
    system = _build_system_prompt(_load_kstar())
    user_msg = f"Title: {title}\nExcerpt: {body_text[:MAX_BODY]}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                max_tokens=300,
                temperature=0,
                messages=messages,
            )
            txt = resp.choices[0].message.content
            raw = _parse_json(txt)
            return _raw_to_vector(raw)
        except (json.JSONDecodeError, KeyError, TypeError):
            if attempt == 0:
                continue
            return dict(_ZERO_VEC)
        except Exception:
            return dict(_ZERO_VEC)


def is_available() -> bool:
    """Check whether the new fine-tuned model name is present and non-null."""
    try:
        state = _load_state()
        return bool(state.get(STATE_KEY))
    except Exception:
        return False


if __name__ == "__main__":
    print(f"Model 2-new available: {is_available()}")
    if is_available():
        state = _load_state()
        print(f"Model: {state.get(STATE_KEY)}")
