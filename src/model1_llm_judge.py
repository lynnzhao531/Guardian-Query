from typing import Optional, List, Dict
"""Model 1: K*-Guided LLM Judge (Claude Sonnet) — MASTER_PLAN_v3.md §5.1.

Scores articles on 7-vector using K* hypotheses as guidance.
Engine: claude-sonnet-4-6 (upgraded from Haiku for better HIGH recall).
"""

import json
import os
import time
import csv
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KSTAR_PATH = PROJECT_ROOT / "knowledge_base" / "K_star.json"
MODEL = "claude-sonnet-4-6"
MAX_BODY_CHARS = 800
MAX_RETRIES = 3
INITIAL_BACKOFF = 5  # seconds

# Cost tracking (accumulated across calls within this process)
_api_stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

# ---------- score mapping ----------

_SCORE_MAP = {
    0: {"score": 0.0, "p0": 0.80, "p05": 0.15, "p1": 0.05},
    1: {"score": 0.0, "p0": 0.80, "p05": 0.15, "p1": 0.05},
    2: {"score": 0.5, "p0": 0.20, "p05": 0.60, "p1": 0.20},
    3: {"score": 1.0, "p0": 0.05, "p05": 0.15, "p1": 0.80},
    4: {"score": 1.0, "p0": 0.05, "p05": 0.15, "p1": 0.80},
    5: {"score": 1.0, "p0": 0.05, "p05": 0.15, "p1": 0.80},
}

DIMENSIONS = [
    "decision", "method_rct", "method_prepost", "method_case_study",
    "method_expert_qual", "method_expert_secondary", "method_gut",
]

# ---------- few-shot examples ----------

_GOLD_FILES = {
    "method_rct": ("rct.csv", "body"),
    "method_prepost": ("prepost.csv", "article_body"),
    "method_case_study": ("casestudy.csv", "article_body"),
    "method_expert_qual": ("expert_qual.csv", "body_text"),
    "method_expert_secondary": ("expert_secondary_quant.csv", "body_text"),
    "method_gut": ("gut_decision.csv", "body_text"),
}


def _load_few_shot_examples() -> str:
    """Load 1 example per method type from gold CSVs."""
    lines = []
    data_dir = PROJECT_ROOT / "data"
    for method, (filename, body_col) in _GOLD_FILES.items():
        path = data_dir / filename
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if row is None:
                continue
        title = row.get("title", "")
        body_snippet = (row.get(body_col) or "")[:200]
        # Build expected scores: HIGH on this method, LOW on others
        scores = {d: 0 for d in DIMENSIONS}
        scores["decision"] = 4
        scores[method] = 5
        lines.append(
            f'Example ({method}):\n'
            f'  Title: "{title}"\n'
            f'  Body snippet: "{body_snippet}..."\n'
            f'  Expected: {json.dumps(scores)}'
        )
    return "\n\n".join(lines)


# ---------- prompt builders ----------

def _build_system_prompt(hypotheses: List[str]) -> str:
    """Build system prompt with K* hypotheses grouped by method type."""
    few_shot = _load_few_shot_examples()

    # Group hypotheses by theme for clarity
    hyp_block = "\n".join(f"  H{i+1}: {h}" for i, h in enumerate(hypotheses))

    return f"""You are an expert classifier for experiment-aversion research.

Your task: given a Guardian article, score it on 7 dimensions (0-5 integer scale).

## K* Hypotheses (validated signals that distinguish HIGH from LOW articles)
{hyp_block}

## Dimension definitions

- decision: Overall, does this article describe a policy decision where evidence
  was used, ignored, or absent? (0=no policy decision, 5=clear experiment-aversion)
- method_rct: Does the article describe or reference a randomised controlled trial?
  (0=none, 5=RCT is the central topic)
- method_prepost: Does the article describe a pre-post or before-after evaluation?
  (0=none, 5=central topic)
- method_case_study: Does the article present a case study of a specific policy
  intervention and its outcomes? (0=none, 5=central topic)
- method_expert_qual: Does the article feature qualitative expert evaluation such
  as a citizens' jury, royal commission, expert panel, public inquiry, or ombudsman?
  (0=none, 5=central topic)
- method_expert_secondary: Does the article use secondary/published quantitative
  data (surveys, hospital records, existing datasets) as the evidentiary foundation?
  (0=none, 5=central topic)
- method_gut: Does the article describe a gut decision — one where evidence or
  formal reasoning is conspicuously ABSENT relative to the scale of the decision?
  NOTE: The ABSENCE of rigorous evidence IS the signal for this dimension. Look for
  decisions that name an organisation making a choice without citing formal analysis.
  (0=no gut decision, 5=clear gut decision with absent evidence)

## Few-shot examples
{few_shot}

## Output format
Return ONLY a JSON object with the 7 keys and integer values 0-5. No explanation.
Example: {{"decision": 3, "method_rct": 0, "method_prepost": 4, "method_case_study": 0, "method_expert_qual": 0, "method_expert_secondary": 0, "method_gut": 0}}"""


def _build_user_prompt(title: str, body_text: str) -> str:
    """Build user prompt with title + truncated body."""
    snippet = body_text[:MAX_BODY_CHARS] if body_text else ""
    return f"Title: {title}\n\nBody (first {MAX_BODY_CHARS} chars):\n{snippet}"


# ---------- response parsing ----------

def _parse_7vector(response_text: str) -> Dict[str, int]:
    """Parse model response into 7-vector of raw 0-5 integers."""
    text = response_text.strip()
    # Extract JSON from possible markdown fences
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]
    raw = json.loads(text)
    result = {}
    for dim in DIMENSIONS:
        val = int(raw[dim])
        if val < 0 or val > 5:
            raise ValueError(f"{dim} out of range: {val}")
        result[dim] = val
    return result


def _map_to_probabilities(raw_scores: Dict[str, int]) -> Dict[str, dict]:
    """Map raw 0-5 scores to discrete {0, 0.5, 1} + probabilities."""
    output = {}
    for dim in DIMENSIONS:
        val = raw_scores[dim]
        mapping = _SCORE_MAP[val]
        output[dim] = {
            "score": mapping["score"],
            "p0": mapping["p0"],
            "p05": mapping["p05"],
            "p1": mapping["p1"],
        }
    return output


# ---------- main scoring function ----------

def score_article(title: str, body_text: str) -> dict:
    """Score an article on the 7-vector using Claude Sonnet with K* guidance.

    Returns dict with keys: decision, method_rct, method_prepost,
    method_case_study, method_expert_qual, method_expert_secondary, method_gut.
    Each value is {"score": float, "p0": float, "p05": float, "p1": float}.
    """
    # Load K* hypotheses
    with open(KSTAR_PATH, encoding="utf-8") as f:
        kstar = json.load(f)
    hypotheses = kstar["hypotheses"]

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0)
    system_prompt = _build_system_prompt(hypotheses)
    user_prompt = _build_user_prompt(title, body_text)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            # Track usage
            _api_stats["calls"] += 1
            _api_stats["input_tokens"] += response.usage.input_tokens
            _api_stats["output_tokens"] += response.usage.output_tokens

            text = response.content[0].text
            raw_scores = _parse_7vector(text)
            return _map_to_probabilities(raw_scores)

        except anthropic.APIStatusError as e:
            last_error = e
            if e.status_code == 529:  # Overloaded
                wait = 30
            else:
                wait = INITIAL_BACKOFF * (2 ** attempt)
            time.sleep(wait)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_error = e
            wait = INITIAL_BACKOFF * (2 ** attempt)
            time.sleep(wait)

    raise RuntimeError(
        f"Model 1 failed after {MAX_RETRIES} retries. Last error: {last_error}"
    )


def get_api_stats() -> dict:
    """Return cumulative API usage stats for cost accounting."""
    return dict(_api_stats)


def reset_api_stats():
    """Reset the API usage counters."""
    _api_stats["calls"] = 0
    _api_stats["input_tokens"] = 0
    _api_stats["output_tokens"] = 0


# ---------- CLI test ----------

if __name__ == "__main__":
    test_title = "Stop and breathe: police staff offered meditation lessons"
    test_body = (
        "Meditation lessons aimed at reducing stress will be made available "
        "to all 200,000 police staff in England and Wales after a trial "
        "across five forces found the practice improved average wellbeing, "
        "life satisfaction, resilience and work performance."
    )
    result = score_article(test_title, test_body)
    print(json.dumps(result, indent=2))
    print(f"\nAPI stats: {get_api_stats()}")
