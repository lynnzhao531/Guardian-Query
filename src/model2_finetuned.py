from __future__ import annotations
"""Model 2: K*-Fine-Tuned GPT (gpt-4o-mini) -- MASTER_PLAN_v3.md SS5.2 / SS6.

Builds preference-pair training data from scored + gold CSVs,
formats OpenAI fine-tuning JSONL, and scores articles via the
fine-tuned model using the 7-vector output format.
"""
import csv, json, random, time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ROOT = Path(__file__).resolve().parents[1]
KSTAR_PATH = ROOT / "knowledge_base" / "K_star.json"
STATE_PATH = ROOT / "project_state" / "STATE.json"
DATA_DIR, KB_DIR = ROOT / "data", ROOT / "knowledge_base"
TRAIN_PATH = KB_DIR / "openai_training_7vec.jsonl"
VAL_PATH = KB_DIR / "openai_validation_7vec.jsonl"
BASE_MODEL, EPOCHS, MAX_BODY, POLL_SEC = "gpt-4o-mini-2024-07-18", 3, 800, 60

DIMS = ["decision", "method_rct", "method_prepost", "method_case_study",
        "method_expert_qual", "method_expert_secondary", "method_gut"]
SCORED = [("rct 2.csv", "method_rct", "Method", "Decision"),
          ("prepost 2.csv", "method_prepost", "Method", "Decision"),
          ("case studies.csv", "method_case_study", "Method", "Decision"),
          ("quantitative.csv", "method_expert_secondary", "Method", "Decision"),
          ("gut.csv", "method_gut", "Method", "Decision")]
GOLD = [("rct.csv", "method_rct"), ("prepost.csv", "method_prepost"),
        ("casestudy.csv", "method_case_study"),
        ("expert_secondary_quant.csv", "method_expert_secondary"),
        ("expert_qual.csv", "method_expert_qual"),
        ("gut_decision.csv", "method_gut")]
BODY_COLS = ["body", "article_body", "body_text", "bodyText"]
_PROB = {0: {"p0": .80, "p05": .15, "p1": .05},
         0.5: {"p0": .20, "p05": .60, "p1": .20},
         1: {"p0": .05, "p05": .15, "p1": .80}}

# ---- helpers ----
def _kstar():
    with open(KSTAR_PATH, encoding="utf-8") as f:
        return json.load(f)["hypotheses"]

def _state():
    with open(STATE_PATH, encoding="utf-8") as f:
        return json.load(f)

def _save(s):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)

def _csv(fn):
    p = DATA_DIR / fn
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def _body(row):
    for c in BODY_COLS:
        if c in row and row[c]:
            return row[c]
    return ""

def _ps(val):
    """Parse score string -> {0, 0.5, 1} or None."""
    if val is None or str(val).strip() == "":
        return None
    v = float(val)
    return {0: 0, 0.5: 0.5, 1: 1}.get(v)

def _art(row, mdim, ms, ds):
    sc = {d: -1 for d in DIMS}
    sc[mdim] = ms
    sc["decision"] = ds if ds is not None else -1
    return {"title": row.get("title", ""), "body": _body(row), "scores": sc}

# ---- preference pairs ----
def build_preference_pairs() -> list:
    """Build preference pairs per method dimension. Prints counts."""
    pairs = []
    counts = {}
    # Step 1: scored files -- bucket by method score, pair higher vs lower
    for fn, mdim, mcol, dcol in SCORED:
        rows = _csv(fn)
        bkt = {0: [], 0.5: [], 1: []}
        for r in rows:
            ms, ds = _ps(r.get(mcol)), _ps(r.get(dcol))
            if ms is None or not r.get("title"):
                continue
            bkt[ms].append(_art(r, mdim, ms, ds))
        n = 0
        for hi, lo in [(1, 0.5), (1, 0), (0.5, 0)]:
            for w in bkt[hi]:
                for la in random.sample(bkt[lo], min(3, len(bkt[lo]))):
                    pairs.append({"winner": w, "loser": la, "dimension": mdim})
                    n += 1
        counts[mdim] = n
    # Step 2: gold winners vs zero-scored losers
    zeros = []
    for fn, mdim, mcol, dcol in SCORED:
        for r in _csv(fn):
            if _ps(r.get(mcol)) == 0 and r.get("title"):
                zeros.append(_art(r, mdim, 0, _ps(r.get(dcol))))
    for fn, mdim in GOLD:
        n = 0
        for r in _csv(fn):
            if not r.get("title"):
                continue
            sc = {d: -1 for d in DIMS}
            sc[mdim] = 1
            sc["decision"] = 1
            w = {"title": r["title"], "body": _body(r), "scores": sc}
            for la in random.sample(zeros, min(3, len(zeros))):
                pairs.append({"winner": w, "loser": la, "dimension": mdim})
                n += 1
        counts[f"{mdim}_gold"] = n
    # Step 3: decision dimension -- pool all scored files
    dbkt = {0: [], 0.5: [], 1: []}
    for fn, mdim, mcol, dcol in SCORED:
        for r in _csv(fn):
            ds = _ps(r.get(dcol))
            if ds is None or not r.get("title"):
                continue
            dbkt[ds].append(_art(r, mdim, _ps(r.get(mcol)), ds))
    n = 0
    for hi, lo in [(1, 0.5), (1, 0), (0.5, 0)]:
        for w in dbkt[hi]:
            for la in random.sample(dbkt[lo], min(3, len(dbkt[lo]))):
                pairs.append({"winner": w, "loser": la, "dimension": "decision"})
                n += 1
    counts["decision"] = n
    print(f"Total preference pairs: {len(pairs)}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    return pairs

# ---- JSONL formatting ----
def _sys_msg(hyps):
    hb = "\n".join(f"  H{i+1}: {h}" for i, h in enumerate(hyps))
    return ("You are an expert classifier for experiment-aversion research.\n"
            "Score the Guardian article on 7 dimensions using {0, 0.5, 1} "
            "or -1 if unscored.\n\n"
            f"## K* Hypotheses\n{hb}\n\n## Dimensions\n"
            + ", ".join(DIMS) +
            "\n\nReturn ONLY JSON with these 7 keys. "
            "Use -1 for dimensions without evidence.")

def build_training_jsonl(pairs) -> tuple:
    """Build 80/20 train/val JSONL. Both winners and losers become examples."""
    hyps = _kstar()
    seen = set()
    examples = []
    for pair in pairs:
        for art in [pair["winner"], pair["loser"]]:
            if art["title"] in seen:
                continue
            seen.add(art["title"])
            examples.append({"messages": [
                {"role": "system", "content": _sys_msg(hyps)},
                {"role": "user", "content":
                 f"Title: {art['title']}\nExcerpt: {art['body'][:MAX_BODY]}"},
                {"role": "assistant", "content": json.dumps(art["scores"])}]})
    random.shuffle(examples)
    split = int(len(examples) * 0.8)
    KB_DIR.mkdir(parents=True, exist_ok=True)
    for path, data in [(TRAIN_PATH, examples[:split]),
                       (VAL_PATH, examples[split:])]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
    t, v = split, len(examples) - split
    print(f"Training examples: {t} -> {TRAIN_PATH.name}")
    print(f"Validation examples: {v} -> {VAL_PATH.name}")
    return t, v

# ---- fine-tuning ----
def submit_finetune() -> Optional[str]:
    """Upload JSONL and submit fine-tuning job. Returns job ID or None."""
    try:
        import openai
    except ImportError:
        print("ERROR: openai package not installed")
        return None
    client = openai.OpenAI()
    try:
        tf = client.files.create(file=open(TRAIN_PATH, "rb"), purpose="fine-tune")
        vf = client.files.create(file=open(VAL_PATH, "rb"), purpose="fine-tune")
        job = client.fine_tuning.jobs.create(
            training_file=tf.id, validation_file=vf.id,
            model=BASE_MODEL, hyperparameters={"n_epochs": EPOCHS})
        print(f"Fine-tuning job submitted: {job.id}")
        return job.id
    except Exception as e:
        print(f"ERROR submitting fine-tune: {e}")
        s = _state()
        s["model2_available"] = False
        _save(s)
        return None

def poll_finetune(job_id) -> Optional[str]:
    """Poll until fine-tuning succeeds or fails. Returns model name or None."""
    try:
        import openai
    except ImportError:
        return None
    client = openai.OpenAI()
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Fine-tune status: {job.status}")
        if job.status == "succeeded":
            mn = job.fine_tuned_model
            s = _state()
            s["latest_finetuned_model_name"] = mn
            s["model2_available"] = True
            _save(s)
            print(f"Fine-tuned model ready: {mn}")
            return mn
        if job.status in ("failed", "cancelled"):
            print(f"Fine-tuning {job.status}: {getattr(job, 'error', '')}")
            s = _state()
            s["model2_available"] = False
            _save(s)
            return None
        time.sleep(POLL_SEC)

def submit_and_wait() -> Optional[str]:
    """Submit fine-tuning job and block until complete. Returns model name."""
    jid = submit_finetune()
    return poll_finetune(jid) if jid else None

# ---- scoring ----
def score_article(title, body_text) -> dict:
    """Score article via fine-tuned model. Returns 7-vector with probabilities.
    Each dim: {"score": float, "p0": float, "p05": float, "p1": float}."""
    st = _state()
    mn = st.get("latest_finetuned_model_name")
    if not mn or not st.get("model2_available"):
        raise RuntimeError("Model 2 not available. Run submit_and_wait() first.")
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai package not installed")
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=mn, max_tokens=128, temperature=0,
        messages=[{"role": "system", "content": _sys_msg(_kstar())},
                  {"role": "user", "content":
                   f"Title: {title}\nExcerpt: {body_text[:MAX_BODY]}"}])
    txt = resp.choices[0].message.content.strip()
    if "```" in txt:
        txt = txt[txt.find("{"):txt.rfind("}") + 1]
    raw = json.loads(txt)
    out = {}
    for dim in DIMS:
        v = float(raw.get(dim, -1))
        if v not in (0, 0.5, 1):
            v = 0
        out[dim] = {"score": v, **_PROB[v]}
    return out

def is_available() -> bool:
    """Check whether the fine-tuned model is ready for scoring."""
    try:
        s = _state()
        return bool(s.get("model2_available") and s.get("latest_finetuned_model_name"))
    except Exception:
        return False

# ---- CLI ----
if __name__ == "__main__":
    random.seed(42)
    print("=" * 60)
    print("Model 2: Building training data from scored + gold CSVs")
    print("=" * 60)
    pairs = build_preference_pairs()
    print()
    t, v = build_training_jsonl(pairs)
    print(f"\nTotal unique training examples: {t + v}")
    print(f"Files saved to {KB_DIR}/")
    print("\nTo fine-tune, call: submit_and_wait()")
