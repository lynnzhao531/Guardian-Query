"""Experiment Aversion Finder dashboard — 4 pages per DASHBOARD_SPEC.md.

Pages:
  1. About This Project   (includes folded Model Health)
  2. The Articles         (Tier A + Tier B)
  3. Query History
  4. Reproduce Files

Run with: bash run_dashboard.sh
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
POOLS = OUTPUTS / "pools"
ROUNDS = OUTPUTS / "rounds"
ROUNDS_ARCHIVE = OUTPUTS / "rounds_v1_archive"
PROJECT_STATE = ROOT / "project_state"
KNOWLEDGE = ROOT / "knowledge_base"
SRC = ROOT / "src"

METHODS = ["rct", "prepost", "case_study", "expert_qual", "expert_secondary", "gut"]
MODELS = ["m1", "m2old", "m2new", "m3", "m4", "m5", "m6"]

st.set_page_config(
    page_title="Experiment Aversion Finder",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Version labelling ──────────────────────────────────────────────────────

def _version_for(round_id: int) -> str:
    if round_id <= 10:
        return "v1"
    if round_id <= 23:
        return "v2"
    if round_id <= 33:
        return "v2-crashed"        # no try/finally, many missing manifests
    if round_id <= 43:
        return "v3-pre-fix"        # rotation bug → all RCT, SEEN_URLS saturated
    return "v3"                     # v3 production (rounds 44+)


# ── Data loaders (cached) ────────────────────────────────────────────────────

AUDITS = ROOT / "audits"


@st.cache_data(ttl=30)
def load_manifests() -> pd.DataFrame:
    """Primary source of truth: every round_manifest.json on disk.

    Manifests are written atomically at the end of every round (even crashes)
    and backfilled for historic rounds. The dashboard should prefer this
    over query_log.csv / audit JSON / file-count guessing.
    """
    rows = []
    for mpath in sorted(ROUNDS.glob("round_*/round_manifest.json"),
                        key=lambda p: int(p.parent.name.split("_")[-1])):
        try:
            with open(mpath) as f:
                m = json.load(f)
        except Exception:
            continue
        rows.append(m)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "round_id" in df.columns:
        df["round_id"] = pd.to_numeric(df["round_id"], errors="coerce").astype("Int64")
        df = df.sort_values("round_id").reset_index(drop=True)
        df["version"] = df["round_id"].apply(lambda r: _version_for(int(r)) if pd.notna(r) else "")
    return df


@st.cache_data(ttl=30)
def load_query_log() -> pd.DataFrame:
    """Thin adapter over load_manifests().

    round_manifest.json is now the single source of truth. This function
    exists so callers that used the old schema keep working: it normalises
    column names (unique_scored → unique_scored_count, etc.) and fills in
    the ``status`` / ``failure_stage`` columns the dashboard displays.
    """
    df = load_manifests()
    if df.empty:
        return df

    # Legacy column aliases so existing dashboard code keeps working
    if "unique_scored" in df.columns and "unique_scored_count" not in df.columns:
        df["unique_scored_count"] = df["unique_scored"]
    if "duplicate_scored" in df.columns and "duplicate_scored_count" not in df.columns:
        df["duplicate_scored_count"] = df["duplicate_scored"]

    # Ensure text columns exist even for rounds with no_data_recorded
    for col in ("base_query", "final_query", "target_method", "phase",
                "guardian_order_by", "status", "failure_stage",
                "failure_message", "reconstructed_from"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # For no-data rounds, show an honest placeholder instead of empty string
    mask_nodata = df["status"] == "no_data_recorded"
    df.loc[mask_nodata & (df["final_query"] == ""), "final_query"] = "— round never recorded"
    df.loc[mask_nodata & (df["base_query"] == ""), "base_query"] = "— round never recorded"
    df.loc[mask_nodata & (df["target_method"] == ""), "target_method"] = "—"

    return df


def _read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_all_tier_b() -> pd.DataFrame:
    """Merge Tier B from per-round files AND pool candidates. Dedupe by url."""
    frames = []

    for rdir in sorted(ROUNDS.glob("round_*")) + sorted(ROUNDS_ARCHIVE.glob("round_*")):
        m = re.search(r"round_(\d+)", rdir.name)
        round_id = int(m.group(1)) if m else None
        for tb in rdir.glob("round_*_tier_b_papers.csv"):
            df = _read_csv_safe(tb)
            if not df.empty:
                df = df.copy()
                if round_id is not None and "round_id" not in df.columns:
                    df["round_id"] = round_id
                df["source_file"] = tb.name
                frames.append(df)

    for pc in sorted(POOLS.glob("pool_*_candidates.csv")):
        df = _read_csv_safe(pc)
        if not df.empty:
            df = df.copy()
            df["source_file"] = pc.name
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "url" in merged.columns:
        merged = merged.drop_duplicates(subset=["url"], keep="first")
    return merged.reset_index(drop=True)


@st.cache_data(ttl=30)
def load_all_tier_a() -> pd.DataFrame:
    """Merge Tier A from per-round tier_a files and pool_*_overall files.
    Pool _overall files have known column-shift corruption; skip rows that
    don't look like valid Guardian URLs."""
    frames = []
    for rdir in sorted(ROUNDS.glob("round_*")) + sorted(ROUNDS_ARCHIVE.glob("round_*")):
        m = re.search(r"round_(\d+)", rdir.name)
        round_id = int(m.group(1)) if m else None
        for ta in rdir.glob("round_*_tier_a_papers.csv"):
            df = _read_csv_safe(ta)
            if not df.empty:
                df = df.copy()
                if round_id is not None and "round_id" not in df.columns:
                    df["round_id"] = round_id
                frames.append(df)
    for po in sorted(POOLS.glob("pool_*_overall.csv")):
        df = _read_csv_safe(po)
        if not df.empty:
            df = df.copy()
            # Guard against the known corrupted pool_*_overall schema
            if "url" in df.columns:
                df = df[df["url"].astype(str).str.startswith(("http://", "https://"))]
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "url" in merged.columns:
        merged = merged.drop_duplicates(subset=["url"], keep="first")
    return merged.reset_index(drop=True)


@st.cache_data(ttl=30)
def load_pool_status() -> dict:
    for name in ("POOL_STATUS.json", "pool_status.json"):
        p = PROJECT_STATE / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return {}


@st.cache_data(ttl=30)
def load_kstar() -> dict:
    p = KNOWLEDGE / "K_star.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


@st.cache_data(ttl=30)
def load_scored_rounds() -> pd.DataFrame:
    frames = []
    for rd in sorted(ROUNDS.glob("round_*/scored_results_full.csv")):
        df = _read_csv_safe(rd)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Sidebar navigation ──────────────────────────────────────────────────────

st.sidebar.title("Experiment Aversion Finder")
page = st.sidebar.radio(
    "Pages",
    ["About This Project", "The Articles", "Query History", "Reproduce Files"],
)
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()


# ── Shared: Model Health block (used in expanders on Page 1 and Page 4) ─────

def render_model_health() -> None:
    scored = load_scored_rounds()
    if scored.empty:
        st.info("No scored articles yet.")
        return

    st.markdown("**Per-model stats (across all scored articles)**")
    rows = []
    for m in MODELS:
        dcol = f"{m}_decision_p1"
        if dcol not in scored.columns:
            continue
        p1 = scored[dcol].astype(float)
        mpcols = [c for c in scored.columns
                  if c.startswith(f"{m}_") and c.endswith("_p1") and c != dcol]
        maxmethod = scored[mpcols].astype(float).max(axis=1) if mpcols else pd.Series([0.0] * len(scored))
        high = int(((p1 >= 0.75) & (maxmethod >= 0.75)).sum())
        mid = int((((p1 >= 0.25) & (p1 < 0.75)) | ((maxmethod >= 0.25) & (maxmethod < 0.75))).sum())
        low = len(scored) - high - mid
        total = len(scored)
        rows.append({
            "Model": m.upper(),
            "% HIGH": f"{high/total:.1%}" if total else "0%",
            "% MID": f"{mid/total:.1%}" if total else "0%",
            "% LOW": f"{low/total:.1%}" if total else "0%",
            "n": total,
            "Status": "OK" if high > 0 else "zero HIGH",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("**Pairwise agreement (fraction of articles where both decide the same)**")
    agreements = np.full((len(MODELS), len(MODELS)), np.nan)
    for i, a in enumerate(MODELS):
        for j, b in enumerate(MODELS):
            ca, cb = f"{a}_decision_p1", f"{b}_decision_p1"
            if ca not in scored.columns or cb not in scored.columns or len(scored) == 0:
                continue
            ha = scored[ca].astype(float) >= 0.75
            hb = scored[cb].astype(float) >= 0.75
            agreements[i, j] = ((ha == hb).sum()) / len(scored)
    agree_df = pd.DataFrame(
        agreements,
        index=[m.upper() for m in MODELS],
        columns=[m.upper() for m in MODELS],
    ).round(2)
    # Plain dataframe — no background_gradient (avoids matplotlib dependency)
    st.dataframe(agree_df, use_container_width=True)
    st.caption("Healthy 0.50–0.70 · Redundant >0.90 · Broken <0.30")

    m2new_col = "m2new_decision_p1"
    if m2new_col in scored.columns:
        nonzero = int((scored[m2new_col].astype(float) > 0).sum())
        total = len(scored)
        st.markdown(
            f"**M2-new diagnostic:** non-zero rate = "
            f"{nonzero/total:.1%} ({nonzero} of {total})"
            if total else "**M2-new diagnostic:** no data"
        )


# ── PAGE 1: About This Project ───────────────────────────────────────────────

def page_about():
    st.title("About This Project")

    st.header("About This Project")

    st.subheader("What we're looking for")
    st.markdown(
        "Articles resembling the experiment-aversion framing: an organization "
        "faces a policy choice, and either picks an option or tests which works "
        "better. We score each article on two dimensions:\n\n"
        "- **Method** — some research method appears in a policy context\n"
        "- **Decision** — the article implies an organization might form a policy "
        "or act based on evidence\n\n"
        "We scored generously to avoid missing positives."
    )

    st.divider()

    st.subheader("What we observed")
    st.markdown(
        "Quick recap (details in the email to the team):\n\n"
        "- **The prototype almost never appears** — most articles describe "
        "\"change vs no change,\" not A vs B with clear evidence-to-decision links\n"
        "- **Very low prevalence** — fewer than 10% score high on both Method and "
        "Decision for most non-RCT methods\n"
        "- **Sparse, heterogeneous positives** — no shared keyword patterns, hard "
        "to summarize structurally\n"
        "- **Noisy human ratings** — within-person inconsistency, low cross-rater "
        "agreement\n\n"
        "This means: classifiers default to \"negative,\" relevance is structural "
        "not keyword-based, and human screening isn't reproducible."
    )

    st.divider()

    st.subheader("How the pipeline works")
    st.markdown(
        "Keyword search + manual screening is too slow and unstable. We shifted "
        "to an automatic triage pipeline:"
    )
    st.code("Search Guardian → Score with 6 models → Consensus (Tier A/B) → Collect by method")
    st.markdown(
        "**Models.** Six deliberately different approaches — two LLMs that reason "
        "about article structure, two fine-tuned classifiers from our labeled data, "
        "an embedding model measuring similarity to known positives, and a "
        "hypothesis-driven classifier. If three very different models all see the "
        "same structure, it's probably there.\n\n"
        "**Queries.** Method-specific terms AND decision terms, NOT garbage (court "
        "trials, sports, entertainment). A bandit learns which method to target "
        "and which query width works, balancing exploration against exploitation. "
        "Every few rounds, the system discovers new search terms from articles "
        "it's already found.\n\n"
        "**Consensus.** Tier A = three or more models agree (an \"AND\" approach — "
        "read these first). Tier B = at least one model flags it (an \"OR\" "
        "approach — candidates, noisier). Tier A is for confidence, Tier B is "
        "for coverage."
    )

    st.divider()

    st.subheader("A note on this pipeline")
    st.markdown(
        "- **Who built it.** A rookie with no CS background who doesn't understand "
        "the internals of most of these models. Implementation done through "
        "discussion with ChatGPT and other AI assistants, iterating based on "
        "what the models actually produced.\n"
        "- **Where the insight comes from.** Wang, Sudhir & Zhou (2025), *\"Why it "
        "Works: Can LLM Hypotheses Improve AI Generated Marketing Content?\"* — "
        "a paper I picked up at an AI conference (Amit was there too). Their key "
        "finding: fine-tuning captures superficial patterns, so use LLMs to "
        "generate hypotheses that constrain what models learn. That matched our "
        "concern exactly.\n"
        "- **How it was implemented.** I transferred that insight into model "
        "selection and query design, then iterated based on results. The models, "
        "thresholds, and query strategy all evolved over ~80 rounds of trial "
        "and error.\n"
        "- **Transparent and reproducible.** All code, configs, and decision "
        "history are in the Reproduce Files page. **If some expert in our team "
        "has suggestions for any part — models, queries, consensus rules, "
        "anything — please let me know. My conclusions come from the current "
        "pipeline, and if revising it leads to different conclusions, I'd want "
        "to know.**"
    )

    st.divider()

    # --- Current Progress ---
    st.subheader("Current progress")
    log = load_query_log()
    pool_status = load_pool_status()
    tier_a = load_all_tier_a()
    tier_b = load_all_tier_b()

    c1, c2, c3, c4 = st.columns(4)
    n_completed = int((log["status"] == "completed").sum()) \
        if not log.empty and "status" in log.columns else 0
    c1.metric("Rounds completed", n_completed)
    scored_sum = int(pd.to_numeric(log.get("scored_count"), errors="coerce").fillna(0).sum()) \
        if not log.empty and "scored_count" in log.columns else 0
    c2.metric("Articles scored", scored_sum)
    c3.metric("Tier A found", int(len(tier_a)))
    c4.metric("Tier B found", int(len(tier_b)))

    st.subheader("Per-method progress (target 35 credits)")
    cols = st.columns(3)
    for i, m in enumerate(METHODS):
        credit = float(pool_status.get(f"overall_credit_{m}", 0) or 0)
        with cols[i % 3]:
            st.progress(min(credit / 35.0, 1.0))
            st.caption(f"**{m}**: {credit:.1f}/35")

    max_round = int(log["round_id"].max()) if not log.empty and "round_id" in log.columns else 0
    phase = "Exploration" if max_round <= 20 else ("Mixed" if max_round <= 60 else "Exploitation")
    st.metric("Current phase", phase)

    with st.expander("Technical details in case you're interested (some expert in our team)"):
        st.markdown(
            "### How everything connects\n\n"
            "The pipeline has five components that feed into each other:\n\n"
            "**K\\* → Models → Queries → Bandit → Vocabulary Discovery → back to Queries**\n\n"
            "How they connect:\n\n"
            "- **K\\*** generates structural hypotheses about what makes articles relevant\n"
            "- **3 of 6 models** consult K\\* when scoring; the other 3 deliberately ignore it\n"
            "- **Queries** retrieve articles from The Guardian for models to score\n"
            "- **The bandit** decides which query to run next, based on what previous rounds produced\n"
            "- **Vocabulary discovery** (every 5 rounds) mines scored articles for new search terms → creates new query candidates for the bandit\n\n"
            "The cycle feeds itself: better articles teach us better search terms, which find more articles.\n\n"
            "**How the query terms were built.** We started from the expert-labeled CSVs — articles already known to be relevant, grouped by method type:\n\n"
            "- **Method-unique terms:** phrases frequent in one method's articles but rare in others (e.g., \"randomised controlled trial\" for RCT, \"without evidence\" for gut). Top 10 per method, ranked by specificity.\n"
            "- **Common method terms:** phrases indicating some research method was used, shared across method queries — \"evaluation,\" \"evidence-based,\" \"findings.\"\n"
            "- **Common decision terms:** phrases signaling policy action regardless of method — \"rolled out,\" \"scrapped,\" \"mandated,\" \"approved.\"\n"
            "- **NOT terms:** garbage keywords to exclude (court trials, sports, entertainment).\n\n"
            "Each query = method-unique + common method terms, AND decision terms, NOT garbage. The bandit chooses which method and how many terms (width k)."
        )

        with st.expander("K* — the knowledge base (Wang et al. 2025)"):
            st.markdown(
                "**What.** 15 validated hypotheses about how high-relevance articles are "
                "structured. Stored in `knowledge_base/K_star.json`.\n\n"
                "**How.** Showed an LLM pairs of (relevant, irrelevant) articles and "
                "asked: \"structurally, what makes the first one relevant?\" Tested "
                "candidates against labeled data. Simulated annealing to find the best "
                "set. This is the abduction-induction-optimization loop from Wang, "
                "Sudhir & Zhou (2025).\n\n"
                "**Why.** Fine-tuning on small data learns shortcuts (\"articles with "
                "'trial' = relevant\"). K\\* forces models to learn *structural* patterns "
                "instead. Three of six models consult K\\* — the other three deliberately "
                "ignore it so we can detect K\\*'s blind spots.\n\n"
                "**Parameters.** 15 hypotheses, 76.7% validation accuracy."
            )
            ks = load_kstar()
            if ks:
                for i, h in enumerate(ks.get("hypotheses", []), 1):
                    st.markdown(f"- **H{i}**: {h}")

        with st.expander("The 6 models"):
            st.markdown(
                "| Model | Approach | K\\*? | Why this fits our data |\n"
                "|---|---|---|---|\n"
                "| M1 (Sonnet) | LLM reasons about structure holistically | Yes | Only approach that understands test→evidence→decide flow. Observation 1: relevance is structural. |\n"
                "| M2-old (GPT SFT) | Fine-tuned on labeled data | No | Deliberately K\\*-free — catches what K\\* misses. Its \"no\" is highly reliable (0% FP). |\n"
                "| M3 (Embeddings) | Cosine similarity to method prototypes + MLP | No | Captures holistic \"feel\" without text reasoning. Decorrelated from LLM-based models. |\n"
                "| M4-v3 (K\\* Ridge) | Rates K\\* hypotheses 0-10, Ridge predicts | Yes | Most interpretable. Best method classifier (11/15 correct). |\n"
                "| M5 (DistilBERT) | Local 3-class neural classifier | No | Zero API cost. Highest single-model recall (97%). Outputs relevance only, not method type. |\n"
                "| M6 (Haiku) | Second LLM, different prompt, 3 K\\* hypotheses | Partial | Cheap independent LLM. Different cognitive framing from M1 — asks different questions. |\n\n"
                "**Design choice.** M1/M4/M6 use K\\*. M2/M3/M5 don't. If K\\* has blind "
                "spots, K\\*-free models still catch articles. Most independent pair: "
                "M1 × M2 (19% error overlap). Most redundant: M3 × M4 (55%).\n\n"
                "**Per-model thresholds.** Each model has its own calibrated threshold. "
                "Discrete models: ≥0.25 (excludes score=0.5 ambiguous articles). "
                "Continuous models: varies by distribution (M3: 0.70, M4: 0.50, M5: 0.30). "
                "Method thresholds disabled for M1/M3/M4/M6 — diagnostic showed their "
                "method signal doesn't discriminate."
            )

        with st.expander("Query system + term construction"):
            st.markdown(
                "**Query structure:**\n"
                "```\n"
                "(METHOD-UNIQUE terms, width k ∈ {3,5,7,10})\n"
                "AND (DECISION terms)\n"
                "NOT (STATIC garbage + METHOD-SPECIFIC excludes + DYNAMIC full-method)\n"
                "```\n\n"
                "**NOT clause layers:**\n"
                "- **Static garbage** (always applied): court terms, sports, entertainment\n"
                "- **Method-specific:** e.g., gut queries exclude \"evaluation,\" \"study\"\n"
                "- **Dynamic:** exclude a method's terms when its pool reaches 80%\n\n"
                "**Why 2 AND + 1 NOT.** First 10 rounds used 4-5 AND clauses → 40 "
                "articles/round, 74% duplicates. Simpler queries found 28× more.\n\n"
                "**NOT clause is unconditional.** Early rounds accidentally ran without it "
                "and retrieved court \"trials\" and TV \"pilots.\"\n\n"
                "**Guardian API.** 3 API keys (rotate on rate limits), section filter "
                "excludes sports/entertainment, order-by alternates relevance/newest."
            )

        with st.expander("Bandit (query optimization)"):
            st.markdown(
                "**What.** Contextual Thompson Sampling picks which method and query width. "
                "Bayesian — balances exploration against exploitation.\n\n"
                "**Why a bandit.** ~24 possible query types. Can't try all (budget). Need "
                "an algorithm that learns which queries produce results from limited rounds.\n\n"
                "**Reward (6 components):**\n"
                "```\n"
                "R = V × [0.20·(Tier A) + 0.15·(Tier B) + 0.35·(method coverage)\n"
                "       + 0.20·(unique rate) − 0.15·(duplicates) + 0.05·(goldmine)]\n"
                "V = max(0.05, min(1, unique_scored/50))\n"
                "```\n"
                "Each term exists for a reason: method coverage weights underserved pools; "
                "unique rate rewards fresh articles; duplicate penalty discourages "
                "re-querying exhausted spaces.\n\n"
                "**What we learned.** Rounds 51-74: bandit locked onto case_study for 24 "
                "rounds after one strong result. Fixes applied:\n"
                "- ρ lowered from 0.97 to 0.92 (forgets old successes 3× faster)\n"
                "- Added per-method saturation feature\n"
                "- Exploration floor: each method ≥1 per 10 rounds\n"
                "- Smarter stuck detector: fires on \"3 of 5 scored <10\" not \"3 consecutive zeros\"\n\n"
                "**Parameters.** ρ=0.92, 12 features, M=30 candidates, ε=0.20 then 0.10."
            )

        with st.expander("Vocabulary discovery"):
            st.markdown(
                "**Problem.** After 3-5 productive rounds per method, all articles "
                "reachable by current terms are in SEEN_URLS. Width changes don't help.\n\n"
                "**Solution.** Every 5 rounds: extract novel phrases from Tier A + top "
                "Tier B articles, classify using dual-LLM consensus:\n"
                "- Haiku proposes (broad)\n"
                "- Sonnet validates (precise)\n"
                "- **Both agree (AND) → strong term** — added to main pool\n"
                "- **One agrees (OR) → trial term** — used 20% of the time for 3 rounds; "
                "graduates if it produces Tier A, dropped if not\n\n"
                "**Why dual-LLM.** Same logic as article scoring — two independent models "
                "making different errors. A term both like is almost certainly useful."
            )

        with st.expander("Consensus and tiers"):
            st.markdown(
                "**Tier A (AND).** ≥3 of 6 models vote HIGH and agree on method.\n\n"
                "**Tier B (OR).** ≥1 model votes HIGH. Subdivided by why consensus failed:\n"
                "- *outlier_high* — one model enthusiastic, others cold\n"
                "- *method_disagree* — relevant but which method?\n"
                "- *threshold_miss* — just below cutoff\n"
                "- *decision_split* — method clear, decision unclear\n\n"
                "**Promotion.** Tier B articles that reappear get scores combined. If "
                "they cross ≥3, they promote to Tier A.\n\n"
                "**Dynamic threshold.** Every 10 rounds: if near-misses outnumber Tier A "
                "by 3:1, threshold lowers by 1. Never below 2."
            )

        with st.expander("Active learning (planned)"):
            st.markdown(
                "**Meta-learner.** After 50+ Tier A: train a model on all 6 models' "
                "continuous scores. Learns interaction patterns that hard thresholds miss. "
                "Will re-rank Tier B by meta-score.\n\n"
                "**Current status.** Vocabulary discovery is active. Meta-learner deferred "
                "until 50+ Tier A articles accumulated."
            )

        with st.expander("Training data hierarchy"):
            st.markdown(
                "| Tier | Weight | Source | Why |\n"
                "|---|---|---|---|\n"
                "| GOLD | 1.0 | Expert-verified CSVs | Hand-labelled ground truth |\n"
                "| SILVER | 0.8 | Human-scored CSVs | Lightly reviewed |\n"
                "| BRONZE | 0.5 | Pipeline Tier A | Models agreed, not human-verified |\n"
                "| UNCERTAIN | 0.2 | Tier B ≥2 models | Weak signal, heavily regularized |\n"
                "| LIKELY_LOW | 0.3 | Scored, no tier | Models agreed it was boring |\n\n"
                "Expert data always dominates pipeline data."
            )

        with st.expander("Version history"):
            st.markdown(
                "- **v1 (rounds 1-10).** 4-5 AND clauses, 4 models, no NOT clause. "
                "40 articles/round, 74% duplicates, 0 Tier A.\n"
                "- **v2 (rounds 11-33).** 2 AND + 1 NOT, 5 models, bandit. First Tier A "
                "round 12. NOT clause accidentally conditional. 14 rounds crashed.\n"
                "- **v3 (rounds 34-43).** Models upgraded, thresholds recalibrated. "
                "Rotation bug locked all rounds on RCT.\n"
                "- **v3-fixed (rounds 44-81).** All 6 models, static NOT clause, stuck "
                "detector. 29 Tier A, 411 Tier B. Bandit over-exploited case_study.\n"
                "- **v3b (rounds 82+).** Vocabulary discovery, smarter stuck detector, "
                "bandit tuned."
            )

        with st.expander("Model health"):
            render_model_health()


# ── PAGE 2: The Articles ─────────────────────────────────────────────────────

def _link_col():
    return st.column_config.LinkColumn("url", display_text="open")


def page_articles():
    st.title("The Articles")

    tier_a = load_all_tier_a()
    tier_b = load_all_tier_b()

    # ── Tier A ────────────────────────────────────────────────────────────
    st.header("Tier A — Strong Finds")
    st.markdown(
        "Multiple models independently agreed these describe real "
        "policy evaluations with decisions. **Read these first.**"
    )

    if tier_a.empty:
        st.info("No Tier A articles yet. Pipeline is still exploring — "
                "Tier B below shows candidates that are one vote short.")
    else:
        view = tier_a.copy()
        if "url" not in view.columns and "url_canon" in view.columns:
            view["url"] = view["url_canon"]
        cols = [c for c in [
            "round_id", "title", "url", "classified_method", "confidence",
            "models_agreeing", "method_certainty"
        ] if c in view.columns]
        st.dataframe(
            view[cols],
            use_container_width=True,
            column_config={"url": _link_col()} if "url" in cols else None,
        )
        st.download_button(
            "Download Tier A CSV",
            tier_a.to_csv(index=False).encode("utf-8"),
            "tier_a_articles.csv",
            "text/csv",
        )

    # ── Tier B ────────────────────────────────────────────────────────────
    st.header("Tier B — Candidates")
    st.markdown(
        "At least one model flagged these as potentially relevant. "
        "Some are close to Tier A, others are long shots.\n\n"
        "**Start with Tier A. If you have time, Tier B sorted by relevance score "
        "surfaces the best candidates.**"
    )

    if tier_b.empty:
        st.info("No Tier B candidates yet.")
        return

    # Build display columns
    tb = tier_b.copy()
    if "url" not in tb.columns and "url_canon" in tb.columns:
        tb["url"] = tb["url_canon"]
    if "article_relevance_score" in tb.columns:
        tb["relevance"] = pd.to_numeric(tb["article_relevance_score"], errors="coerce").round(3)
    if "models_agreeing_high" in tb.columns:
        near = pd.to_numeric(tb["models_agreeing_high"], errors="coerce")
        tb["near_tier_a"] = near >= 2
    display_cols = [c for c in [
        "round_id", "title", "url", "classified_method", "method_certainty",
        "relevance", "models_agreeing_high", "disagreement_type", "near_tier_a",
    ] if c in tb.columns]

    # Filters
    c1, c2, c3 = st.columns(3)
    method_filter = c1.multiselect("Method", METHODS, default=[], key="tb_m")
    cert_filter = c2.multiselect(
        "Certainty",
        sorted(tb["method_certainty"].dropna().unique().tolist())
        if "method_certainty" in tb.columns else [],
        default=[], key="tb_c",
    )
    dtype_filter = c3.multiselect(
        "Disagreement type",
        sorted(tb["disagreement_type"].dropna().unique().tolist())
        if "disagreement_type" in tb.columns else [],
        default=[], key="tb_d",
    )

    view = tb
    if method_filter and "classified_method" in view.columns:
        view = view[view["classified_method"].isin(method_filter)]
    if cert_filter and "method_certainty" in view.columns:
        view = view[view["method_certainty"].isin(cert_filter)]
    if dtype_filter and "disagreement_type" in view.columns:
        view = view[view["disagreement_type"].isin(dtype_filter)]

    st.markdown(f"**{len(view)} candidates** (from {len(tier_b)} total)")
    st.dataframe(
        view[display_cols],
        use_container_width=True,
        column_config={"url": _link_col()} if "url" in display_cols else None,
    )
    st.download_button(
        "Download Tier B CSV",
        tier_b.to_csv(index=False).encode("utf-8"),
        "tier_b_articles.csv",
        "text/csv",
    )

    with st.expander("Technical details in case you're interested (some expert in our team)"):
        st.markdown(
            "**Disagreement types in Tier B:**\n"
            "- *outlier_high* — one model sees something others don't\n"
            "- *method_disagree* — relevant but unclear which method\n"
            "- *threshold_miss* — scores moderate, just below cutoff\n"
            "- *decision_split* — unclear if policy decision happened\n"
        )
        st.markdown("### How Articles Enter Each Tier")
        st.markdown(
            "Every article the pipeline fetches is scored by six models on seven "
            "dimensions (one 'is this a policy decision?' dimension plus six method "
            "dimensions — RCT, pre-post, case study, expert qualitative, expert "
            "secondary/quantitative, gut decision). Each score is a probability "
            "triple `{p0, p0.5, p1}`. A model is said to vote HIGH on an article if "
            "**both** its decision-p1 ≥ 0.75 **and** its max-method-p1 ≥ 0.75 — that is, "
            "it thinks the article describes a decision *and* it's confident about "
            "which method was used."
        )
        st.markdown(
            "**Tier A** — an article is promoted to Tier A when **three or more** "
            "models independently vote HIGH *and* the methods they pick agree within "
            "ε = 0.02 (i.e. the top-method probabilities are essentially tied). The "
            "reasoning: three independent models with heterogeneous features (K\\*-"
            "guided LLM, K\\*-free fine-tune, pure embedding similarity) don't agree "
            "by accident. If all three see the same structure, it's probably there."
        )
        st.markdown(
            "**Tier B** — an article lands in Tier B whenever **at least one** "
            "model flags it HIGH but consensus didn't form. Tier B is subdivided by "
            "*why* consensus failed:\n\n"
            "- **`outlier_high`** — one model is enthusiastic and the others are cold. "
            "Often a false positive, but sometimes the enthusiastic model is the only "
            "one trained on the right prototype.\n"
            "- **`method_disagree`** — models agree the article is relevant but fight "
            "over whether it's, say, a pre-post study or a case study. These are the "
            "best candidates for some expert in our team to adjudicate.\n"
            "- **`threshold_miss`** — several models scored just below 0.75. A clean "
            "merge with one more supporting round usually promotes these.\n"
            "- **`decision_split`** — the method is clear but models disagree on "
            "whether any real decision was taken (as opposed to the article merely "
            "discussing research).\n\n"
            "**Promotion from Tier B → Tier A**: at the start of every round, before "
            "we pay for new scoring, we re-merge the current round's model outputs "
            "with any Tier B article that overlaps. If the combined vote now crosses "
            "the ≥3 HIGH + method-agreement threshold, the article is promoted in "
            "place. This is how articles that were 'nearly there' in round 5 can "
            "become Tier A in round 20 without ever being re-fetched."
        )

        st.markdown("### Continuous Relevance Score")
        st.markdown(
            "The numeric 'relevance' column on Tier B isn't a model output — it's a "
            "weighted aggregate we use for ranking only:\n\n"
            "`article_relevance = Σᵢ (wᵢ × decision_p1ᵢ × max(method_p1ᵢ))`\n\n"
            "where the weights wᵢ come from each model's validated reliability on the "
            "stratified test set (M1 currently carries the most weight at 0.28, M4-v3, "
            "M5 and M6 around 0.15 each, etc.)."
        )

        st.markdown("### Method Certainty Thresholds")
        st.markdown(
            "The `method_certainty` column summarises how confident the winning model "
            "is about which of the six methods was used:\n\n"
            "- **HIGH** — top method p ≥ 0.80. Strong signal; usually safe to trust.\n"
            "- **MEDIUM** — 0.50 ≤ top method p < 0.80. The model has a guess but "
            "it's hedging.\n"
            "- **LOW** — top method p < 0.50. The model is basically shrugging; treat "
            "the method label as a hint, not a claim."
        )


# ── PAGE 3: Query History ────────────────────────────────────────────────────

def page_query_history():
    st.title("Query History")

    log = load_query_log()
    if log.empty:
        st.info("No rounds logged yet.")
        return

    st.markdown(
        "**Note: Some rounds score 0 articles — this means all retrieved articles "
        "were already seen in previous rounds, not that the query failed. The "
        "first ~43 rounds were learning rounds (tuning models and queries). "
        "Rounds 44+ are the production system where most finds come from.**"
    )

    # Summary stats — derived from round_manifest.json status
    status_col = log["status"].astype(str)
    n_completed = int((status_col == "completed").sum())
    n_crashed = int((status_col == "crashed").sum())
    n_nodata = int((status_col == "no_data_recorded").sum())

    st.header("Summary stats")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rounds on disk", int(log["round_id"].max()))
    c2.metric("Completed", f"{n_completed}/{len(log)}")
    c3.metric("Crashed", n_crashed)
    c4.metric("No data recorded", n_nodata)

    st.header("Round-by-round table")
    st.caption(
        "Single source of truth: `outputs/rounds/round_*/round_manifest.json`. "
        "Every round writes a manifest in a try/finally, even on crash. "
        "`status=no_data_recorded` means the round never reached the point of "
        "writing any artifact; `status=crashed` means it failed partway through "
        "(see `failure_stage`)."
    )

    show_advanced = st.checkbox("Show advanced columns")

    disp = log.copy()
    disp["query_text"] = disp["final_query"].astype(str)
    disp["query_preview"] = disp["query_text"].apply(
        lambda s: (s[:80] + "…") if len(s) > 80 else s
    )

    # Default visible columns: simple — Round, Method, Query, Scored, Tier A, Tier B
    base_cols = [c for c in [
        "round_id", "target_method", "query_preview",
        "scored_count", "tier_a_count", "tier_b_count",
    ] if c in disp.columns]

    # Advanced (behind toggle): phase, widths, reward, goldmine, diagnostic fields
    advanced_cols = [c for c in [
        "version", "status", "phase", "query_width_k", "decision_width",
        "guardian_order_by", "unique_rate", "reward_R", "goldmine_triggered",
        "total_available", "candidates_retrieved", "duplicate_scored_count",
        "failure_stage", "reconstructed", "reconstructed_from", "failure_message",
    ] if c in disp.columns]

    cols = base_cols + (advanced_cols if show_advanced else [])
    table = disp[cols].copy()
    # For display: leave honest strings alone, show blank for truly-missing numerics
    for c in table.columns:
        if table[c].dtype.kind in "fi":
            table[c] = table[c].apply(lambda v: "—" if pd.isna(v) else (int(v) if float(v).is_integer() else v))
    table = table.rename(columns={"query_preview": "query (first 80 chars)"})
    st.dataframe(table, use_container_width=True)

    with st.expander(f"Rounds that didn't complete ({n_crashed + n_nodata})"):
        bad = log[log["status"].isin(["crashed", "no_data_recorded"])][
            [c for c in ["round_id", "status", "failure_stage", "failure_message",
                         "reconstructed_from"] if c in log.columns]
        ]
        if bad.empty:
            st.success("Every round on disk has status=completed.")
        else:
            st.dataframe(bad, use_container_width=True)
            st.caption(
                "`no_data_recorded` = round dir was empty at backfill time, so we "
                "have no idea what it tried to do. `crashed` = partial artifacts "
                "exist; `failure_stage` tells you where it fell over."
            )

    with st.expander("Show full query text for a round"):
        round_ids = sorted(disp["round_id"].dropna().astype(int).unique().tolist())
        if round_ids:
            picked = st.selectbox("Round", round_ids)
            row = disp[disp["round_id"] == picked].iloc[0]
            qt = str(row.get("query_text", "")) or "N/A"
            st.code(qt)
            if "base_query" in row and str(row["base_query"]) != "nan":
                st.caption(f"base_query: {row['base_query']}")
        else:
            st.info("No rounds available.")

    st.header("Trend charts")
    log_sorted = log.sort_values("round_id")
    numeric_cols = [c for c in ("scored_count", "unique_rate", "tier_a_count",
                                "tier_b_count", "reward_R") if c in log_sorted.columns]
    for c in numeric_cols:
        s = pd.to_numeric(log_sorted[c], errors="coerce")
        if s.notna().any():
            st.subheader(c.replace("_", " ").title())
            st.line_chart(pd.DataFrame({c: s.values}, index=log_sorted["round_id"].values))
    if "target_method" in log_sorted.columns:
        counts = log_sorted["target_method"].dropna().value_counts()
        if not counts.empty:
            st.subheader("Rounds per target method")
            st.bar_chart(counts)


# ── PAGE 4: Reproduce Files ──────────────────────────────────────────────────

def _first_docstring(path: Path) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    m = re.search(r'^\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', txt, re.DOTALL | re.MULTILINE)
    if m:
        first_line = m.group(1).strip().split("\n")[0]
        return first_line[:140]
    for line in txt.splitlines()[:5]:
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("# ").strip()[:140]
    return ""


def _download_file(path: Path, label: str | None = None):
    if not path.exists():
        st.caption(f"_missing: {path.name}_")
        return
    try:
        data = path.read_bytes()
    except Exception as e:
        st.caption(f"_read error {path.name}: {e}_")
        return
    st.download_button(
        label or path.name,
        data,
        file_name=path.name,
        key=f"dl_{path}",
    )


def page_reproduce_files():
    st.header("Reproduce Files")
    st.markdown(
        "Everything needed to reproduce this pipeline is below. Start with "
        "`CLAUDE.md` for the overview, then `REVISED_ARCHITECTURE.md` for the "
        "full specification. The code in `src/` runs the pipeline; the models in "
        "`models/` are the trained classifiers."
    )

    st.subheader("Documentation")
    for name in ["CLAUDE.md", "MASTER_PLAN_v3.md", "REVISED_ARCHITECTURE.md",
                 "DASHBOARD_SPEC.md", "HANDOFF.md", "RUNBOOK.md"]:
        p = ROOT / name
        if p.exists():
            st.markdown(f"**{name}**")
            _download_file(p)
    for rel in ["knowledge_base/K_star.json", "project_state/DPO_GUARDS.md"]:
        p = ROOT / rel
        if p.exists():
            st.markdown(f"**{rel}**")
            _download_file(p)

    versions_dir = ROOT / "docs" / "versions"
    if versions_dir.exists():
        version_files = sorted(versions_dir.glob("*"))
        if version_files:
            st.markdown("**Prior versions (docs/versions/)**")
            cols = st.columns(3)
            for i, p in enumerate(version_files):
                if p.is_file():
                    with cols[i % 3]:
                        _download_file(p)

    st.subheader("Source Code")
    src_files = sorted(SRC.glob("*.py"))
    if src_files:
        rows = []
        for p in src_files:
            rows.append({"file": p.name, "description": _first_docstring(p)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.markdown("**Download individual files:**")
        cols = st.columns(3)
        for i, p in enumerate(src_files):
            with cols[i % 3]:
                _download_file(p)

    st.subheader("Models")
    models_root = ROOT / "models"
    if models_root.exists():
        model_dirs = sorted([p for p in models_root.iterdir() if p.is_dir()])
        if model_dirs:
            st.caption("Trained classifier directories — see each model's README/config.")
            for p in model_dirs:
                st.markdown(f"- `models/{p.name}/`")
        else:
            st.caption("_no model directories found_")
    else:
        st.caption("_models/ directory not present_")

    st.subheader("Configuration & State")
    for rel in ["project_state/STATE.json", "project_state/CONFIG.yaml",
                "project_state/POOL_STATUS.json",
                "project_state/BANDIT_STATE.json",
                "project_state/METHOD_ROTATION.json",
                "project_state/DISCOVERED_TERMS.json",
                "project_state/THRESHOLD_HISTORY.json"]:
        p = ROOT / rel
        if p.exists():
            st.markdown(f"**{rel}**")
            _download_file(p)

    st.subheader("How to Reproduce")
    st.markdown(
        """
1. Read **CLAUDE.md** for project overview
2. Read **REVISED_ARCHITECTURE.md** for complete system specification
3. Install dependencies: `pip install -r requirements.txt`
4. Set API keys in `.env` (Guardian ×3, Anthropic, OpenAI)
5. Run: `python src/run_query_loop.py --max_rounds 30 --resume`
6. Results appear in `outputs/rounds/` and `outputs/pools/`
7. Dashboard: `./run_dashboard.sh` → http://localhost:8501
"""
    )

    with st.expander("Model Health (technical)"):
        render_model_health()


# ── Router ───────────────────────────────────────────────────────────────────

if page == "About This Project":
    page_about()
elif page == "The Articles":
    page_articles()
elif page == "Query History":
    page_query_history()
elif page == "Reproduce Files":
    page_reproduce_files()
