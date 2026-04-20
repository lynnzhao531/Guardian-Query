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
    [
        "Revised Pipeline (After Meeting)",
        "About This Project",
        "The Articles",
        "Query History",
        "Reproduce Files",
    ],
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
                "- **v3b (rounds 82-110).** Vocabulary discovery, smarter stuck detector, "
                "bandit tuned.\n"
                "- **v3c (rounds 111-130).** Pagination with random offset, GOLD CSV "
                "mining, decision-only queries, skip-when-unique<5. 62 Tier A in 20 "
                "rounds but RCT pool filled 2.4× over; other methods still starved.\n"
                "- **v3d (rounds 131+).** Starved-method focus: RCT phrases moved to "
                "GLOBAL_EXCLUDE, bandit RCT override, M5 zero-cost pre-filter, "
                "compound title relevance, near-miss reward, cross-method pool mining, "
                "40% trial rate for starved methods."
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


# ── Page: Revised Pipeline (After Meeting) ──────────────────────────────────

def render_revised_pipeline() -> None:
    import os

    st.title("Revised Pipeline (After Meeting) — Progress Log")

    st.markdown(
        "**What the project does.** We search articles in *The "
        "Guardian* for examples that match a specific prototype: an "
        "organization tests or compares policy options using a "
        "research method, and decides what to do based on the "
        "result. Most articles don't match this prototype. Earlier "
        "work built a pipeline of 6 models to score articles and a "
        "bandit-driven query refinement loop to retrieve more of "
        "them. Progress has been limited; in the team meeting we "
        "reviewed why and redesigned the pipeline.\n\n"
        "This page documents each concern raised in that meeting, "
        "the response we took, and the evidence behind it."
    )

    st.info(
        "**Status:** Stage 0 ✅  |  Stage A ✅  |  "
        "Stage B ⏳ running  |  Stage C ⏸  |  Stage D ⏸"
    )

    st.markdown("**The revised pipeline at a glance:**")

    st.code(
        "  Stage 0            Stage A              Stage B              "
        "Stage C              Stage D\n"
        "  Benchmark     →    Rule exclusion   →   Topic model    →     "
        "Apply models     →   Human\n"
        "  6 models on         3,150 → 1,937        1,937 → (TBD)        "
        "to survivors          review\n"
        "  46 articles         by 9 rules           by LDA               "
        "using Stage 0         top rank\n"
        "                                                                "
        "performance\n",
        language=None,
    )

    st.markdown(
        "Concerns 1 and 2 are about **which articles** we put into "
        "the analysis. Concerns 3, 4, 5 are about **how the 6 "
        "models behave**. The evidence for 3, 4, 5 all comes from "
        "one **benchmark experiment**, described when we reach "
        "Concern 4."
    )

    st.divider()

    def _dl(label, path, key):
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(
                    label, f.read(),
                    file_name=os.path.basename(path),
                    key=key,
                )

    # ── CONCERN 1 ──
    st.header(
        "Concern 1 — Don't trust model classifications until the "
        "models are validated"
    )

    st.markdown(
        "**Problem.** The 6 models haven't been benchmarked against "
        "ground truth. Asking them to find target articles is risky "
        "— model errors propagate into our candidate list.\n\n"
        "A second problem: the pipeline currently runs classifiers "
        "on about **~100 articles per round**. Classifier behavior "
        "at that sample size is unstable. What we conclude from one "
        "round rarely reproduces in the next.\n\n"
        "**Meeting suggestion.** Work from the 6 seed CSVs (3,150 "
        "articles). Apply a **two-stage cut**: first **rule-based** "
        "exclusion, then **LDA-based** exclusion. Only apply "
        "classifiers to what survives both stages.\n\n"
        "**Stage A — rule-based exclusion [done].** 3,150 articles "
        "merged from the 6 seed CSVs; 9 rules based on *The "
        "Guardian*'s URL taxonomy (live blogs, opinion, sports, "
        "entertainment sections, etc.). Every rule is individually "
        "justifiable — the table below shows each rule and its "
        "reason. **1,213 articles removed (38.5% of pool)** with "
        "no interpretive judgment required."
    )

    exclusion_data = [
        ("G1 — Live blogs and \"as it happened\"", 528, 2622,
         "Live blogs describe unfolding events, not completed policy "
         "evaluations"),
        ("G2 — Opinion (/commentisfree/)", 309, 2313,
         "Opinion pieces are argument, not evidence-based evaluation"),
        ("G3 — Sport and football", 92, 2221,
         "Sports articles are not about organizational policy"),
        ("G4 — Culture (books/TV/film/music/stage/art)", 164, 2057,
         "Cultural coverage is review/feature, not policy evaluation"),
        ("G5 — Lifestyle (food/travel/fashion/games)", 63, 1994,
         "Lifestyle content is consumer-facing, not institutional"),
        ("G6 — Obituaries", 16, 1978,
         "Obituaries are biographical, not evaluative"),
        ("G7 — Letters / historical archive", 9, 1969,
         "Reader letters and archival pieces are not about current "
         "evaluation"),
        ("G8 — Masterclasses / shop", 14, 1955,
         "Commercial Guardian content is not editorial policy coverage"),
        ("G9 — News briefings / digests", 18, 1937,
         "Daily briefings summarize; they don't evaluate"),
    ]
    excl_df = pd.DataFrame(
        exclusion_data,
        columns=["Rule", "Removed", "Remaining", "Justification"],
    )
    st.dataframe(excl_df, hide_index=True, use_container_width=True)

    st.markdown(
        "#### Stage B — LDA topic modeling\n\n"
        "**What LDA does.** Latent Dirichlet Allocation groups "
        "articles into themes by finding clusters of words that "
        "tend to co-occur. Each article becomes a mixture of "
        "themes — for example, an article might be 70% \"medical "
        "trials\" and 30% \"UK politics\" if it discusses both. "
        "The method needs one input from us: **K**, the number of "
        "themes to look for.\n\n"
        "**What we asked of it.** Two things. First, identify "
        "topics clearly unrelated to policy testing — candidates "
        "for further exclusion. Second, build a thematic map of "
        "the 1,937 survivors so each article carries a topic label "
        "into downstream scoring. We define *target articles* "
        "(for later use on this page) as articles that resemble "
        "the experiment-aversion prototype: an organization tests "
        "policy options using a research method and decides based "
        "on results.\n\n"
        "**Headline finding.** We tested **62 parameter "
        "configurations** across preprocessing, vocabulary "
        "filtering, and Dirichlet priors. None fixed the low "
        "topic-label stability we observed. Splitting the corpus "
        "by newspaper section (in case heterogeneity was the "
        "cause) only raised stability from 0.22 to 0.31. "
        "**Instability at fine-grained K appears to be a property "
        "of this corpus: small (1,937 articles) and thematically "
        "mixed, so individual topic boundaries shift across random "
        "seeds — but document CLUSTERS (which articles group "
        "together) are more stable than the WORDS labeling each "
        "cluster.** This is a finding, not a failure, and it "
        "shapes how Stage C uses Stage B's output."
    )

    _k_sweep = "outputs/stage_b/v2/K_sweep/K_sweep_plots.png"
    if os.path.exists(_k_sweep):
        st.image(
            _k_sweep, use_column_width=True,
            caption="Coherence (how interpretable topics are) and "
                    "stability (how reproducible they are across "
                    "random seeds) as we vary K. Coherence peaks "
                    "around K=15 while stability decreases "
                    "monotonically. We chose K=15 as the primary "
                    "model — coherence-optimal at a still-usable "
                    "stability level. K=5 is mentioned in the fold "
                    "below as a stable-but-coarse reference.",
        )
    else:
        st.warning(f"K-sweep plot not found at {_k_sweep}")

    st.markdown(
        "**The 15 topics.** Each row below is one theme LDA found. "
        "The \"training overlap\" column tells us what fraction of "
        "that topic's articles appear in our earlier expert-"
        "labeled training data — high values mean our models have "
        "seen that territory before; low values mean the models "
        "will need to generalize. The \"relevance hypothesis\" "
        "column is LDA's auto-flag based on policy-adjacent "
        "vocabulary, for later human review."
    )

    _topics_data = [
        (0, 96, 30, "International affairs and rights",
         "Mixed — review needed"),
        (1, 49, 18, "US politics", "Unlikely target"),
        (2, 84, 30, "Media and press coverage",
         "Mixed — review needed"),
        (3, 266, 36, "Youth services and mental health",
         "Mixed — review needed"),
        (4, 66, 18, "Mixed (water / schools / families)",
         "Mixed — review needed"),
        (5, 66, 23, "War and international security",
         "Mixed — review needed"),
        (6, 278, 17, "UK economic policy", "Likely target"),
        (7, 82, 21, "Mixed (corporate / aid)",
         "Mixed — review needed"),
        (8, 134, 16, "Australian government reviews",
         "Mixed — review needed"),
        (9, 217, 86, "Medical trials and research",
         "Likely target"),
        (10, 70, 24, "Housing and household costs",
         "Mixed — review needed"),
        (11, 164, 31, "Climate and energy policy",
         "Mixed — review needed"),
        (12, 84, 56, "Covid and vaccines", "Likely target"),
        (13, 211, 14, "Criminal justice", "Likely target"),
        (14, 70, 9, "UK benefits and carer's allowance",
         "Mixed — review needed"),
    ]
    _topics_df = pd.DataFrame(
        _topics_data,
        columns=["Topic", "Articles", "Training %",
                 "Theme (our reading)", "Relevance hypothesis"],
    )
    st.dataframe(_topics_df, hide_index=True,
                 use_container_width=True)
    st.caption(
        "Theme labels reflect our reading of the top words and "
        "exemplar articles; LDA's raw top-word lists are in the "
        "downloadable topic cards. Dan: suggest labels in the fold "
        "below if you'd word them differently."
    )

    st.markdown(
        "**Article-cluster stability.** For any given article, we "
        "measured how consistently it gets grouped with the same "
        "companion articles when LDA is re-run with different "
        "random seeds. Median stability is 0.26 across 5 seeds — "
        "modest, but meaningfully higher than the word-label "
        "stability of 0.22. Interpretation: when LDA re-runs, the "
        "TOP-20 WORDS of each topic drift more than the ARTICLES "
        "that compose each cluster. The clusters are real "
        "structures in the corpus; the words we use to label them "
        "are partly an artifact of random initialization. For "
        "downstream Stage C use, the article-to-cluster assignment "
        "carries signal even though individual topic labels are "
        "imperfect.\n\n"
        "**Independent validation via BERTopic.** We ran a "
        "different method (embedding-based clustering) on the "
        "same 1,937 articles. Configured for very-coarse "
        "granularity (2 macro-themes), BERTopic and LDA agreed on "
        "10 of 15 LDA topics (their dominant-assigned articles "
        "fell inside BERTopic's macro-structure). At finer "
        "granularities, the two methods disagreed on boundaries "
        "but identified similar broad structure. Triangulation is "
        "partial — not perfect agreement, but not fundamental "
        "disagreement either.\n\n"
        "**What Stage B gives Stage C.** Every surviving article "
        "now has (a) a dominant topic label, (b) a full "
        "15-dimensional topic distribution, and (c) a flag for "
        "whether its topic has high or low overlap with our "
        "training data. Stage C can use these as context when "
        "applying the 6 models."
    )

    with st.expander(
        "How we set up the LDA — step-by-step assumption check"
    ):
        st.markdown(
            "Standard LDA practice (Blei, Ng & Jordan 2003; "
            "stability framing from Griffiths & Steyvers 2004; "
            "coherence from Röder et al. 2015; critique from "
            "Hoyle et al. 2021) gives a checklist of assumptions. "
            "We tested each on our data rather than assuming.\n\n"
            "| # | Assumption | How we checked | Verdict |\n"
            "|---|---|---|---|\n"
            "| 1 | Documents long enough for topic mixtures | "
            "Length stats: median 827 words, P10=512, P90=1679 | "
            "OK — all well above the minimum viable threshold |\n"
            "| 2 | Bag-of-words is acceptable (word order "
            "ignored) | Tested 5 preprocessing variants including "
            "bigrams (P3) and named-entity preservation (P4) at "
            "K=20 | Acceptable limitation — no variant improved "
            "stability meaningfully. BERTopic (which is not "
            "bag-of-words) triangulates. |\n"
            "| 3 | Documents are exchangeable (no time effects "
            "modeled) | Corpus spans 2000–2025; cannot test "
            "directly | Acknowledged limitation — static model. "
            "Dynamic LDA (Blei & Lafferty 2006) would address "
            "this. |\n"
            "| 4 | Dirichlet priors don't arbitrarily shape "
            "results | 25-cell grid: α ∈ {0.01, 0.1, 1.0, "
            "symmetric, auto} × η ∈ same set, 3 seeds each | No "
            "cell broke stability = 0.35. Learned priors "
            "(auto/auto) selected. |\n"
            "| 5 | K (number of topics) is correctly specified | "
            "9 K values (5, 8, 10, 12, 15, 18, 20, 25, 30), 5 "
            "seeds each, coherence + stability + held-out "
            "perplexity | K=15 won c_v coherence (0.466). K=5 won "
            "stability (0.39). Picked K=15 as primary because "
            "topics there are interpretable; K=5 documented as "
            "stability reference. |\n"
            "| 6 | Vocabulary filter not arbitrary | 12-cell grid: "
            "min_df ∈ {3, 5, 10, 15} × max_df ∈ {0.3, 0.5, 0.7}, "
            "3 seeds each | No cell broke stability = 0.35. Kept "
            "standard defaults (min_df=5, max_df=0.5). |\n"
            "| 7 | Topics are human-interpretable | Per-topic "
            "top-20 words + 5 exemplar documents reviewed; c_v "
            "coherence (Röder 2015) + c_npmi as secondary; "
            "BERTopic triangulation as independent method | "
            "Interpretable (see topic table above); c_v has known "
            "issues per Hoyle 2021, mitigated by multi-method "
            "check. |\n\n"
            "**Decision rule when tests conflicted.** When c_v "
            "coherence and stability disagreed across "
            "preprocessing variants, we preferred stability "
            "(Round 2 framing: \"structural understanding over "
            "filter purity\"). Same rule for vocab and prior "
            "grids. For K, coherence ranking was used as the "
            "primary criterion because peak-coherence topics are "
            "what Dan will actually read.\n\n"
            "**What was skipped deliberately, and why.** We did "
            "not run Dynamic LDA (Blei & Lafferty 2006) — our "
            "corpus spans 25 years but 83% is 2018+ so "
            "within-window mixing likely dominates cross-year "
            "drift. We did not run Hierarchical LDA — the flat "
            "K=15 topics are already at the edge of "
            "interpretability for a 1,937-doc corpus. Both can be "
            "added as follow-ups if Dan wants to commission them."
        )

    with st.expander(
        "Why we think topic-label instability isn't a bug — "
        "the 62-configuration grid"
    ):
        st.markdown(
            "62 parameter configurations in the sensitivity grids "
            "all produced topic-word Jaccard stability in a tight "
            "0.19–0.25 band, far below the 0.7 threshold that "
            "would indicate stable topic boundaries.\n\n"
            "**What stability measures.** We ran LDA with 3 "
            "different random seeds at each configuration. For "
            "each topic in each run, we take the top-20 "
            "highest-probability words. We compare those word "
            "lists across seeds; Jaccard = intersection over "
            "union. Average across all topics gives the stability "
            "number.\n\n"
            "**What 0.22 means in plain terms.** When you re-run "
            "LDA with a different random seed, only about 22% of "
            "each topic's top-20 words reappear in the same "
            "position across runs. In other words, topic labels "
            "are noisy — they're not a stable basis for downstream "
            "filtering of articles by topic name.\n\n"
            "**What we varied:**"
        )
        _vocab = "outputs/stage_b/v2/vocab_sensitivity/grid_heatmap.png"
        if os.path.exists(_vocab):
            st.image(
                _vocab, use_column_width=True,
                caption="Vocabulary filter grid: 12 cells "
                        "(min_df × max_df). Color = stability. "
                        "None breached 0.35; horizontal pattern "
                        "shows stability is roughly invariant to "
                        "vocabulary choice.",
            )
        _prior = "outputs/stage_b/v2/prior_sensitivity/alpha_eta_stability_heatmap.png"
        if os.path.exists(_prior):
            st.image(
                _prior, use_column_width=True,
                caption="Prior grid: 25 cells (α × η). Color = "
                        "stability. Same finding — no combination "
                        "reached stability 0.35.",
            )
        st.markdown(
            "**Why section-stratification didn't help.** If the "
            "corpus were made up of cleanly distinct section-level "
            "sub-corpora, running LDA within each section should "
            "give more stable topics. We tried it on the 10 "
            "sections with enough articles:\n\n"
            "| Section | n | best K | stability |\n"
            "|---|---:|---:|---:|\n"
            "| World News | 242 | 3 | **0.53** |\n"
            "| Business | 115 | 5 | 0.35 |\n"
            "| Science | 79 | 3 | 0.36 |\n"
            "| Society | 292 | 8 | 0.28 |\n"
            "| Australia news | 361 | 8 | 0.28 |\n"
            "| US news | 115 | 5 | 0.27 |\n"
            "| Environment | 108 | 5 | 0.25 |\n"
            "| Politics | 111 | 8 | 0.24 |\n"
            "| UK news | 122 | 8 | 0.21 |\n"
            "| Education | 82 | 8 | 0.27 |\n\n"
            "World News is the one clear case where within-section "
            "structure is more stable, and its setup tells us why: "
            "242 articles with K=3 gives ~80 articles per topic — "
            "plenty of data per theme and thematically contained "
            "(international news). Elsewhere, per-topic sample "
            "sizes are thin (~15–30 articles per topic at K=5-8) "
            "and themes bleed across section boundaries, so "
            "stability improvements are modest.\n\n"
            "**Conclusion.** Stability is bounded by per-topic "
            "sample size in our corpus. At K=15 with 1,937 "
            "articles, each topic gets ~130 articles on average — "
            "thematically mixed enough that word-level labels "
            "don't cleanly reproduce across seeds. This is a "
            "property of our data, not a property of our choices."
        )

    with st.expander(
        "Topic inventory — full detail and download"
    ):
        st.markdown(
            "The topic table above uses short descriptive labels. "
            "Below is a training-overlap bar chart for a fast "
            "visual sense of which topics the models have seen "
            "(high bars) vs where they'll need to generalize "
            "(low bars)."
        )
        _overlap = "outputs/stage_b/v2/plots/plot_topic_overlap.png"
        if os.path.exists(_overlap):
            st.image(
                _overlap, use_column_width=True,
                caption="Per-topic training-data overlap. T9 "
                        "(medical trials) is dominated by "
                        "training-data articles — our models have "
                        "high prior knowledge here. T14 (carer's "
                        "allowance), T13 (criminal justice), and "
                        "T8 (Australian reviews) are mostly new "
                        "territory.",
            )
        else:
            st.warning(
                f"Topic-overlap plot not found at {_overlap}"
            )
        st.markdown(
            "**Individual topic cards.** Each topic has a detailed "
            "card with top 30 words, exemplar articles, peripheral "
            "articles, BERTopic correspondence, date range, and "
            "section concentration. Download the full inventory or "
            "individual cards below."
        )
        _index_path = "outputs/stage_b/v2/topic_profiles/INDEX.md"
        if os.path.exists(_index_path):
            with open(_index_path, "rb") as f:
                st.download_button(
                    "INDEX.md (all 15 topics)", f.read(),
                    file_name="INDEX.md", key="stageb_index",
                )
        for _k in range(15):
            _path = f"outputs/stage_b/v2/topic_profiles/topic_{_k}.md"
            if os.path.exists(_path):
                with open(_path, "rb") as f:
                    st.download_button(
                        f"topic_{_k}.md", f.read(),
                        file_name=f"topic_{_k}.md",
                        key=f"stageb_topic_{_k}",
                    )

    with st.expander(
        "BERTopic triangulation — full detail"
    ):
        st.markdown(
            "BERTopic uses sentence-transformer embeddings + "
            "HDBSCAN clustering. It's a completely different "
            "method from LDA (embedding-based, not bag-of-words; "
            "density-based, not probabilistic). Agreement between "
            "the two methods is evidence that the structure LDA "
            "finds is real; disagreement tells us where the "
            "methods have different inductive biases.\n\n"
            "We ran BERTopic at three granularities by varying "
            "min_cluster_size:\n\n"
            "| BERTopic config | min_cluster_size | BT topics "
            "found | LDA topics with doc-overlap ≥ 50% |\n"
            "|---|---:|---:|---:|\n"
            "| Default | 10 | 38 | 1 of 15 |\n"
            "| Coarse | 40 | 7 | 5 of 15 |\n"
            "| Very coarse | 80 | 2 | 10 of 15 |\n\n"
            "**Reading the table.** At default settings, BERTopic "
            "finds many fine-grained clusters (38) and LDA and "
            "BERTopic disagree on most boundaries (only 1 of LDA's "
            "15 topics has dominant-assigned articles that cluster "
            "together in BERTopic). At coarser settings, BERTopic "
            "collapses toward macro-structure (7, then 2 "
            "clusters), and LDA topics start fitting inside "
            "BERTopic's coarser groups (5, then 10 of 15 LDA "
            "topics \"live inside\" a BERTopic macro-cluster).\n\n"
            "**Interpretation.** Both methods see structure in "
            "this corpus at a macro level — there are ~2 big "
            "thematic groups that both agree on. Below that, the "
            "methods carve finer boundaries differently. Neither "
            "is wrong; they're capturing the same corpus at "
            "different resolutions. For our purpose (identifying "
            "policy-test articles), both the LDA 15 and the "
            "BERTopic macro-2 views are useful, but at different "
            "levels of abstraction."
        )

    with st.expander(
        "Article-cluster stability — distribution and training effects"
    ):
        _stability = "outputs/stage_b/v2/plots/plot_article_stability.png"
        if os.path.exists(_stability):
            st.image(
                _stability, use_column_width=True,
                caption="Distribution of per-article companion "
                        "stability across 5 LDA runs at K=15. "
                        "Articles in the expert-labeled training "
                        "data have higher median stability (0.32) "
                        "than new articles (0.24) — LDA partially "
                        "re-derives the structure the training "
                        "labels already encoded.",
            )
        else:
            st.warning(f"Stability plot not found at {_stability}")
        st.markdown(
            "**What this tells us.**\n\n"
            "- Median article-cluster stability is 0.26 — modest "
            "but above topic-label stability (0.22).\n"
            "- The top 25% of articles have stability ≥ 0.38, "
            "meaning their cluster membership is reasonably "
            "reproducible.\n"
            "- The bottom 25% have stability ≤ 0.16 — these are "
            "articles with genuinely ambiguous topic membership "
            "(high entropy in their topic distribution).\n"
            "- Training articles (articles previously labeled by "
            "experts) are more stable (median 0.32) than new "
            "articles (0.24). LDA has something to \"pull them "
            "toward\" — their content resembles training "
            "exemplars so they consistently cluster near them.\n\n"
            "**For downstream use.** When Stage C weights model "
            "scores by topic context, high-stability articles' "
            "topic assignments are more trustworthy. Low-stability "
            "articles may need human review regardless of model "
            "scores."
        )

    with st.expander(
        "Methodology document and reproducibility"
    ):
        st.markdown(
            "**Tools and versions.**\n"
            "- Python 3.9\n"
            "- gensim 4.3.3 (LDA implementation; Blei 2003 "
            "variational inference)\n"
            "- spaCy 3.x (lemmatization)\n"
            "- sentence-transformers with all-MiniLM-L6-v2 "
            "(BERTopic embeddings)\n"
            "- Random seeds: [42, 13, 99, 7, 123]\n\n"
            "**Key references.**\n"
            "- Blei, Ng, Jordan (2003) — \"Latent Dirichlet "
            "Allocation\" (JMLR). The original LDA paper.\n"
            "- Griffiths & Steyvers (2004) — \"Finding scientific "
            "topics\" (PNAS). Introduced stability via repeated "
            "sampling.\n"
            "- Röder, Both, Hinneburg (2015) — c_v coherence "
            "validated as the best automated metric correlating "
            "with human judgment.\n"
            "- Hoyle et al. (2021) — \"Is Automated Topic Model "
            "Evaluation Broken?\" Showed c_v has limitations; "
            "we mitigate with multiple metrics + BERTopic "
            "triangulation + human inspection of exemplars.\n\n"
            "**Full methodology document and reproducibility:**"
        )
        _methodology = "outputs/stage_b/v2/METHODOLOGY_v2.md"
        if os.path.exists(_methodology):
            with open(_methodology, "rb") as f:
                st.download_button(
                    "METHODOLOGY_v2.md — full decision log",
                    f.read(), file_name="METHODOLOGY_v2.md",
                    key="stageb_methodology",
                )
        st.markdown(
            "**Reproduction.** The full script is in scripts/ "
            "(see repo). It takes approximately 3–4 hours "
            "end-to-end on modern hardware. All intermediate "
            "models are saved for auditing.\n\n"
            "**What to check if you want to push further.** Two "
            "analyses were deliberately skipped: Dynamic LDA "
            "(temporal evolution of topics) and Hierarchical LDA "
            "(tree-structured topic splits). Both are runnable as "
            "follow-ups if that level of detail is needed."
        )

    st.markdown(
        "**📋 What to look at to audit Stage B (in order):**\n\n"
        "1. The 15-topic table above — do our topic labels match "
        "what you'd conclude from the top-words?\n"
        "2. Download INDEX.md and skim it — 30 seconds per topic.\n"
        "3. Open a few individual topic cards (topic_6.md, "
        "topic_9.md, topic_13.md, topic_14.md cover the "
        "interesting range).\n"
        "4. Open the \"How we set up the LDA\" fold — confirm the "
        "assumptions table covers what you'd want checked.\n"
        "5. Open METHODOLOGY_v2.md for the full decision log if "
        "auditing the whole process."
    )

    with st.expander("Technical details and reproducible files"):
        st.markdown(
            "**Per-method contamination.** Each method's seed query "
            "pulled in different proportions of noise. Examples:\n"
            "- prepost query: 43.5% live blogs (articles using \"before "
            "and after\" conversationally)\n"
            "- expertquantitative: 55% noise (corpus too small and "
            "generic)\n"
            "- expertqualitative: 22% live blogs (parliamentary "
            "reporters mention \"expert panel\" constantly)\n\n"
            "**Training-data overlap.** 608 of the 1,937 survivors "
            "(31.4%) appear in expert-labeled training data. The "
            "remaining 1,329 are articles where models need to "
            "generalize.\n\n"
            "**Files:**"
        )
        _dl("merged_all_articles.csv — all 3,150 articles",
            "outputs/stage_a/merged_all_articles.csv",
            "revpipe_c1_merged")
        _dl("survivors.csv — 1,937 after exclusion",
            "outputs/stage_a/survivors.csv", "revpipe_c1_survivors")
        _dl("stage_a_audit.md — full audit report",
            "outputs/stage_a/stage_a_audit.md", "revpipe_c1_audit")

        st.markdown("**Per-group exclusion files:**")
        group_names = {
            1: "liveblog", 2: "opinion", 3: "sport", 4: "culture",
            5: "lifestyle", 6: "obituaries", 7: "letters_archive",
            8: "shop", 9: "briefings",
        }
        for i, name in group_names.items():
            _dl(
                f"excluded_G{i}_{name}.csv",
                f"outputs/stage_a/excluded_G{i}_{name}.csv",
                f"revpipe_c1_g{i}",
            )

    st.divider()

    # ── CONCERN 2 ──
    st.header(
        "Concern 2 — The bandit's query refinement happens on an "
        "open system"
    )

    st.markdown(
        "**Problem.** The bandit updates queries based on results, "
        "and the article pool grows round-over-round. Bandit "
        "algorithms assume a **closed action space**; ours isn't. "
        "Updates on a moving target introduce noise rather than "
        "improvement.\n\n"
        "**Meeting suggestion.** Pause the bandit. Work from one "
        "round's results.\n\n"
        "**What we did.** Paused. No new rounds since the meeting. "
        "All analysis on this page uses the frozen 3,150-article "
        "seed set.\n\n"
        "The benchmark below (Concerns 3, 4, 5) uses **46 "
        "hand-crafted synthetic articles** — not drawn from the "
        "seed set. They test model behavior under known ground "
        "truth, independent of the frozen corpus."
    )

    st.divider()

    # ── THE BENCHMARK (Concern 4) ──
    st.header("The benchmark experiment")

    st.markdown(
        "The next three concerns (3, 4, 5) are about the 6 models' "
        "behavior. To say anything rigorous about model behavior, "
        "we need articles where the correct answer is known. That "
        "construction — and its rationale — comes first. Concerns "
        "3 and 5 then read different views of the benchmark's "
        "results."
    )

    st.subheader(
        "Concern 4 — Models treated as equally reliable without "
        "evidence"
    )

    st.markdown(
        "**Problem.** The consensus rule treats every model's vote "
        "as equally informative. We never measured which model is "
        "most accurate, so **equal weights are unjustified**.\n\n"
        "**Meeting suggestion.** Benchmark the models against "
        "**ground truth**. Dan proposed **predicting article "
        "section** — *The Guardian*'s API provides the section "
        "label for free.\n\n"
        "**Compatibility check.**"
    )

    compat_data = [
        ("Suggested task",
         "Predict article section from title + body"),
        ("Model output format",
         "7-dimension relevance score: one decision score + six "
         "method-type scores. Not a section label."),
        ("Compatibility",
         "Incompatible. Re-prompting for section would test a "
         "different task from what the pipeline does."),
    ]
    compat_df = pd.DataFrame(
        compat_data, columns=["Step", "Finding"]
    )
    st.dataframe(compat_df, hide_index=True, use_container_width=True)

    st.markdown(
        "**Alternative that preserves the ground-truth logic.** "
        "We hand-crafted **46 synthetic articles** across 5 "
        "categories. We define the correct answer for each. Every "
        "model scores every article. Same \"measure models against "
        "known truth\" logic Dan proposed — compatible with our "
        "models' 7-dimensional output format (one decision score + "
        "6 method-type scores per article).\n\n"
        "**Category design — the framework for interpreting "
        "everything below.** Each article targets a specific "
        "combination of two independent signals our models score:\n"
        "- **Decision signal** — does the article describe an "
        "organization making a policy decision?\n"
        "- **Method signal** — does the article use research-method "
        "vocabulary (RCT, case study, pilot)?"
    )

    cat_data = [
        ("**Target** (A)",       "high", "high",
         "NHS tests two screening methods; decides based on results"),
        ("**Topical** (B)",      "high", "low",
         "Minister announces policy with no evaluation"),
        ("**Trap** (C)",         "low",  "high",
         "Court criminal trial described as \"controlled trial\""),
        ("**Unrelated** (D)",    "low",  "low",
         "Football match report"),
        ("**Borderline** (E)",   "mid",  "mid",
         "Pilot unclear if test or rollout"),
    ]
    cat_df = pd.DataFrame(
        cat_data,
        columns=["Category", "Decision", "Method", "Example"],
    )
    st.dataframe(cat_df, hide_index=True, use_container_width=True)
    st.caption(
        "Descriptive names used throughout this page. Data files "
        "use the letter codes A–E."
    )

    st.markdown(
        "**Key finding — per-method calibration per model.** For "
        "each research method, does each model correctly score high "
        "on articles labeled with that method, and low on articles "
        "that aren't?\n\n"
        "The plot below breaks the 6 method dimensions into 6 "
        "panels. In each panel, the green dots are articles labeled "
        "with that method (they should score high). The grey "
        "distribution is articles NOT labeled with that method "
        "(they should score low)."
    )

    _scatter = "outputs/benchmark/v1/plots/plot_scatter_decision_method.png"
    if os.path.exists(_scatter):
        st.image(
            _scatter, use_column_width=True,
            caption="Per-method calibration. Each panel is one "
                    "method. Green dots = articles labeled with "
                    "that method (expected HIGH). Grey distribution "
                    "= articles NOT labeled with that method "
                    "(expected LOW). A well-calibrated model has "
                    "its green dots clearly above its grey "
                    "distribution.",
        )
    else:
        st.warning(
            "Plot not found at " + _scatter + ". "
            "Run scripts/build_revised_pipeline_plots.py."
        )

    st.markdown(
        "Reading the plot per panel: a green dot high on the "
        "y-axis with the grey distribution low below it means the "
        "model distinguishes \"has this method\" from \"doesn't "
        "have this method\" cleanly. A green dot low, or a grey "
        "distribution that's high, means the model is confused. "
        "Look across the 6 panels to see which methods each model "
        "is best at.\n\n"
        "Across methods, M6 Haiku and M5 tend to show the cleanest "
        "separation. M3 and M4-v3 show green dots that are not "
        "reliably higher than grey distributions on several "
        "methods — a calibration weakness. The `gut` panel has no "
        "green dots because the benchmark has no articles labeled "
        "with that method."
    )

    with st.expander("Technical details and reproducible files"):
        st.markdown(
            "**AUC on the decision dimension — works for all 7 "
            "models.** Since AUC depends only on the ordering of a "
            "single score, we can compute it on decision_p1 for "
            "every model (including the three discrete ones). This "
            "is the cleanest per-model classification metric.\n\n"
            "| Model | AUC (Target vs Unrelated) | AUC (Target vs Not-target) |\n"
            "|---|---|---|\n"
            "| M1 Sonnet | 1.00 | 0.90 |\n"
            "| M2-old    | 1.00 | 0.88 |\n"
            "| M2-new    | 1.00 | 0.90 |\n"
            "| M3        | 1.00 | 0.86 |\n"
            "| M4-v3     | 0.96 | 0.75 |\n"
            "| M5        | 1.00 | 0.91 |\n"
            "| M6 Haiku  | 1.00 | 1.00 |\n\n"
            "The authoritative numbers are in per_model_metrics.csv "
            "(downloadable below) since they're recomputed from "
            "current scores.\n\n"
            "**Per-trap analysis (Trap category).** Which specific "
            "trap articles fooled which models. See "
            "false_positive_analysis.csv for article-level detail. "
            "Summary pattern: M3 and M4-v3 are most susceptible to "
            "keyword traps (4–6 of 10 Traps scored > 0.5 on some "
            "method); M1 and M6 reject nearly all.\n\n"
            "**Files:**"
        )
        _dl("per_model_metrics.csv — AUC, precision, category accuracy",
            "outputs/benchmark/v1/per_model_metrics.csv",
            "revpipe_c4_metrics")
        _dl("category_breakdown.csv — mean/median scores per category",
            "outputs/benchmark/v1/category_breakdown.csv",
            "revpipe_c4_catbreak")
        _dl("false_positive_analysis.csv — per-trap analysis",
            "outputs/benchmark/v1/false_positive_analysis.csv",
            "revpipe_c4_fp")
        _dl("raw_scores.csv — every model × every article × 7 dimensions",
            "outputs/benchmark/v1/raw_scores.csv", "revpipe_c4_raw")
        _dl("synthetic_benchmark_articles.csv — the 46 test articles",
            "synthetic_benchmark_articles.csv", "revpipe_c4_articles")
        _dl("synthetic_benchmark_answers.csv — ground truth labels",
            "synthetic_benchmark_answers.csv", "revpipe_c4_answers")

    st.divider()

    # ── CONCERN 3 ──
    st.header(
        "Concern 3 — Discrete thresholds throw away information"
    )

    st.markdown(
        "**Problem.** The pipeline converts each model's continuous "
        "score into HIGH/LOW using a **per-model calibrated "
        "threshold**. An article at 0.91 is treated the same as "
        "0.51; 0.49 and 0.51 become opposites. Thresholds were "
        "tuned empirically, not derived from principle.\n\n"
        "**Meeting suggestion.** Use **continuous probabilities** "
        "throughout. Plot **density and CDF** of each model's "
        "decision scores on benchmark articles so the distribution "
        "shape is visible.\n\n"
        "**Reading the benchmark for decision-signal separation.** "
        "The plot below shows each model's decision_p1 distribution "
        "on articles that are high-decision by design "
        "(Categories A + B, 20 articles) vs low-decision by design "
        "(Categories C + D, 20 articles). Category E is excluded "
        "(mid-by-design). A model whose green curve (high) sits "
        "clearly to the right of its grey curve (low) separates "
        "the decision signal cleanly."
    )

    _density = "outputs/benchmark/v1/plots/plot_density_decision.png"
    if os.path.exists(_density):
        st.image(
            _density, use_column_width=True,
            caption="Each panel shows one model's decision_p1 "
                    "distribution. Green = articles high on decision "
                    "by design (A+B). Grey = articles low on "
                    "decision by design (C+D). Wider separation = "
                    "stronger decision signal. Discrete models "
                    "(M1, M2_old, M2_new) produce three spikes at "
                    "canonical values regardless of class — no "
                    "threshold recovers finer gradation.",
        )
    else:
        st.warning(
            "Density plot not found. Run "
            "scripts/build_revised_pipeline_plots.py."
        )

    st.markdown(
        "**Key finding.** Three models (M1 Sonnet, M2-old, M2-new) "
        "produce outputs concentrated at three values "
        "{0.05, 0.20, 0.80}. Their internal scoring is "
        "architecturally discrete; no threshold setting can make "
        "them continuous. The four continuous models (M3, M4-v3, "
        "M5, M6) produce smooth distributions and separate the "
        "decision signal to varying degrees — M5 and M6 show "
        "cleaner separation than M3 and M4-v3.\n\n"
        "Discrete output isn't automatically bad — discrete models "
        "still participate in classification (see Concern 4's AUC "
        "table in the fold). What they can't do is rank articles "
        "by fine-grained relevance (see Concern 5)."
    )

    with st.expander("Technical details and reproducible files"):
        st.markdown(
            "**Per-method decision separation (continuous models "
            "only).** The top-level plot shows the decision "
            "dimension. For the method dimensions, here is the "
            "parallel view: for each of the 6 methods, does each "
            "continuous model separate articles labeled with that "
            "method from articles not labeled with it?\n\n"
            "Discrete models are excluded here because their outputs "
            "lack the resolution for meaningful density. See the "
            "top-level plot for how the discrete models behave on "
            "the decision dimension."
        )

        _plot_per_method = "outputs/benchmark/v1/plots/plot_per_method.png"
        if os.path.exists(_plot_per_method):
            st.image(
                _plot_per_method, use_column_width=True,
                caption="Rows = methods. Columns = the 4 continuous "
                        "models. Green = articles labeled with this "
                        "method (expected HIGH). Grey = articles "
                        "not labeled with this method (expected "
                        "LOW). For small n (< 3 expected-high "
                        "articles), rug lines replace density "
                        "curves.",
            )
        else:
            st.warning(
                "Per-method plot not found at " + _plot_per_method + "."
            )

        st.markdown(
            "**CDF version of the decision plot.** If you want CDFs "
            "instead of densities, score_distributions.csv contains "
            "per-article per-model decision scores; re-plot as ECDF "
            "by category for the CDF view Dan asked about.\n\n"
            "**Possible extraction of continuous scores from LLM "
            "models.** M1 and M6 are LLM-based. Token-level "
            "probabilities from the logprobs API could in principle "
            "give continuous scores even from currently-discrete "
            "models. Future work; requires modifying the scoring "
            "pipeline to capture logprobs.\n\n"
            "**Files:**"
        )
        _dl("score_distributions.csv — long-format score data",
            "outputs/benchmark/v1/score_distributions.csv",
            "revpipe_c3_scores")

    st.divider()

    # ── CONCERN 5 ──
    st.header(
        "Concern 5 — Different models' labels aren't directly comparable"
    )

    st.markdown(
        "**Problem.** The pipeline applies a **different calibrated "
        "threshold per model** — each model's HIGH/LOW cutoff was "
        "tuned independently. \"M1 says HIGH\" and \"M3 says HIGH\" "
        "mean different things at the score level; thresholds have "
        "made the labels incomparable. A secondary issue is that "
        "some models produce continuous probabilities while others "
        "produce three discrete values (see Concern 3), compounding "
        "the threshold problem.\n\n"
        "**Meeting suggestion.** Each model produces a "
        "**rank-ordering** of the same articles. Rank orders are "
        "internal to each model, independent of where thresholds "
        "sit, and comparable across output formats up to ties.\n\n"
        "**Reading the benchmark for ranking agreement.** We rank "
        "the 46 benchmark articles by decision_p1 for each of the "
        "4 continuous models. Discrete models are excluded: their "
        "3-valued outputs create massive ties that degrade rank "
        "correlation.\n\n"
        "Decision-dimension ranking is the simplest, cleanest "
        "criterion — it uses the same score every model produces, "
        "without any method choice or composite aggregation. The "
        "pairwise Spearman correlation below measures how similarly "
        "the models order the articles."
    )

    _heatmap = "outputs/benchmark/v1/plots/plot_ranking_agreement.png"
    if os.path.exists(_heatmap):
        st.image(
            _heatmap, use_column_width=True,
            caption="Pairwise Spearman rank correlation. 4 "
                    "continuous models. Each cell: how similarly "
                    "two models rank the 46 articles by "
                    "decision_p1.",
        )
    else:
        st.warning(
            "Heatmap not found. Run "
            "scripts/build_revised_pipeline_plots.py."
        )

    st.markdown(
        "**Key finding.** Pairs whose cell is dark blue (high "
        "Spearman) order articles similarly on the decision "
        "dimension. Lighter cells mark pairs that see the decision "
        "signal differently. See the fold for per-method ranking "
        "agreement (6 heatmaps, one per method dimension).\n\n"
        "No hardcoded numbers here — the heatmap shows the current "
        "values."
    )

    with st.expander("Technical details and reproducible files"):
        st.markdown(
            "**Per-method ranking agreement.** The top-level "
            "heatmap uses the decision dimension. Here are 6 more "
            "heatmaps, one per method, showing how the 4 continuous "
            "models rank articles by each method's score "
            "separately. If two models agree strongly on the "
            "decision dimension but disagree on a specific method, "
            "they're capturing the decision signal commonly but "
            "differ on that method's identification."
        )

        _ranking_pm = "outputs/benchmark/v1/plots/plot_per_method_ranking.png"
        if os.path.exists(_ranking_pm):
            st.image(
                _ranking_pm, use_column_width=True,
                caption="Per-method Spearman rank correlation among "
                        "4 continuous models. Dark blue = agreement "
                        "on that method's ranking.",
            )
        else:
            st.warning(
                "Per-method ranking plot not found at "
                + _ranking_pm + "."
            )

        st.markdown(
            "**Why discrete models are excluded.** M1, M2-old, "
            "M2-new produce at most three distinct scores; their "
            "rankings contain many ties, and Spearman degrades. "
            "They participate in classification (Concern 4's AUC "
            "table) but not in ranking.\n\n"
            "**Why decision alone at top level, not a composite.** "
            "Ranking by decision_p1 uses a single score every model "
            "produces. It requires no aggregation choice. Composites "
            "involving method scores are a second-order question — "
            "whether models agree on a specific method — and that "
            "question is answered by the per-method heatmaps "
            "above.\n\n"
            "**Kendall tau and top-K overlap.** Alternative "
            "rank-agreement metrics in ranking_agreement.csv. "
            "Kendall tau behaves similarly to Spearman. Top-K "
            "overlap measures the raw number of articles that two "
            "models share in their top-K by score.\n\n"
            "**Files:**"
        )
        _dl("ranking_agreement.csv — Spearman, Kendall, top-K per pair",
            "outputs/benchmark/v1/ranking_agreement.csv",
            "revpipe_c5_ranking")

    st.divider()

    with st.expander("Full file index"):
        st.markdown(
            "All reproducible outputs available in the repository. "
            "Download individual files via the per-concern folds "
            "above, or browse the repo directly.\n\n"
            "**Stage 0 — Benchmark** (`outputs/benchmark/v1/`)\n"
            "- synthetic_benchmark_articles.csv, "
            "synthetic_benchmark_answers.csv\n"
            "- raw_scores.csv — every model × every article × "
            "7 dimensions\n"
            "- per_model_metrics.csv — AUC and per-category metrics\n"
            "- category_breakdown.csv — per-category mean/median "
            "scores\n"
            "- false_positive_analysis.csv — per-trap analysis\n"
            "- ranking_agreement.csv — Spearman, Kendall, top-K per "
            "model pair\n"
            "- score_distributions.csv — long-format per-article "
            "per-model scores\n\n"
            "**Stage 0 plots** (`outputs/benchmark/v1/plots/`)\n"
            "- plot_density_decision.png — Concern 3 top-level\n"
            "- plot_scatter_decision_method.png — Concern 4 "
            "per-method calibration\n"
            "- plot_ranking_agreement.png — Concern 5 Spearman "
            "heatmap\n"
            "- plot_per_method.png — Concern 3 fold per-method "
            "density\n"
            "- plot_per_method_ranking.png — Concern 5 fold "
            "per-method rankings\n\n"
            "**Stage A — Exclusion** (`outputs/stage_a/`)\n"
            "- merged_all_articles.csv — 3,150 articles with "
            "training flag\n"
            "- survivors.csv — 1,937 after 9 exclusion rules\n"
            "- excluded_G1_liveblog.csv through "
            "excluded_G9_briefings.csv\n"
            "- stage_a_audit.md — full audit\n\n"
            "**Stage B — LDA** (`outputs/stage_b/v2/`)\n"
            "- METHODOLOGY_v2.md — complete methodology and "
            "decision log\n"
            "- K_sweep/K_sweep_plots.png, K_sweep/sweep_results.csv, "
            "K_sweep/K_decision.md\n"
            "- vocab_sensitivity/grid_heatmap.png, "
            "vocab_sensitivity/decision.md\n"
            "- prior_sensitivity/alpha_eta_stability_heatmap.png, "
            "prior_sensitivity/decision.md\n"
            "- section_stratified/summary.md\n"
            "- bertopic/alignment_analysis.md\n"
            "- article_stability/article_stability_summary.md\n"
            "- topic_profiles/INDEX.md, topic_profiles/topic_0.md "
            "through topic_14.md\n"
            "- plots/plot_topic_overlap.png (training overlap per "
            "topic)\n"
            "- plots/plot_article_stability.png (per-article "
            "cluster stability)\n\n"
            "**Stage C — Apply models** (pending)\n\n"
            "**Stage D — Human review** (pending)\n\n"
            "Repo: https://github.com/lynnzhao531/Guardian-Query"
        )


# ── Router ───────────────────────────────────────────────────────────────────

if page == "Revised Pipeline (After Meeting)":
    render_revised_pipeline()
elif page == "About This Project":
    page_about()
elif page == "The Articles":
    page_articles()
elif page == "Query History":
    page_query_history()
elif page == "Reproduce Files":
    page_reproduce_files()
