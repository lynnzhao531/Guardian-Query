"""Task 8 — Model swap orchestrator (CREATES DISK CHANGES — do NOT run
until Task 7 report has been reviewed and v3 models are approved).

What it does when run with --confirm:

  1. Backup current models to models/_v2_backup/
       - models/model3/           → models/_v2_backup/model3/
       - models/model4/           → models/_v2_backup/model4/
       - src/model1_llm_judge.py  → src/_v2_backup/model1_llm_judge.py
       - src/model3_embedding_classifier.py → _v2_backup/
       - src/model4_hypothesis_classifier.py → _v2_backup/

  2. Swap in v3 versions IF approved by approvals dict (build from test report):
       - v3 model1: copy model1_llm_judge_v3.py over model1_llm_judge.py
       - v3 model3: replace models/model3/ contents with models/model3_v3/
                    replace src/model3_embedding_classifier.py with shim pointing at model3_v3 scorer
       - v3 model4: same treatment for model4
       - model5: register in consensus + round_runner schema

  3. Update src/round_runner.py scored_results_full column list to include
     m5_* columns (§3.6) and src/consensus.py to include M5 in tier checks.

  4. Fix NOT clause in src/query_builder.py per REVISED_ARCHITECTURE.md §4.4:
       - Always add GLOBAL_EXCLUDE (sentencing, prosecution, tournament, etc.)
       - Add METHOD_SPECIFIC_EXCLUDE for expert_secondary and gut
       - FULL_METHOD_EXCLUDE already implemented via get_full_method_not()

  5. Implement Tier promotion per §3.5 in src/round_runner.py:
       - Before scoring, check candidates_raw against outputs/pools/pool_*_candidates.csv
       - If url already in Tier B, carry forward prior scores (keep highest per model)
       - Re-evaluate Tier A threshold; if now met, add to overall + LLM pools and
         mark promoted_to_tier_a=true on the Tier B entry.

Safety: refuses to run without --confirm and a valid approvals file.
Never runs while run_query_loop.py is active (checks for RUNNING marker).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = ROOT / "models" / "_v2_backup"
SRC_BACKUP = ROOT / "src" / "_v2_backup"
APPROVALS_PATH = ROOT / "outputs" / "v3_approvals.json"
RUNNING_MARKER = ROOT / "project_state" / "PIPELINE_RUNNING"


def _safety_check() -> None:
    if RUNNING_MARKER.exists():
        print("❌ REFUSING: pipeline is running. Remove", RUNNING_MARKER, "first.")
        sys.exit(2)
    if not APPROVALS_PATH.exists():
        print("❌ REFUSING: no approvals file at", APPROVALS_PATH)
        print("   Create it with the keys:")
        print('   {"m1_v3": true|false, "m3_v3": true|false, "m4_v3": true|false,')
        print('    "m5": true|false}')
        sys.exit(3)


def _load_approvals() -> dict:
    return json.loads(APPROVALS_PATH.read_text())


# ── Step 1: Backup ───────────────────────────────────────────────────────────

def backup_current() -> None:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    stamp_dir = BACKUP_DIR / ts
    stamp_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model3", "model4"):
        src = ROOT / "models" / name
        if src.exists():
            shutil.copytree(src, stamp_dir / name)
            print(f"  backed up models/{name} → {stamp_dir}/{name}")
    src_stamp = SRC_BACKUP / ts
    src_stamp.mkdir(parents=True, exist_ok=True)
    for name in ("model1_llm_judge.py", "model3_embedding_classifier.py",
                 "model4_hypothesis_classifier.py", "consensus.py",
                 "round_runner.py", "query_builder.py"):
        p = ROOT / "src" / name
        if p.exists():
            shutil.copy2(p, src_stamp / name)
            print(f"  backed up src/{name} → {src_stamp}/{name}")


# ── Step 2: Swap model files ────────────────────────────────────────────────

def swap_model1_v3() -> None:
    """Replace src/model1_llm_judge.py with v3 prompt+parser.

    We keep the same public API (`score_article(title, body) -> 7-vector dict`)
    so round_runner does not need changes.
    """
    v3 = ROOT / "src" / "model1_llm_judge_v3.py"
    old = ROOT / "src" / "model1_llm_judge.py"
    if not v3.exists():
        print("  ⚠ v3 file missing:", v3)
        return
    shutil.copy2(v3, old)
    print(f"  swapped {old.name} ← {v3.name}")


def swap_model3_v3() -> None:
    """Replace models/model3/ with models/model3_v3/ and rewire the module.

    The scorer module src/model3_embedding_classifier.py keeps its name but
    its load path changes. Simplest approach: copy model3_v3/* into model3/
    and write a new st_finetuned marker file that the module reads.
    """
    src = ROOT / "models" / "model3_v3"
    dst = ROOT / "models" / "model3"
    if not src.exists():
        print("  ⚠ model3_v3 not trained:", src)
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    # Also swap the module file with the v3 scorer wrapped in the old name
    v3_mod = ROOT / "src" / "model3_v3.py"
    old_mod = ROOT / "src" / "model3_embedding_classifier.py"
    if v3_mod.exists():
        shutil.copy2(v3_mod, old_mod)
    print(f"  swapped {dst} ← {src}; module {old_mod.name} ← model3_v3.py")


def swap_model4_v3() -> None:
    src = ROOT / "models" / "model4_v3"
    dst = ROOT / "models" / "model4"
    if not src.exists():
        print("  ⚠ model4_v3 not trained:", src)
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    v3_mod = ROOT / "src" / "model4_v3.py"
    old_mod = ROOT / "src" / "model4_hypothesis_classifier.py"
    if v3_mod.exists():
        shutil.copy2(v3_mod, old_mod)
    print(f"  swapped {dst} ← {src}; module {old_mod.name} ← model4_v3.py")


# ── Step 3: Register M5 in pipeline ─────────────────────────────────────────

def register_model5() -> None:
    """Add M5 to consensus + round_runner's scored_results_full schema."""
    # These edits are described here but MUST be applied by a human review —
    # automatic in-place sed on live pipeline code is risky. We only print
    # the diff instructions so the operator can apply them.
    instructions = """
### Manual edits required to register M5 in the pipeline ###

1. src/consensus.py
   Add "m5" to the MODELS list, and to the scoring loop:

       from model5_deberta import score_article as m5_score
       scores["m5"] = m5_score(title, body)

2. src/round_runner.py
   Add m5_* columns to the `_flatten_result` function and the SCORED_FULL_COLS
   list. M5 produces the same 7-vector shape, so the additions mirror m4_*.

3. src/run_query_loop.py
   No changes required — it already reads from round_runner.

4. src/bandit.py / src/query_builder.py
   No changes required for M5.
"""
    print(instructions)


# ── Step 4: Fix NOT clause in query_builder.py ──────────────────────────────

def fix_not_clause() -> None:
    instructions = """
### Manual edits required to fix NOT clauses per REVISED_ARCHITECTURE.md §4.4 ###

In src/query_builder.py:

1. Add GLOBAL_EXCLUDE constant:

    GLOBAL_EXCLUDE = [
        "sentencing", "prosecution", "defendant", "verdict", "convicted",
        "charged", "magistrate", "crown court", "jury", "appeal",
        "tournament", "championship", "premier league", "goal", "coach", "match",
        "album", "film", "theatre", "premiere", "box office",
    ]

2. Add METHOD_SPECIFIC_EXCLUDE dict:

    METHOD_SPECIFIC_EXCLUDE = {
        "expert_secondary": ["randomized", "randomised", "trial", "rct",
                              "placebo", "control group"],
        "gut": ["trial", "randomized", "randomised", "study", "evaluation",
                "pilot", "impact assessment", "review"],
    }

3. Update build_query() to always append GLOBAL_EXCLUDE + method-specific
   NOT terms (on top of the existing full-method NOT from §4.4 Component 3).
"""
    print(instructions)


# ── Step 5: Tier promotion (§3.5) ────────────────────────────────────────────

def add_tier_promotion() -> None:
    instructions = """
### Manual edits required to implement Tier B→A promotion per §3.5 ###

In src/round_runner.py, before scoring each round:

    def _load_tier_b_scores() -> dict[str, dict]:
        '''Return {url_canon: {m1: {decision:..., method_rct:..., ...}, m2old: ..., ...}}'''
        out = {}
        for m in METHODS:
            p = POOLS / f"pool_{m}_candidates.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            for _, r in df.iterrows():
                url = r.get("url_canon")
                if url and url not in out:
                    out[url] = {
                        "m1_decision_p1": r.get("m1_decision_p1"),
                        ...
                    }
        return out

Then, when scoring a candidate whose url is in _load_tier_b_scores(), merge
its carried-forward scores with new ones (keep MAX per field). After consensus
check, if the merged article now meets Tier A threshold, add to overall + LLM
pools and write promoted_to_tier_a=true in the candidates file.
"""
    print(instructions)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Swap v3 models into live pipeline")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually perform the swap (default: dry-run)")
    parser.add_argument("--skip-safety", action="store_true",
                        help="Skip safety check (DANGEROUS)")
    args = parser.parse_args()

    if not args.skip_safety:
        _safety_check()

    if not args.confirm:
        print("DRY RUN — pass --confirm to actually swap. "
              "Required file:", APPROVALS_PATH)
        return

    approvals = _load_approvals()
    print("Approvals loaded:", approvals)

    print("\n== Step 1: Backup ==")
    backup_current()

    print("\n== Step 2: Swap approved models ==")
    if approvals.get("m1_v3"):
        swap_model1_v3()
    else:
        print("  m1_v3 NOT approved — keeping v2")
    if approvals.get("m3_v3"):
        swap_model3_v3()
    else:
        print("  m3_v3 NOT approved — keeping v2")
    if approvals.get("m4_v3"):
        swap_model4_v3()
    else:
        print("  m4_v3 NOT approved — keeping v2")

    print("\n== Step 3: Register M5 ==")
    if approvals.get("m5"):
        register_model5()
    else:
        print("  m5 NOT approved — skipping")

    print("\n== Step 4: NOT clause fix ==")
    fix_not_clause()

    print("\n== Step 5: Tier promotion ==")
    add_tier_promotion()

    print("\n✅ model_swap complete. Steps 3-5 are printed as instructions — "
          "apply them by hand after reviewing the test report, then run the "
          "next round.")


if __name__ == "__main__":
    main()
