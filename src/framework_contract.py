"""
Framework Contract Validation per MASTER_PLAN_v3.md §21.
Pre- and post-round validators. FATAL on failure.
"""

import json
import csv
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
STATE_DIR = PROJECT_ROOT / "project_state"
KB_DIR = PROJECT_ROOT / "knowledge_base"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
AUDIT_DIR = PROJECT_ROOT / "audits"

METHODS = ["rct", "prepost", "case_study", "expert_qual", "expert_secondary", "gut"]
DIMS = ["decision", "method_rct", "method_prepost", "method_case_study",
        "method_expert_qual", "method_expert_secondary", "method_gut"]


class ValidationError(Exception):
    pass


def validate_repo_structure():
    """Check required directories and files exist."""
    errors = []
    for d in [DATA_DIR, SRC_DIR, STATE_DIR, KB_DIR, OUTPUT_DIR, AUDIT_DIR,
              OUTPUT_DIR / "rounds", OUTPUT_DIR / "pools"]:
        d.mkdir(parents=True, exist_ok=True)

    required_files = [
        KB_DIR / "K_star.json",
        STATE_DIR / "CONFIG.yaml",
        STATE_DIR / "CONTRACT.json",
        STATE_DIR / "STATE.json",
        STATE_DIR / "COST_LEDGER.json",
    ]
    for f in required_files:
        if not f.exists():
            errors.append(f"Missing: {f}")

    if errors:
        raise ValidationError("repo_structure: " + "; ".join(errors))
    return True


def validate_config_immutables():
    """Check CONTRACT.json immutable keys haven't changed."""
    contract_path = STATE_DIR / "CONTRACT.json"
    if not contract_path.exists():
        raise ValidationError("CONTRACT.json missing")

    with open(contract_path) as f:
        contract = json.load(f)

    if contract.get("run_mode") == "RUN":
        # In RUN mode, verify hashes of key scripts haven't changed
        hash_path = STATE_DIR / "framework_hashes.json"
        if hash_path.exists():
            with open(hash_path) as f:
                old_hashes = json.load(f)
            for script_name, old_hash in old_hashes.items():
                script_path = SRC_DIR / script_name
                if script_path.exists():
                    new_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
                    if new_hash != old_hash:
                        raise ValidationError(f"Script {script_name} changed in RUN mode")
    return True


def validate_models_exist():
    """Check K_star.json exists, sklearn models exist."""
    kstar_path = KB_DIR / "K_star.json"
    if not kstar_path.exists():
        raise ValidationError("K_star.json missing")

    with open(kstar_path) as f:
        kstar = json.load(f)
    if "hypotheses" not in kstar or len(kstar["hypotheses"]) == 0:
        raise ValidationError("K_star.json has no hypotheses")

    # Check model3 files
    model3_dir = PROJECT_ROOT / "models" / "model3"
    if not model3_dir.exists() or not list(model3_dir.glob("*.pkl")):
        raise ValidationError("Model 3 files missing")

    return True


def validate_pool_files_exist_or_create():
    """Ensure all 12 pool CSVs exist."""
    pools_dir = OUTPUT_DIR / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)

    for m in METHODS:
        for suffix in ["overall", "LLM"]:
            pool_path = pools_dir / f"pool_{m}_{suffix}.csv"
            if not pool_path.exists():
                with open(pool_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    header = ["url", "title", "classified_method", "tied_methods",
                              "confidence", "credit", "round_id", "source"]
                    writer.writerow(header)
    return True


def validate_round_outputs(round_id):
    """Validate all required round output files exist and have correct schema."""
    errors = []
    round_dir = OUTPUT_DIR / "rounds" / f"round_{round_id}"
    if not round_dir.exists():
        raise ValidationError(f"Round directory missing: {round_dir}")

    # Check per-model CSVs
    state = json.load(open(STATE_DIR / "STATE.json"))
    model2_avail = state.get("model2_available", False)

    required_csvs = [
        f"round_{round_id}_model1_papers.csv",
        f"round_{round_id}_model3_papers.csv",
        f"round_{round_id}_model4_papers.csv",
        f"round_{round_id}_high_relevant_papers.csv",
    ]
    if model2_avail:
        required_csvs.append(f"round_{round_id}_model2_papers.csv")

    for fname in required_csvs:
        fpath = round_dir / fname
        if not fpath.exists():
            errors.append(f"Missing: {fpath}")

    # Check query_log has a row for this round
    qlog_path = OUTPUT_DIR / "query_log.csv"
    if qlog_path.exists():
        with open(qlog_path) as f:
            reader = csv.DictReader(f)
            round_ids = [row.get("round_id") for row in reader]
        if str(round_id) not in round_ids:
            errors.append(f"query_log missing round {round_id}")

    if errors:
        raise ValidationError("round_outputs: " + "; ".join(errors))
    return True


def validate_scored_rows(scored_results):
    """Validate scored rows have outputs for all models and all 7 dims."""
    errors = []
    for i, row in enumerate(scored_results):
        for dim in DIMS:
            for model_key in ["model1", "model3", "model4"]:
                model_data = row.get(model_key, {})
                if model_data is None:
                    continue
                dim_data = model_data.get(dim)
                if dim_data is None:
                    errors.append(f"Row {i}: {model_key} missing {dim}")
                    continue
                # Check probabilities sum to ~1
                p_sum = dim_data.get("p0", 0) + dim_data.get("p05", 0) + dim_data.get("p1", 0)
                if abs(p_sum - 1.0) > 0.02:
                    errors.append(f"Row {i}: {model_key}/{dim} probs sum={p_sum:.3f}")
                # Check score valid
                score = dim_data.get("score")
                if score not in (0, 0.5, 1):
                    errors.append(f"Row {i}: {model_key}/{dim} invalid score={score}")

    if errors:
        raise ValidationError("scored_rows: " + "; ".join(errors[:10]))
    return True


def validate_query(query_str, target_method):
    """Validate query contains required clauses."""
    errors = []
    q = query_str.lower()

    if q.count(" and ") < 2:
        errors.append("Query must contain AND at least twice")

    # Check for at least one policy context token
    policy_tokens = ["policy", "programme", "program", "scheme", "service",
                     "government", "council", "nhs"]
    if not any(t in q for t in policy_tokens):
        errors.append("Missing policy context term")

    # Check for at least one decision term
    decision_tokens = ["pilot", "rolled out", "trialled", "implemented",
                       "introduced", "launched", "funded", "plans to"]
    if not any(t in q for t in decision_tokens):
        errors.append("Missing decision term")

    if errors:
        raise ValidationError("query: " + "; ".join(errors))
    return True


def save_framework_hashes():
    """Save SHA256 hashes of key scripts."""
    scripts = ["run_query_loop.py", "round_runner.py", "bandit.py",
               "query_builder.py", "framework_contract.py", "knowledge_synthesizer.py"]
    hashes = {}
    for s in scripts:
        path = SRC_DIR / s
        if path.exists():
            hashes[s] = hashlib.sha256(path.read_bytes()).hexdigest()
    hash_path = STATE_DIR / "framework_hashes.json"
    with open(hash_path, "w") as f:
        json.dump(hashes, f, indent=2)
    return hashes


def write_round_audit(round_id, audit_data):
    """Write per-round audit files."""
    audit_dir = AUDIT_DIR / f"round_{round_id}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Markdown audit
    md_path = audit_dir / f"round_{round_id}_audit.md"
    with open(md_path, "w") as f:
        f.write(f"# Round {round_id} Audit\n\n")
        for key, val in audit_data.items():
            f.write(f"- **{key}**: {val}\n")

    # JSON contract
    json_path = audit_dir / f"round_{round_id}_contract.json"
    with open(json_path, "w") as f:
        json.dump(audit_data, f, indent=2, default=str)

    return md_path, json_path
