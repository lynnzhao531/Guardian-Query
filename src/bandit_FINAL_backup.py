"""Contextual Linear Thompson Sampling bandit — REVISED_ARCHITECTURE.md §5.

11-feature space, 6-component reward, 3-phase curriculum.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STATE_PATH = _PROJECT_ROOT / "project_state" / "BANDIT_STATE.json"

METHOD_NAMES = [
    "rct", "prepost", "case_study",
    "expert_qual", "expert_secondary", "gut",
]

# §5.2: 11 features
FEATURE_NAMES = [
    "target_progress",      # 0: progress of target method (0-1)
    "overall_progress",     # 1: avg progress across all methods (0-1)
    "round_phase",          # 2: curriculum phase (1/3, 2/3, or 1.0)
    "recent_unique_rate",   # 3: unique rate from last 5 rounds (0-1)
    "recent_tier_b_rate",   # 4: tier B rate from last 5 rounds (0-1)
    "method_width",         # 5: k/10 (0.3 to 1.0)
    "decision_width",       # 6: num decision terms / 20
    "has_full_not",         # 7: 1 if NOT clause applied, 0 otherwise
    "est_log_available",    # 8: log1p(total_available) / 10
    "method_is_fresh",      # 9: 1 if target_progress < 0.1
    "query_novelty",        # 10: 1 if novel query, 0 if duplicate
]
FEATURE_DIM = len(FEATURE_NAMES)  # 11


class Bandit:
    """Contextual Linear Thompson Sampling with decay and curriculum."""

    def __init__(self, rho: float = 0.97):
        self.rho = rho
        self.d = FEATURE_DIM
        # Bayesian linear regression state
        self.B = np.eye(self.d, dtype=np.float64)
        self.f = np.zeros(self.d, dtype=np.float64)
        self.n_updates = 0

    # ── §5.4 Curriculum ─────────────────────────────────────────────────
    @staticmethod
    def get_phase(round_num: int) -> int:
        """Return curriculum phase: 1 (1-20), 2 (21-60), 3 (61+)."""
        if round_num <= 20:
            return 1
        elif round_num <= 60:
            return 2
        return 3

    @staticmethod
    def get_epsilon(round_num: int) -> float:
        """Return exploration rate for curriculum phase."""
        phase = Bandit.get_phase(round_num)
        if phase == 1:
            return 0.20
        elif phase == 2:
            return 0.10
        return 0.05

    # ── Feature extraction ──────────────────────────────────────────────
    @staticmethod
    def extract_features(candidate: Dict[str, Any]) -> np.ndarray:
        """Extract 11-dim feature vector from candidate dict."""
        feats = candidate.get("features", {})
        x = np.zeros(FEATURE_DIM, dtype=np.float64)
        for i, name in enumerate(FEATURE_NAMES):
            x[i] = float(feats.get(name, 0.0))
        return x

    # ── §5.4 Selection with curriculum ──────────────────────────────────
    def select_query(
        self,
        candidates: List[Dict[str, Any]],
        round_num: int,
    ) -> Dict[str, Any]:
        """Pick a candidate via Thompson Sampling with epsilon-greedy curriculum.

        Phase 1 (rounds 1-20): rotation with ε=0.20 exploration
        Phase 2 (rounds 21-60): bandit with ε=0.10, warm-started
        Phase 3 (rounds 61+): full bandit with ε=0.05
        """
        if not candidates:
            raise ValueError("Empty candidate list")

        phase = self.get_phase(round_num)
        epsilon = self.get_epsilon(round_num)

        # Phase 1: mostly rotation (round-robin across methods)
        if phase == 1 and self.n_updates < 5:
            # Pure rotation for first few rounds
            method_idx = (round_num - 1) % len(METHOD_NAMES)
            target = METHOD_NAMES[method_idx]
            method_cands = [c for c in candidates if c.get("method") == target]
            if method_cands:
                chosen = method_cands[0]
                chosen["_bandit_score"] = 0.0
                chosen["_selection_mode"] = "rotation"
                return chosen

        # Epsilon-greedy: explore with probability epsilon
        if np.random.random() < epsilon:
            chosen = candidates[np.random.randint(len(candidates))]
            chosen["_bandit_score"] = 0.0
            chosen["_selection_mode"] = "explore"
            return chosen

        # Thompson Sampling
        B_inv = np.linalg.inv(self.B)
        mu = B_inv @ self.f
        try:
            theta = np.random.multivariate_normal(mu, B_inv)
        except np.linalg.LinAlgError:
            logger.warning("Sampling failed; falling back to MAP estimate")
            theta = mu

        best_val = -np.inf
        best_cand = candidates[0]
        for cand in candidates:
            x = self.extract_features(cand)
            val = float(theta @ x)
            if val > best_val:
                best_val = val
                best_cand = cand

        best_cand["_bandit_score"] = best_val
        best_cand["_selection_mode"] = "thompson"
        logger.info("Bandit selected method=%s k=%s score=%.4f mode=thompson",
                    best_cand.get("method"), best_cand.get("width_k"), best_val)
        return best_cand

    # ── §5.3 Reward computation (6 components) ──────────────────────────
    @staticmethod
    def compute_reward(round_results: Dict[str, Any]) -> float:
        """R = V × [0.20·tier_a + 0.15·tier_b + 0.35·Σ(w_m·U_m) + 0.20·unique_rate - 0.15·dup_rate + 0.05·goldmine]

        V = min(1, unique_scored / 50)
        """
        unique_scored = round_results.get("unique_scored", 0)
        tier_a = round_results.get("tier_a_count", 0)
        tier_b = round_results.get("tier_b_count", 0)
        unique_rate = round_results.get("unique_rate", 0.0)
        dup_rate = round_results.get("duplicate_rate", 0.0)
        goldmine = 1.0 if round_results.get("goldmine_triggered", False) else 0.0

        # Method-weighted credit: Σ w_m · U_m
        method_credit = 0.0
        per_method = round_results.get("per_method_credit", {})
        per_progress = round_results.get("per_method_progress", {})
        for m in METHOD_NAMES:
            U_m = per_method.get(m, 0.0)
            prog_m = per_progress.get(m, 0.0)
            w_m = max(0.0, 1.0 - prog_m)
            method_credit += w_m * U_m

        V = min(1.0, unique_scored / 50.0)
        R = V * (
            0.20 * min(tier_a, 5) / 5.0
            + 0.15 * min(tier_b, 20) / 20.0
            + 0.35 * min(method_credit, 1.0)
            + 0.20 * unique_rate
            - 0.15 * dup_rate
            + 0.05 * goldmine
        )

        logger.info("Reward: V=%.2f, tier_a=%d, tier_b=%d, unique=%.2f, dup=%.2f → R=%.4f",
                    V, tier_a, tier_b, unique_rate, dup_rate, R)
        return float(R)

    # ── Posterior update with decay ─────────────────────────────────────
    def update(self, features: np.ndarray, reward: float) -> None:
        """Update posterior with decay ρ."""
        x = np.asarray(features, dtype=np.float64)
        self.B = self.rho * self.B + np.outer(x, x)
        self.f = self.rho * self.f + reward * x
        self.n_updates += 1
        logger.debug("Bandit updated (n=%d, reward=%.4f)", self.n_updates, reward)

    # ── Persistence ─────────────────────────────────────────────────────
    def save_state(self, path: str | Path | None = None) -> None:
        path = Path(path) if path else _STATE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "B": self.B.tolist(),
            "f": self.f.tolist(),
            "n_updates": self.n_updates,
            "rho": self.rho,
            "d": self.d,
        }
        with open(path, "w") as fh:
            json.dump(state, fh, indent=2)
        logger.info("Bandit state saved to %s (%d updates)", path, self.n_updates)

    def load_state(self, path: str | Path | None = None) -> None:
        path = Path(path) if path else _STATE_PATH
        if not path.exists():
            logger.info("No bandit state at %s; starting fresh", path)
            return
        with open(path) as fh:
            state = json.load(fh)
        d = state.get("d", self.d)
        if d != self.d:
            logger.warning("Dimension mismatch (saved=%d, current=%d); resetting", d, self.d)
            return
        self.B = np.array(state["B"], dtype=np.float64)
        self.f = np.array(state["f"], dtype=np.float64)
        self.n_updates = state.get("n_updates", 0)
        logger.info("Bandit state loaded from %s (%d updates)", path, self.n_updates)

    def reset(self) -> None:
        """Reset to fresh state (for clean slate)."""
        self.B = np.eye(self.d, dtype=np.float64)
        self.f = np.zeros(self.d, dtype=np.float64)
        self.n_updates = 0
        logger.info("Bandit reset to fresh state")
