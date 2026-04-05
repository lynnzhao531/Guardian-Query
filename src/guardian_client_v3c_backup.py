from __future__ import annotations
"""Guardian API client with key rotation and throttling (MASTER_PLAN_v3 §10)."""

import json
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_KEY_STATE_PATH = _PROJECT_ROOT / "project_state" / "guardian_key_state.json"
_ROTATION_LOG = _PROJECT_ROOT / "audits" / "guardian_api_rotation.log"
_ENV_PATH = _PROJECT_ROOT / ".env"

SEARCH_ENDPOINT = "https://content.guardianapis.com/search"

# ---------------------------------------------------------------------------
# Logging (rotation log - never contains key values)
# ---------------------------------------------------------------------------
_rot_logger = logging.getLogger("guardian_rotation")
_rot_logger.setLevel(logging.INFO)
if not _rot_logger.handlers:
    _ROTATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(_ROTATION_LOG, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    _rot_logger.addHandler(_fh)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fingerprint(key: str) -> str:
    """Last 4 chars of a key - safe to log."""
    return f"...{key[-4:]}" if key else "NONE"


def _is_quota_error(resp: requests.Response) -> bool:
    """True when the response signals rate-limit or quota exhaustion."""
    if resp.status_code == 429:
        return True
    if resp.status_code in (401, 403):
        body = resp.text.lower()
        if any(w in body for w in ("rate", "quota", "limit", "exceeded")):
            return True
    return False


def _is_retryable(resp: requests.Response) -> bool:
    return resp.status_code in (429, 503, 529)


# ---------------------------------------------------------------------------
# GuardianClient
# ---------------------------------------------------------------------------

class GuardianClient:
    """Thread-safe Guardian Content API client with key rotation."""

    def __init__(self, *, min_interval_sec: float = 1.0):
        load_dotenv(_ENV_PATH)
        self._keys: list[str] = []
        for i in (1, 2, 3):
            k = os.getenv(f"GUARDIAN_API_KEY_{i}")
            if k:
                self._keys.append(k)
        if not self._keys:
            raise RuntimeError("No GUARDIAN_API_KEY_* found in .env")

        self._min_interval = min_interval_sec
        self._last_used: dict[str, float] = {k: 0.0 for k in self._keys}
        self._active_idx: int = 0
        self._rotation_count: int = 0
        self._KEY_STATE_PATH = _KEY_STATE_PATH
        self._load_state()

    # -- state persistence ---------------------------------------------------

    def _load_state(self):
        if self._KEY_STATE_PATH.exists():
            try:
                state = json.loads(self._KEY_STATE_PATH.read_text())
                idx = state.get("active_idx", 0)
                if 0 <= idx < len(self._keys):
                    self._active_idx = idx
                self._rotation_count = state.get("rotation_count", 0)
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_state(self):
        self._KEY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._KEY_STATE_PATH.write_text(json.dumps({
            "active_idx": self._active_idx,
            "rotation_count": self._rotation_count,
            "active_fingerprint": _fingerprint(self._keys[self._active_idx]),
        }, indent=2))

    # -- key management ------------------------------------------------------

    @property
    def _active_key(self) -> str:
        return self._keys[self._active_idx]

    def _rotate(self, reason: str):
        old_fp = _fingerprint(self._active_key)
        self._active_idx = (self._active_idx + 1) % len(self._keys)
        self._rotation_count += 1
        new_fp = _fingerprint(self._active_key)
        _rot_logger.info("ROTATE  %s -> %s  reason=%s  total=%d",
                         old_fp, new_fp, reason, self._rotation_count)
        self._save_state()

    def _throttle(self, key: str):
        elapsed = time.time() - self._last_used[key]
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_used[key] = time.time()

    # -- core request --------------------------------------------------------

    def _request(self, params: dict, *, max_retries: int = 3) -> dict:
        """Execute a Guardian API request with retry + rotation."""
        for attempt in range(max_retries + 1):
            key = self._active_key
            self._throttle(key)
            params_with_key = {**params, "api-key": key}

            resp = requests.get(SEARCH_ENDPOINT, params=params_with_key, timeout=30)

            if resp.ok:
                return resp.json()

            if _is_quota_error(resp):
                self._rotate(f"quota/status={resp.status_code}")
                continue

            if _is_retryable(resp) and attempt < max_retries:
                self._rotate(f"retryable/status={resp.status_code}")
                backoff = 2 ** attempt
                logger.warning("Guardian %d, backoff %.1fs", resp.status_code, backoff)
                time.sleep(backoff)
                continue

            resp.raise_for_status()

        raise RuntimeError("Guardian API: max retries exceeded")

    # -- public API ----------------------------------------------------------

    def search(self, query: str, page: int = 1, page_size: int = 50,
               section: str = "", order_by: str = "relevance") -> dict:
        """Raw search returning parsed JSON response."""
        params = {
            "q": query,
            "page": page,
            "page-size": page_size,
            "show-fields": "bodyText,headline",
            "order-by": order_by,
        }
        if section:
            params["section"] = section
        return self._request(params)

    def preflight(self, query: str, section: str = "") -> int:
        """Return total_available for *query* without fetching bodies.

        If caller does not supply ``section``, fall back to the project-wide
        garbage-section exclusion so preflight totals match actual retrieval.
        """
        if not section:
            try:
                from query_builder import get_section_filter
                section = get_section_filter()
            except Exception:
                section = ""
        params = {
            "q": query,
            "page": 1,
            "page-size": 1,
            "show-fields": "headline",
        }
        if section:
            params["section"] = section
        data = self._request(params)
        return data.get("response", {}).get("total", 0)

    def fetch_pages(self, query: str, pages: int = 5,
                    page_size: int = 50,
                    total_available: int = 0,
                    section: str = "",
                    order_by: str = "relevance") -> list[dict]:
        """Fetch multiple pages and return flat list of article dicts."""
        import math
        # Cap pages to what actually exists
        if total_available > 0:
            max_pages = math.ceil(total_available / page_size)
            pages = min(pages, max_pages)
        articles: list[dict] = []
        for pg in range(1, pages + 1):
            try:
                data = self.search(query, page=pg, page_size=page_size,
                                   section=section, order_by=order_by)
            except Exception as e:
                logger.warning("fetch_pages page %d failed: %s", pg, e)
                break
            results = data.get("response", {}).get("results", [])
            if not results:
                break
            for r in results:
                fields = r.get("fields", {})
                articles.append({
                    "title": fields.get("headline", r.get("webTitle", "")),
                    "url": r.get("webUrl", ""),
                    "body_text": fields.get("bodyText", ""),
                    "sectionId": r.get("sectionId", ""),
                })
        return articles

    def get_rotation_stats(self) -> dict:
        return {
            "active_fingerprint": _fingerprint(self._active_key),
            "active_idx": self._active_idx,
            "rotation_count": self._rotation_count,
            "num_keys": len(self._keys),
        }
