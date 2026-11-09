"""Polite HTTP client: retries with backoff and a per-host minimum request interval."""
from __future__ import annotations

import threading
import time
from urllib.parse import urlparse

import requests

USER_AGENT = "ai-robotics-trends/1.0 (+research; weekly aggregate statistics)"
# Minimum seconds between requests to the same host (keeps us under public rate limits)
MIN_INTERVAL = {
    "hn.algolia.com": 0.40,           # ~10k req/hour limit
    "mastodon.social": 1.05,          # 300 req / 5 min for anonymous clients
    "api.bsky.app": 0.08,
    "arctic-shift.photon-reddit.com": 1.0,   # returns 422 "Timeout. Maybe slow down" under load
}

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_locks: dict[str, threading.Lock] = {}
_last: dict[str, float] = {}
_guard = threading.Lock()


def _throttle(host: str) -> None:
    gap = MIN_INTERVAL.get(host, 0.1)
    with _guard:
        lock = _locks.setdefault(host, threading.Lock())
    with lock:
        wait = _last.get(host, 0) + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[host] = time.monotonic()


def get_json(url: str, params: dict | None = None, tries: int = 7, timeout: int = 45):
    """GET JSON. Returns parsed JSON, or {"_err": status} after exhausting retries."""
    host = urlparse(url).netloc
    status = "fail"
    for attempt in range(tries):
        _throttle(host)
        try:
            r = _session.get(url, params=params, timeout=timeout)
            status = r.status_code
            if r.status_code == 200:
                return r.json()
            transient = r.status_code in (429, 500, 502, 503, 504) or (
                r.status_code == 422 and ("timeout" in r.text.lower() or "slow down" in r.text.lower()))
            if transient:
                reset = r.headers.get("Retry-After")
                time.sleep(float(reset) if reset and reset.isdigit() else min(60, 3 * 2 ** attempt))
                continue
            return {"_err": r.status_code, "_body": r.text[:200]}
        except requests.RequestException:
            time.sleep(min(60, 3 * 2 ** attempt))
    return {"_err": status}
