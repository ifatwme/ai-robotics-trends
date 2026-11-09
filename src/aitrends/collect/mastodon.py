"""Mastodon public hashtag timelines (no auth). Samples up to N newest posts per tag-day.

Mastodon status IDs are snowflakes (ms timestamp << 16), so day boundaries can be addressed via max_id/since_id.
"""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor

from .. import config as C
from ..http import get_json


def _ts_to_id(t: dt.datetime) -> str:
    return str(int(t.timestamp() * 1000) << 16)


def collect() -> None:
    base = f"{C.MASTODON_INSTANCE}/api/v1/timelines/tag/"
    jobs = [(tag, d) for tag in C.MASTODON_TAGS for d in C.days()]

    def one(job):
        tag, d = job
        out, max_id = [], _ts_to_id(d + dt.timedelta(days=1))
        while len(out) < C.MASTODON_MAX_PER_DAY:
            res = get_json(base + tag, {"limit": 40, "max_id": max_id, "since_id": _ts_to_id(d)})
            if not isinstance(res, list) or not res:
                break
            for s in res:
                out.append({"id": s["id"], "created": s["created_at"], "lang": s.get("language"),
                            "html": s.get("content", ""), "favs": s.get("favourites_count", 0),
                            "boosts": s.get("reblogs_count", 0), "replies": s.get("replies_count", 0),
                            "acct": s["account"]["acct"], "tag": tag})
            if len(res) < 40:
                break
            max_id = res[-1]["id"]
        return out

    rows = []
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(one, jobs):
            rows.extend(r)
    (C.DATA_DIR / "masto.json").write_text(json.dumps(rows))
    print(f"mastodon: {len(rows)} posts", flush=True)
