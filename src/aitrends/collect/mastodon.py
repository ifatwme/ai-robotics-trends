"""Mastodon public hashtag timelines (no auth). Samples up to N newest posts per tag-day.

Mastodon status IDs are snowflakes (ms timestamp << 16), so day boundaries can be addressed via max_id/since_id.
Failed tag-days are retried and, if still failing, recorded in masto_failed.json so the volume models treat
them as missing (NaN) instead of zero.
"""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor

from .. import config as C
from ..http import get_json

BASE = f"{C.MASTODON_INSTANCE}/api/v1/timelines/tag/"


def _ts_to_id(t: dt.datetime) -> str:
    return str(int(t.timestamp() * 1000) << 16)


def _fetch(job) -> tuple[str, str, list[dict], bool]:
    tag, d = job
    out, max_id, ok = [], _ts_to_id(d + dt.timedelta(days=1)), True
    while len(out) < C.MASTODON_MAX_PER_DAY:
        res = get_json(BASE + tag, {"limit": 40, "max_id": max_id, "since_id": _ts_to_id(d)})
        if not isinstance(res, list):
            ok = False
            break
        if not res:
            break
        for s in res:
            out.append({"id": s["id"], "created": s["created_at"], "lang": s.get("language"),
                        "html": s.get("content", ""), "favs": s.get("favourites_count", 0),
                        "boosts": s.get("reblogs_count", 0), "replies": s.get("replies_count", 0),
                        "acct": s["account"]["acct"], "bot": bool(s["account"].get("bot")), "tag": tag})
        if len(res) < 40:
            break
        max_id = res[-1]["id"]
    return tag, d.date().isoformat(), out, ok


def collect(fill_gaps: bool = False) -> None:
    """Full collection, or (fill_gaps=True) re-fetch only tag-days that are empty/failed in the existing file."""
    path, fpath = C.DATA_DIR / "masto.json", C.DATA_DIR / "masto_failed.json"
    jobs = [(tag, d) for tag in C.MASTODON_TAGS for d in C.days()]
    rows: list[dict] = []
    if fill_gaps and path.exists():
        rows = json.loads(path.read_text())
        cnt: dict[tuple[str, str], int] = {}
        for r in rows:
            cnt[(r["tag"], r["created"][:10])] = cnt.get((r["tag"], r["created"][:10]), 0) + 1
        todo = set()
        for tag in C.MASTODON_TAGS:
            per_day = [cnt.get((tag, d.date().isoformat()), 0) for d in C.days()]
            saturated = sorted(per_day)[len(per_day) // 2] >= 0.9 * C.MASTODON_MAX_PER_DAY
            for d, n in zip(C.days(), per_day):
                # empty day, or (for tags that normally hit the cap) a partial day = interrupted pagination
                if n == 0 or (saturated and n < C.MASTODON_MAX_PER_DAY * 0.9):
                    todo.add((tag, d.date().isoformat()))
        rows = [r for r in rows if (r["tag"], r["created"][:10]) not in todo]
        jobs = [(t, d) for t, d in jobs if (t, d.date().isoformat()) in todo]
        print(f"mastodon: re-fetching {len(jobs)} empty/partial tag-days", flush=True)
    failed = []
    for attempt in range(3):
        with ThreadPoolExecutor(3 if attempt == 0 else 1) as ex:
            results = list(ex.map(_fetch, jobs))
        failed = []
        for tag, day, out, ok in results:
            rows.extend(out)
            if not ok:
                failed.append((tag, day))
        if not failed:
            break
        print(f"mastodon: retrying {len(failed)} failed tag-days", flush=True)
        jobs = [(t, dt.datetime.fromisoformat(d).replace(tzinfo=C.UTC)) for t, d in failed]
    uniq = {(r["tag"], r["id"]): r for r in rows}
    path.write_text(json.dumps(list(uniq.values())))
    fpath.write_text(json.dumps([{"tag": t, "day": d} for t, d in failed]))
    print(f"mastodon: {len(uniq)} posts, {len(failed)} tag-days still failed", flush=True)
