"""Bluesky (public AppView search API, no auth): daily keyword counts + top posts."""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor

from .. import config as C
from ..http import get_json

API = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
CAP = 10_000  # hitsTotal saturates here


def _hits(q: str, a: dt.datetime, b: dt.datetime) -> int | None:
    res = get_json(API, {"q": q, "since": C.iso(a), "until": C.iso(b), "limit": 1, "lang": "en"})
    return res.get("hitsTotal") if "_err" not in res else None


def _hits_uncapped(q: str, a: dt.datetime, b: dt.datetime) -> int | None:
    n = _hits(q, a, b)
    if n is not None and n >= CAP and (b - a) > dt.timedelta(minutes=30):
        mid = a + (b - a) / 2
        x, y = _hits_uncapped(q, a, mid), _hits_uncapped(q, mid, b)
        return None if x is None or y is None else x + y
    return n


def collect_counts() -> None:
    jobs = [(k, q, d) for k, q in C.BSKY_COUNT_KEYWORDS.items() for d in C.days()]

    def one(job):
        k, q, d = job
        if k in C.BSKY_HIGH_VOLUME:   # 3-hour slices, recursively split if still capped
            total, t = 0, d
            while t < d + dt.timedelta(days=1):
                n = _hits_uncapped(q, t, t + dt.timedelta(hours=3))
                if n is None:
                    total = None
                    break
                total += n
                t += dt.timedelta(hours=3)
            n = total
        else:
            n = _hits(q, d, d + dt.timedelta(days=1))
        return {"kw": k, "day": d.date().isoformat(), "hits": n}

    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(one, jobs))
    (C.DATA_DIR / "bsky_counts.json").write_text(json.dumps(rows))
    print(f"bluesky counts: {len(rows)} keyword-days, {sum(r['hits'] is None for r in rows)} failed", flush=True)


def collect_posts(pages: int = 2) -> None:
    jobs = [(k, q, d) for k, q in C.BSKY_POST_QUERIES.items() for d in C.days()]

    def one(job):
        k, q, d = job
        out, cursor = [], None
        for _ in range(pages):
            p = {"q": q, "since": C.iso(d), "until": C.iso(d + dt.timedelta(days=1)), "limit": 100,
                 "sort": "top", "lang": "en"}
            if cursor:
                p["cursor"] = cursor
            res = get_json(API, p)
            for po in res.get("posts", []):
                rec = po.get("record", {})
                out.append({"uri": po["uri"], "author": po["author"]["handle"], "text": rec.get("text", ""),
                            "created": rec.get("createdAt"), "likes": po.get("likeCount", 0),
                            "reposts": po.get("repostCount", 0), "replies": po.get("replyCount", 0),
                            "quotes": po.get("quoteCount", 0), "query": k})
            cursor = res.get("cursor")
            if not cursor:
                break
        return out

    rows = []
    with ThreadPoolExecutor(6) as ex:
        for r in ex.map(one, jobs):
            rows.extend(r)
    (C.DATA_DIR / "bsky_posts.json").write_text(json.dumps(rows))
    print(f"bluesky posts: {len(rows)}", flush=True)
