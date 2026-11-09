"""Hacker News via the Algolia search API: story titles + comment counts per query-day."""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor

from .. import config as C
from ..http import get_json

API = "https://hn.algolia.com/api/v1/search_by_date"


def _fetch(q: str, d: dt.datetime) -> dict:
    nd = d + dt.timedelta(days=1)
    nf = f"created_at_i>={int(d.timestamp())},created_at_i<{int(nd.timestamp())}"
    st = get_json(API, {"query": q, "tags": "story", "numericFilters": nf, "hitsPerPage": 1000})
    cm = get_json(API, {"query": q, "tags": "comment", "numericFilters": nf, "hitsPerPage": 0})
    stories = [{"id": h["objectID"], "title": h.get("title") or "", "url": h.get("url"),
                "points": h.get("points") or 0, "comments": h.get("num_comments") or 0,
                "created": h.get("created_at"), "author": h.get("author")} for h in st.get("hits", [])]
    return {"q": q, "day": d.date().isoformat(), "story_hits": st.get("nbHits"),
            "comment_hits": cm.get("nbHits"), "stories": stories}


def collect() -> None:
    jobs = [(q, d) for q in C.HN_QUERIES for d in C.days()]
    with ThreadPoolExecutor(3) as ex:
        rows = list(ex.map(lambda j: _fetch(*j), jobs))
    # second pass for anything that failed (rate limiting)
    for _ in range(3):
        bad = [i for i, r in enumerate(rows) if r["comment_hits"] is None or r["story_hits"] is None]
        if not bad:
            break
        print(f"hacker news: retrying {len(bad)} failed query-days", flush=True)
        for i in bad:
            r = rows[i]
            rows[i] = _fetch(r["q"], dt.datetime.fromisoformat(r["day"]).replace(tzinfo=C.UTC))
    (C.DATA_DIR / "hn.json").write_text(json.dumps(rows))
    print(f"hacker news: {len(rows)} query-days, {sum(len(r['stories']) for r in rows)} story hits, "
          f"{sum(r['comment_hits'] is None for r in rows)} failed", flush=True)
