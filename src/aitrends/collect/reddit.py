"""Reddit posts via the Arctic Shift archive API (reddit.com itself blocks most datacenter IPs)."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from .. import config as C
from ..http import get_json

API = "https://arctic-shift.photon-reddit.com/api/posts/search"
FIELDS = "id,title,selftext,score,num_comments,created_utc,subreddit,link_flair_text,url,author"


def _sub(name: str, a: int, b: int) -> list[dict]:
    out, after = [], a
    while True:
        res = get_json(API, {"subreddit": name, "after": after, "before": b, "limit": 100,
                             "sort": "asc", "fields": FIELDS}, timeout=90)
        data = res.get("data") if isinstance(res, dict) else None
        if not data:
            if "_err" in res:
                print(f"reddit: r/{name} failed at {after}: {res.get('_err')}", flush=True)
            break
        out.extend(data)
        nxt = data[-1]["created_utc"]
        after = nxt if nxt > after else after + 1
        if len(data) < 100:
            break
    for o in out:
        o["selftext"] = (o.get("selftext") or "")[:1500]
    print(f"reddit: r/{name} {len(out)}", flush=True)
    return out


def collect() -> None:
    base, _, end = C.window()
    a, b = int(base.timestamp()), int(end.timestamp())
    rows = []
    with ThreadPoolExecutor(2) as ex:
        for r in ex.map(lambda s: _sub(s, a, b), C.REDDIT_SUBS):
            rows.extend(r)
    uniq = {r["id"]: r for r in rows}
    (C.DATA_DIR / "reddit.json").write_text(json.dumps(list(uniq.values())))
    print(f"reddit: {len(uniq)} posts", flush=True)
