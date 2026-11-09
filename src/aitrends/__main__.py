"""CLI: python -m aitrends <stage> [<stage> ...]

Stages (run in this order by `all`):
  collect   download raw data for the rolling window (Bluesky, Hacker News, Mastodon, Reddit)
  corpus    unify + filter + label documents
  enrich    embeddings + sentiment (cached, incremental)
  volume    count statistics (NB GLMs, bursts, change points, share of voice, outlook)
  text      topics, momentum, emerging terms, sentiment tests, engagement model
  history   append weekly headline metrics to history/weekly_metrics.csv
  site      render the static website
  all       analysis stages (corpus..site); add `collect` first to refresh data
"""
from __future__ import annotations

import json
import sys
import time

from . import config as C

STAGES = ["collect", "corpus", "enrich", "volume", "text", "history", "site"]


def run_stage(name: str) -> None:
    t = time.time()
    print(f"==> {name}", flush=True)
    if name == "collect":
        from .collect import bluesky, hackernews, mastodon, reddit
        bluesky.collect_counts()
        bluesky.collect_posts()
        hackernews.collect()
        reddit.collect()
        mastodon.collect()
    elif name == "corpus":
        from . import corpus
        corpus.build()
    elif name == "enrich":
        from . import enrich
        enrich.run()
    elif name == "volume":
        from .analysis import volume
        volume.run()
    elif name == "text":
        from .analysis import text
        text.run()
    elif name == "history":
        from . import history
        history.run()
    elif name == "site":
        from .site import build
        build.build()
    else:
        raise SystemExit(f"unknown stage {name!r}; choose from {STAGES + ['all']}")
    print(f"<== {name} done in {time.time() - t:.0f}s", flush=True)


def main(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    stages = []
    for a in argv:
        stages += STAGES[1:] if a == "all" else [a]
    print("window:", json.dumps(C.window_label()), flush=True)
    (C.OUT_DIR / "run_meta.json").write_text(json.dumps({"window": C.window_label(), "stages": stages}, indent=1))
    for s in stages:
        run_stage(s)


if __name__ == "__main__":
    main(sys.argv[1:])
