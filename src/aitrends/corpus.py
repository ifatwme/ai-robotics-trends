"""Unify all platforms into one document table (DATA_DIR/corpus.pkl) with domain + period labels."""
from __future__ import annotations

import html
import json
import re

import numpy as np
import pandas as pd

from . import config as C

AI_CS = re.compile(r"\b(AI|A\.I\.|AGI|ASI|LLMs?|GPT[-\w.]*|ML|GenAI|NLP|SLMs?|VLMs?|VLAs?)\b")
AI_CI = re.compile(
    r"(artificial intelligence|machine learning|deep learning|neural net|\bllm|chatgpt|openai|anthropic|\bclaude\b|"
    r"\bgemini\b|\bgrok\b|\bxai\b|deepseek|\bqwen|\bllama\b|mistral|copilot|\bcodex\b|cursor ai|chatbot|generative|"
    r"agentic|ai agent|\bsora\b|midjourney|stable diffusion|nvidia|deepfake|vibe.?cod|prompt engineer|fine.?tun|"
    r"transformer model|diffusion model|\bkimi\b|perplexity|hugging ?face|superintelligen|data ?cent(er|re)|"
    r"coding agent|\bmodel weights|\bopus\b|\bsonnet\b)", re.I)
ROBO = re.compile(
    r"(robot|humanoid|optimus|unitree|figure ai|figure 0\d|boston dynamics|robotaxi|waymo|zoox|self.driving|"
    r"driverless|autonomous (vehicle|car|driving|truck)|\bfsd\b|\bdrones?\b|exoskeleton|\bcobot|quadruped|gripper|"
    r"\b1x\b|neo gamma|apptronik|agility robotics|\bros2?\b|manipulator|physical ai|embodied)", re.I)
ROBO_TAGS = {"#robotics", "#robot", "#robots", "#humanoid"}


def classify(text: str) -> str | None:
    a = bool(AI_CS.search(text) or AI_CI.search(text))
    r = bool(ROBO.search(text))
    return "AI+Robotics" if a and r else "AI" if a else "Robotics" if r else None


def _strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>|</p>", " ", s or "")
    return html.unescape(re.sub(r"<[^>]+>", "", s))


def _load(name: str):
    p = C.DATA_DIR / name
    if not p.exists():
        print(f"corpus: {name} missing, skipped", flush=True)
        return []
    return json.loads(p.read_text())


def build() -> pd.DataFrame:
    rows = []
    for p in _load("bsky_posts.json"):
        rows.append(dict(platform="Bluesky", id="bs:" + p["uri"], text=p["text"], created=p["created"],
                         author=p["author"], eng=p["likes"] + 2 * p["reposts"] + p["replies"] + p["quotes"],
                         likes=p["likes"], comments=p["replies"], source="bsky:" + p["query"], url=None))
    stories = {s["id"]: s for r in _load("hn.json") for s in r["stories"]}
    for s in stories.values():
        rows.append(dict(platform="HackerNews", id="hn:" + s["id"], text=s["title"], created=s["created"],
                         author=s["author"], eng=s["points"] + s["comments"], likes=s["points"],
                         comments=s["comments"], source="hn:story",
                         url=f"https://news.ycombinator.com/item?id={s['id']}"))
    for m in _load("masto.json"):
        if m.get("lang") not in (None, "en"):
            continue
        rows.append(dict(platform="Mastodon", id="ma:" + m["id"], text=_strip_html(m["html"]), created=m["created"],
                         author=m["acct"], eng=m["favs"] + 2 * m["boosts"] + m["replies"], likes=m["favs"],
                         comments=m["replies"], source="#" + m["tag"], url=None))
    for p in _load("reddit.json"):
        body = p.get("selftext") or ""
        text = (p.get("title") or "") + ("" if body in ("", "[removed]", "[deleted]") else ". " + body)
        rows.append(dict(platform="Reddit", id="rd:" + p["id"], text=text,
                         created=pd.to_datetime(p["created_utc"], unit="s", utc=True).isoformat(),
                         author=p.get("author"), eng=(p.get("score") or 0) + (p.get("num_comments") or 0),
                         likes=p.get("score") or 0, comments=p.get("num_comments") or 0,
                         source="r/" + p["subreddit"], url=f"https://www.reddit.com/comments/{p['id']}"))

    base, cur, end = C.window()
    df = pd.DataFrame(rows)
    df["created"] = pd.to_datetime(df.created, utc=True, errors="coerce", format="mixed")
    df = df[(df.created >= base) & (df.created < end)].copy()
    df["text"] = df.text.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df.text.str.len() >= 15].drop_duplicates("id")

    df["domain"] = df.text.map(classify)
    sub = df.source.str.removeprefix("r/")
    df.loc[df.domain.isna() & sub.isin(C.REDDIT_AI_SUBS), "domain"] = "AI"
    df.loc[df.domain.isna() & sub.isin(C.REDDIT_ROBOTICS_SUBS), "domain"] = "Robotics"
    df.loc[df.domain.isna() & df.source.isin(ROBO_TAGS), "domain"] = "Robotics"
    df.loc[df.domain.isna() & (df.platform == "Mastodon"), "domain"] = "AI"
    df = df[df.domain.notna()].copy()

    norm = df.text.str.lower().str.replace(r"https?://\S+", "", regex=True).str[:200]
    dup = pd.Series(list(zip(df.platform, norm))).duplicated().values
    print(f"corpus: dropped {dup.sum()} exact duplicates", flush=True)
    df = df[~dup]
    df["period"] = np.where(df.created >= cur, "current", "baseline")
    df["day"] = df.created.dt.strftime("%Y-%m-%d")
    df = df.reset_index(drop=True)
    df.to_pickle(C.DATA_DIR / "corpus.pkl")
    print("corpus:\n" + df.groupby(["platform", "period"]).size().unstack().to_string(), flush=True)
    return df
