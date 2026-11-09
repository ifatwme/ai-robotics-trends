"""Central configuration: paths, analysis window and keyword lists.

The analysis window is rolling:
    baseline = [end - CURRENT_DAYS - BASELINE_DAYS, end - CURRENT_DAYS)
    current  = [end - CURRENT_DAYS, end)
`end` is exclusive and defaults to today 00:00 UTC (so only complete days are used).
Override with WINDOW_END=YYYY-MM-DD for reproducible / historical runs.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

UTC = dt.timezone.utc

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "out"))
SITE_DIR = Path(os.environ.get("SITE_DIR", "site"))
HISTORY_DIR = Path(os.environ.get("HISTORY_DIR", "history"))
for _d in (DATA_DIR, OUT_DIR / "fig", SITE_DIR, HISTORY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CURRENT_DAYS = int(os.environ.get("CURRENT_DAYS", "7"))
BASELINE_DAYS = int(os.environ.get("BASELINE_DAYS", "28"))
REPO_URL = os.environ.get("REPO_URL", "")
SEED = 42


def window() -> tuple[dt.datetime, dt.datetime, dt.datetime]:
    """Return (baseline_start, current_start, end) as aware UTC datetimes."""
    raw = os.environ.get("WINDOW_END")
    if raw:
        end = dt.datetime.fromisoformat(raw).replace(tzinfo=UTC)
    else:
        now = dt.datetime.now(UTC)
        end = now.replace(hour=0, minute=0, second=0, microsecond=0)
    cur = end - dt.timedelta(days=CURRENT_DAYS)
    base = cur - dt.timedelta(days=BASELINE_DAYS)
    return base, cur, end


def days() -> list[dt.datetime]:
    base, _, end = window()
    out, d = [], base
    while d < end:
        out.append(d)
        d += dt.timedelta(days=1)
    return out


def iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def window_label() -> dict:
    base, cur, end = window()
    last = end - dt.timedelta(days=1)
    return {
        "baseline_start": base.date().isoformat(),
        "baseline_end": (cur - dt.timedelta(days=1)).date().isoformat(),
        "current_start": cur.date().isoformat(),
        "current_end": last.date().isoformat(),
        "end_exclusive": end.date().isoformat(),
        "current_days": CURRENT_DAYS,
        "baseline_days": BASELINE_DAYS,
    }


# ---------------------------------------------------------------- keywords
BSKY_COUNT_KEYWORDS = {
    # AI
    "AI": "AI", "artificial intelligence": '"artificial intelligence"', "LLM": "LLM",
    "ChatGPT": "ChatGPT", "OpenAI": "OpenAI", "Claude": "Claude", "Anthropic": "Anthropic",
    "Gemini": "Gemini", "Grok": "Grok", "DeepSeek": "DeepSeek", "Nvidia": "Nvidia",
    "AI agent": '"AI agent"', "AGI": "AGI", "generative AI": '"generative AI"',
    "AI slop": '"AI slop"', "AI bubble": '"AI bubble"', "data center": '"data center"',
    "deepfake": "deepfake", "AI art": '"AI art"', "vibe coding": '"vibe coding"',
    "AI regulation": '"AI regulation"', "AI jobs": '"AI jobs"', "Sora": "Sora",
    # Robotics
    "robot": "robot", "robotics": "robotics", "humanoid": "humanoid", "Optimus": "Optimus",
    "Unitree": "Unitree", "Figure AI": '"Figure AI"', "Boston Dynamics": '"Boston Dynamics"',
    "robotaxi": "robotaxi", "Waymo": "Waymo", "self-driving": "self-driving", "drone": "drone",
    "1X NEO": '"1X" NEO',
}
# Terms whose daily count exceeds Bluesky's 10k hitsTotal cap -> counted in adaptive sub-day windows
BSKY_HIGH_VOLUME = {"AI": "AI", "artificial intelligence": '"artificial intelligence"'}
BSKY_POST_QUERIES = {
    "AI": "AI", "artificial intelligence": '"artificial intelligence"', "LLM": "LLM",
    "AI agent": '"AI agent"', "robot": "robot", "robotics": "robotics", "humanoid": "humanoid",
    "robotaxi": "robotaxi",
}
AI_BRANDS = ["OpenAI", "ChatGPT", "Claude", "Anthropic", "Gemini", "Grok", "DeepSeek", "Nvidia", "Sora"]
ROBOTICS_BRANDS = ["Optimus", "Unitree", "Figure AI", "Boston Dynamics", "Waymo", "1X NEO", "robotaxi"]

HN_QUERIES = [
    "AI", "LLM", "OpenAI", "Anthropic", "Claude", "GPT", "Gemini", "machine learning", "neural",
    "DeepSeek", "Nvidia", "AGI", "Copilot", "Cursor", "Grok", "Llama", "Mistral", "Qwen", "chatbot",
    "agent", "robot", "robots", "robotics", "humanoid", "Waymo", "Tesla", "drone", "self-driving",
    "autonomous", "data center",
]
HN_VOLUME_SERIES = ["AI", "LLM", "OpenAI", "Anthropic", "Claude", "Gemini", "Nvidia", "DeepSeek", "AGI",
                    "Grok", "Qwen", "robot", "robots", "robotics", "humanoid", "Waymo", "self-driving", "drone"]

MASTODON_INSTANCE = os.environ.get("MASTODON_INSTANCE", "https://mastodon.social")
MASTODON_TAGS = ["ai", "artificialintelligence", "llm", "genai", "chatgpt", "openai", "aiart",
                 "robotics", "robot", "robots", "humanoid", "machinelearning"]
MASTODON_MAX_PER_DAY = 120

REDDIT_AI_SUBS = ["artificial", "MachineLearning", "OpenAI", "ChatGPT", "singularity", "LocalLLaMA", "ClaudeAI",
                  "GeminiAI", "Bard", "ArtificialInteligence", "agi", "StableDiffusion", "aiArt"]
REDDIT_ROBOTICS_SUBS = ["robotics", "humanoidrobots", "SelfDrivingCars", "waymo"]
REDDIT_GENERAL_SUBS = ["Futurology", "technology", "teslainvestorsclub"]  # keyword-filtered
REDDIT_SUBS = REDDIT_AI_SUBS + REDDIT_ROBOTICS_SUBS + REDDIT_GENERAL_SUBS
