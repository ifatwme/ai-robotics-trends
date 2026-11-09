"""Append this run's headline metrics to history/weekly_metrics.csv (committed to git) and chart them."""
from __future__ import annotations

import json

import matplotlib
import pandas as pd

from . import config as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

O = C.OUT_DIR


def _read(name):
    p = O / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def run() -> pd.DataFrame:
    w = C.window_label()
    row: dict = {"week_start": w["current_start"], "week_end": w["current_end"]}
    V = _read("volume_rate_ratios.csv")
    if not V.empty:
        b = V[V.platform == "Bluesky"].set_index("series")
        for k in ["AI", "LLM", "Claude", "OpenAI", "ChatGPT", "Gemini", "Nvidia", "AI bubble", "data center",
                  "robot", "humanoid", "robotics", "Waymo", "robotaxi", "Optimus"]:
            if k in b.index:
                row[f"bsky_daily_{k}"] = round(float(b.loc[k, "cur_daily"]), 1)
    for f, pref in [("sov_ai_brands.csv", "sov_ai"), ("sov_robotics_brands.csv", "sov_rb")]:
        S = _read(f)
        for _, r in S.iterrows():
            row[f"{pref}_{r.entity}"] = round(float(r.cur_share) * 100, 2)
    SR = _read("sentiment_by_domain_platform.csv")
    if not SR.empty:
        for dom, g in SR.groupby("domain"):
            row[f"net_sentiment_{dom}"] = round(float((g.cur_mean * g.n_cur).sum() / g.n_cur.sum()), 4)
    ts = json.loads((O / "text_summary.json").read_text()) if (O / "text_summary.json").exists() else {}
    row["docs_total"] = ts.get("n_docs")
    M = _read("topic_momentum.csv")
    if not M.empty:
        row["top_rising_topic"] = M.iloc[0].label
    H_path = C.HISTORY_DIR / "weekly_metrics.csv"
    H = pd.read_csv(H_path) if H_path.exists() else pd.DataFrame()
    if not H.empty:
        H = H[H.week_end != row["week_end"]]
    H = pd.concat([H, pd.DataFrame([row])], ignore_index=True).sort_values("week_end")
    H.to_csv(H_path, index=False)

    if len(H) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        for ax, cols, title in [(axes[0], ["bsky_daily_AI"], "Bluesky 'AI' posts/day (weekly mean)"),
                                (axes[1], ["bsky_daily_robot", "bsky_daily_humanoid", "bsky_daily_Waymo"],
                                 "Bluesky robotics posts/day (weekly mean)")]:
            for c in cols:
                if c in H:
                    ax.plot(pd.to_datetime(H.week_end), H[c], marker="o", label=c.replace("bsky_daily_", ""))
            ax.set_title(title)
            ax.legend(fontsize=7)
            ax.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(O / "fig/history.png", dpi=130)
        plt.close()
    print(f"history: {len(H)} weeks recorded", flush=True)
    return H
