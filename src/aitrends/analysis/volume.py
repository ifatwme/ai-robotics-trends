"""Attention / volume statistics.

Per daily series (keyword on Bluesky, query on HN comments, subreddit, Mastodon tag):
- Negative-binomial GLM  y ~ current_week + C(day_of_week)  -> incidence-rate ratio (IRR) + 95% CI
  (falls back to quasi-Poisson if NB fails), Benjamini-Hochberg FDR across series
- Robust burst detection: robust z vs trailing-14-day median/MAD
- Change points: binary segmentation on log counts with a BIC-style penalty
- Share of voice: chi-square homogeneity, Cramer's V, adjusted standardized residuals
- Next-7-day outlook: NB GLM on the last 28 days + parameter/count-noise simulation
"""
from __future__ import annotations

import json
import warnings

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .. import config as cfg  # not "C": patsy formulas use C() for categoricals

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
O = cfg.OUT_DIR


def load_series() -> dict[tuple[str, str], pd.DataFrame]:
    days = [d.date().isoformat() for d in cfg.days()]
    series: dict[tuple[str, str], pd.DataFrame] = {}

    def add(key, counts: pd.Series):
        counts = counts.reindex(days)
        series[key] = pd.DataFrame({"day": days, "y": counts.values})

    p = cfg.DATA_DIR / "bsky_counts.json"
    if p.exists():
        bc = pd.DataFrame(json.loads(p.read_text()))
        for kw, g in bc.groupby("kw"):
            add(("Bluesky", kw), g.set_index("day").hits)
    p = cfg.DATA_DIR / "hn.json"
    if p.exists():
        hn = pd.DataFrame(json.loads(p.read_text()))
        for q in cfg.HN_VOLUME_SERIES:
            g = hn[hn.q == q]
            if len(g):
                add(("HN comments", q), g.set_index("day").comment_hits)
    p = cfg.DATA_DIR / "reddit.json"
    if p.exists():
        rd = pd.read_json(p)
        if len(rd):
            rd["day"] = pd.to_datetime(rd.created_utc, unit="s", utc=True).dt.strftime("%Y-%m-%d")
            for s, g in rd.groupby("subreddit"):
                add(("Reddit posts", "r/" + s), g.groupby("day").size().reindex(days, fill_value=0))
    p = cfg.DATA_DIR / "masto.json"
    if p.exists():
        ma = pd.read_json(p)
        fp = cfg.DATA_DIR / "masto_failed.json"
        failed_days: dict[str, set] = {}
        for f in (json.loads(fp.read_text()) if fp.exists() else []):
            failed_days.setdefault(f["tag"], set()).add(f["day"])
        if "bot" in ma and ma.bot.notna().all():  # only with complete bot info (avoids period bias)
            ma = ma[~ma.bot.fillna(False).astype(bool)]
        if len(ma):
            ma["created"] = pd.to_datetime(ma.created, utc=True)
            ma["day"] = ma.created.dt.strftime("%Y-%m-%d")
            for tag, g in ma.groupby("tag"):
                est = {}
                for day, h in g.groupby("day"):
                    n = len(h)
                    if n >= cfg.MASTODON_MAX_PER_DAY:  # truncated sample -> extrapolate from observed rate
                        span = (h.created.max() - h.created.min()).total_seconds() / 3600
                        n = round(n / max(span, 0.25) * 24)
                    est[day] = n
                ser = pd.Series(est).reindex(days, fill_value=0).astype(float)
                ser[[d for d in failed_days.get(tag, ()) if d in ser.index]] = np.nan  # missing, not zero
                add(("Mastodon #", tag), ser)

    _, cur, _ = cfg.window()
    for g in series.values():
        g["date"] = pd.to_datetime(g.day)
        g["dow"] = g.date.dt.dayofweek
        g["cur"] = (g.date >= pd.Timestamp(cur.date())).astype(int)
    return series


def nb_irr(g: pd.DataFrame):
    g = g.dropna(subset=["y"]).copy()
    g["y"] = g.y.astype(float)
    if g.y.sum() < 30 or g.cur.nunique() < 2:
        return None
    try:
        m = smf.negativebinomial("y ~ cur + C(dow)", data=g).fit(disp=0, maxiter=200)
        b, se, p, model = m.params["cur"], m.bse["cur"], m.pvalues["cur"], "NB2"
        if not np.isfinite(se):
            raise ValueError
    except Exception:
        m = smf.glm("y ~ cur + C(dow)", data=g, family=sm.families.Poisson()).fit(scale="X2")
        b, se, p, model = m.params["cur"], m.bse["cur"], m.pvalues["cur"], "quasi-Poisson"
    return dict(base_daily=g[g.cur == 0].y.mean(), cur_daily=g[g.cur == 1].y.mean(), irr=np.exp(b),
                lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=p, model=model)


def binseg(x: np.ndarray, max_cp: int = 3) -> list[int]:
    n = len(x)
    pen = 2 * np.log(n) * max(np.var(x), 1e-9)

    def cost(a, b):
        s = x[a:b]
        return ((s - s.mean()) ** 2).sum()

    cps, segs = [], [(0, n)]
    for _ in range(max_cp):
        best = None
        for a, b in segs:
            if b - a < 6:
                continue
            base = cost(a, b)
            for t in range(a + 3, b - 3):
                gain = base - cost(a, t) - cost(t, b)
                if best is None or gain > best[0]:
                    best = (gain, t, a, b)
        if best is None or best[0] < pen:
            break
        _, t, a, b = best
        cps.append(t)
        segs.remove((a, b))
        segs += [(a, t), (t, b)]
    return sorted(cps)


def share_of_voice(series, names, label):
    rows = []
    for n in names:
        g = series.get(("Bluesky", n))
        if g is None:
            continue
        rows.append([n, g[g.cur == 0].y.sum(), g[g.cur == 1].y.sum()])
    T = pd.DataFrame(rows, columns=["entity", "baseline", "current"]).set_index("entity").astype(float)
    T = T[(T.sum(1) > 0)]
    chi2, p, dof, exp = stats.chi2_contingency(T.values)
    n = T.values.sum()
    cv = np.sqrt(chi2 / (n * (min(T.shape) - 1)))
    rs, cs = T.values.sum(1, keepdims=True) / n, T.values.sum(0, keepdims=True) / n
    resid = (T.values - exp) / np.sqrt(exp * (1 - rs) * (1 - cs))
    T["base_share"] = T.baseline / T.baseline.sum()
    T["cur_share"] = T.current / T.current.sum()
    T["share_change_pp"] = (T.cur_share - T.base_share) * 100
    T["adj_resid"] = resid[:, 1]
    T.to_csv(O / f"sov_{label}.csv")
    return T, dict(chi2=float(chi2), p=float(p), dof=int(dof), cramers_v=float(cv))


def outlook(g: pd.DataFrame, sims: int = 4000, seed: int = cfg.SEED):
    """Next-7-day projection from a day-of-week NB model on the last 28 days (no trend extrapolation)."""
    h = g.dropna(subset=["y"]).tail(28).copy()
    h["y"] = h.y.astype(float)
    m = smf.negativebinomial("y ~ C(dow)", data=h).fit(disp=0, maxiter=300)
    alpha = m.params["alpha"]
    _, _, end = cfg.window()
    fut = pd.DataFrame({"date": pd.date_range(pd.Timestamp(end.date()), periods=7)})
    X = pd.get_dummies(fut.date.dt.dayofweek.astype("category").cat.set_categories(range(7)),
                       prefix="C(dow)", drop_first=True).astype(float)
    X.columns = [f"C(dow)[T.{c.split('_')[-1]}]" for c in X.columns]
    X.insert(0, "Intercept", 1.0)
    params = m.params.drop("alpha")
    X = X[params.index]
    rng = np.random.default_rng(seed)
    beta = rng.multivariate_normal(params.values, m.cov_params().loc[params.index, params.index].values, sims)
    mu = np.exp(beta @ X.values.T)
    y = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu)).sum(1)
    cur_week = g[g.cur == 1].y.sum()
    return dict(current_week=float(cur_week), next7_median=float(np.median(y)),
                pi80_lo=float(np.percentile(y, 10)), pi80_hi=float(np.percentile(y, 90)))


def run() -> dict:
    series = load_series()
    res = [dict(platform=k[0], series=k[1], **r) for k, g in series.items() if (r := nb_irr(g))]
    V = pd.DataFrame(res)
    V["q_fdr"] = multipletests(V.p, method="fdr_bh")[1]
    V["pct_change"] = (V.irr - 1) * 100
    V = V.sort_values(["platform", "irr"], ascending=[True, False])
    V.to_csv(O / "volume_rate_ratios.csv", index=False)
    print(f"volume: {len(V)} series, {(V.q_fdr < .05).sum()} significant", flush=True)

    bursts = []
    for (plat, name), g in series.items():
        y = g.set_index("day").y.astype(float)
        if y.sum() < 100:
            continue
        for i in range(14, len(y)):
            base = y.iloc[i - 14:i].dropna()
            if len(base) < 10 or np.isnan(y.iloc[i]):
                continue
            med = base.median()
            mad = max((base - med).abs().median() * 1.4826, np.sqrt(max(med, 1)))
            z = (y.iloc[i] - med) / mad
            if z >= 3.5 and y.iloc[i] >= 20:
                bursts.append(dict(platform=plat, series=name, day=y.index[i], count=y.iloc[i], baseline=med,
                                   robust_z=z))
    Bu = pd.DataFrame(bursts, columns=["platform", "series", "day", "count", "baseline", "robust_z"])
    Bu.sort_values("robust_z", ascending=False).to_csv(O / "volume_bursts.csv", index=False)

    cps = []
    for key in [("Bluesky", k) for k in ["AI", "robot", "humanoid", "AI bubble", "Claude", "OpenAI", "Gemini",
                                         "Nvidia", "data center", "AI slop", "robotaxi"]] + [("HN comments", "AI"),
                                                                                              ("HN comments", "Claude")]:
        g = series.get(key)
        if g is None or g.y.isna().any():
            continue
        x = np.log1p(g.y.values.astype(float))
        for t in binseg(x):
            cps.append(dict(platform=key[0], series=key[1], change_day=g.day.iloc[t],
                            mean_before=float(np.expm1(x[:t].mean())), mean_after=float(np.expm1(x[t:].mean()))))
    pd.DataFrame(cps).to_csv(O / "volume_changepoints.csv", index=False)

    sov_ai, t_ai = share_of_voice(series, cfg.AI_BRANDS, "ai_brands")
    sov_rb, t_rb = share_of_voice(series, cfg.ROBOTICS_BRANDS, "robotics_brands")

    pr = []
    for key in [("Bluesky", "AI"), ("Bluesky", "robot"), ("Bluesky", "humanoid"), ("Bluesky", "Claude"),
                ("Bluesky", "OpenAI"), ("Bluesky", "Nvidia"), ("Bluesky", "AI bubble"), ("HN comments", "AI")]:
        if key in series:
            try:
                pr.append(dict(series=f"{key[0]}: {key[1]}", **outlook(series[key])))
            except Exception as e:  # noqa: BLE001
                print("outlook failed", key, e)
    pd.DataFrame(pr).to_csv(O / "outlook_next7.csv", index=False)

    # ---------------- figures
    _, cur, _ = cfg.window()
    curx = pd.Timestamp(cur.date())
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, keys, title in [(axes[0], ["AI", "artificial intelligence", "LLM", "Claude", "OpenAI", "Nvidia"],
                             "Bluesky — daily English posts, AI terms"),
                            (axes[1], ["robot", "humanoid", "robotics", "Waymo", "robotaxi"],
                             "Bluesky — daily English posts, robotics terms")]:
        for k in keys:
            if ("Bluesky", k) in series:
                g = series[("Bluesky", k)]
                ax.plot(g.date, g.y, marker=".", label=k)
        ax.axvspan(curx, g.date.max() + pd.Timedelta(hours=12), color="tab:orange", alpha=.08)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=6)
        ax.set_yscale("log")
        ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(O / "fig/volume_bluesky.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 4))
    for k in ["AI bubble", "data center", "AI slop", "AI agent", "vibe coding", "deepfake", "AI regulation"]:
        if ("Bluesky", k) in series:
            g = series[("Bluesky", k)]
            ax.plot(g.date, g.y, marker=".", label=k)
    ax.axvspan(curx, g.date.max() + pd.Timedelta(hours=12), color="tab:orange", alpha=.08)
    ax.legend(ncol=7, fontsize=7)
    ax.grid(alpha=.3)
    ax.set_title("Bluesky — narrative keywords (shaded = current week)")
    plt.tight_layout()
    plt.savefig(O / "fig/volume_narratives.png")
    plt.close()

    sub = V[V.platform == "Bluesky"].sort_values("irr")
    fig, ax = plt.subplots(figsize=(7, 8))
    col = ["tab:green" if (q < .05 and i > 1) else "tab:red" if q < .05 else "grey" for q, i in zip(sub.q_fdr, sub.irr)]
    ax.errorbar(sub.irr, range(len(sub)), xerr=[sub.irr - sub.lo, sub.hi - sub.irr], fmt="none", ecolor="lightgrey")
    ax.scatter(sub.irr, range(len(sub)), c=col, zorder=3)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub.series)
    ax.axvline(1, color="k", lw=.8)
    ax.set_xscale("log")
    ax.set_title("Bluesky: current week vs 4-week baseline\n(daily-rate ratio, NB GLM, 95% CI; colour = FDR q<0.05)")
    plt.tight_layout()
    plt.savefig(O / "fig/irr_bluesky.png")
    plt.close()

    for T, lab in [(sov_ai, "ai_brands"), (sov_rb, "robotics_brands")]:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        x = np.arange(len(T))
        ax.bar(x - .2, T.base_share * 100, .4, label="4-week baseline")
        ax.bar(x + .2, T.cur_share * 100, .4, label="current week")
        ax.set_xticks(x)
        ax.set_xticklabels(T.index, rotation=30, ha="right")
        ax.set_ylabel("% share of voice")
        ax.legend()
        ax.set_title(f"Bluesky share of voice — {lab.replace('_', ' ')}")
        plt.tight_layout()
        plt.savefig(O / f"fig/sov_{lab}.png")
        plt.close()

    summary = dict(sov_ai_test=t_ai, sov_robotics_test=t_rb, n_series=int(len(V)), n_sig=int((V.q_fdr < .05).sum()))
    (O / "volume_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
