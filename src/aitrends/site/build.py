"""Render analysis outputs (OUT_DIR) into a static, dependency-free website (SITE_DIR).

Findings are generated automatically from the statistics each week; drop an `out/findings_override.html`
to replace them with a hand-written narrative.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import shutil

import pandas as pd

from .. import config as C

O, SITE = C.OUT_DIR, C.SITE_DIR
esc = html.escape


def _read(name):
    p = O / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _json(name):
    p = O / name
    return json.loads(p.read_text()) if p.exists() else {}


pct = lambda v: f"{v:+.0f}%"  # noqa: E731
f2 = lambda v: f"{v:.2f}"  # noqa: E731
f3 = lambda v: f"{v:.3f}"  # noqa: E731
sh = lambda v: f"{v * 100:.1f}%"  # noqa: E731
num = lambda v: f"{v:,.0f}"  # noqa: E731
pv = lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"  # noqa: E731
pp = lambda v: f"{v:+.2f}"  # noqa: E731


def table(df, cols, fmt=None, max_rows=25, cls=""):
    if df is None or df.empty:
        return "<p class=muted>Not available this week.</p>"
    fmt = fmt or {}
    out = [f"<div class=tw><table class='{cls}'><thead><tr>" + "".join(f"<th>{esc(c[1])}</th>" for c in cols)
           + "</tr></thead><tbody>"]
    for _, r in df.head(max_rows).iterrows():
        cells = []
        for c, _ in cols:
            v = r.get(c, "")
            if c in fmt and pd.notna(v):
                try:
                    v = fmt[c](v)
                except (TypeError, ValueError):
                    pass
            cells.append(f"<td>{esc(str(v)) if pd.notna(v) else ''}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(out) + "</tbody></table></div>"


def findings(V, M, BU, SA, SRB, EA, ER, SDP, ts, w) -> str:
    """Rule-based narrative from the statistics (only FDR-significant effects are called changes)."""
    li = []
    if not V.empty:
        b = V[V.platform == "Bluesky"].set_index("series")
        if "AI" in b.index:
            r = b.loc["AI"]  # a Series: use r["col"], since attributes like .pct_change are Series methods
            verdict = "a significant change" if r["q_fdr"] < .05 else "no significant change"
            li.append(f"<b>Overall AI volume:</b> Bluesky 'AI' averaged {r['cur_daily']:,.0f} posts/day this week vs "
                      f"{r['base_daily']:,.0f}/day in the baseline ({pct(r['pct_change'])}, {verdict}; "
                      f"q={pv(r['q_fdr'])}).")
        sig = V[V.q_fdr < .05]
        up = sig[sig.irr > 1].sort_values("irr", ascending=False).head(6)
        dn = sig[sig.irr < 1].sort_values("irr").head(6)
        if len(up):
            li.append("<b>Rising (FDR q&lt;0.05):</b> " + ", ".join(
                f"{esc(r.series)} <span class=up>{pct(r.pct_change)}</span> <span class=tag>{esc(r.platform)}</span>"
                for r in up.itertuples()) + ".")
        if len(dn):
            li.append("<b>Falling (FDR q&lt;0.05):</b> " + ", ".join(
                f"{esc(r.series)} <span class=down>{pct(r.pct_change)}</span> <span class=tag>{esc(r.platform)}</span>"
                for r in dn.itertuples()) + ".")
    if not BU.empty:
        cb = (BU[BU.day >= w["current_start"]].rename(columns={"count": "n"})  # .count clashes with tuple.count
              .sort_values("robust_z", ascending=False).drop_duplicates("series").head(6))
        if len(cb):
            li.append("<b>Spikes this week:</b> " + ", ".join(
                f"{esc(r.series)} on {r.day} ({r.n:,.0f} vs ~{r.baseline:,.0f} typical)" for r in cb.itertuples())
                      + ".")
    for T, name in [(SA, "AI brands"), (SRB, "robotics brands")]:
        if not T.empty:
            T = T.sort_values("share_change_pp")
            g, l_ = T.iloc[-1], T.iloc[0]
            li.append(f"<b>Share of voice, {name}:</b> biggest gain {esc(g.entity)} ({pp(g.share_change_pp)} pp, now "
                      f"{sh(g.cur_share)}); biggest loss {esc(l_.entity)} ({pp(l_.share_change_pp)} pp).")
    if not M.empty:
        r = M[(M.q_fdr < .05) & (M.cmh_or > 1)].head(4)
        if len(r):
            li.append("<b>Fastest-growing topics:</b> " + "; ".join(
                f"“{esc(x.label)}” (OR {x.cmh_or:.2f}, {pp(x.share_change_pp)} pp)" for x in r.itertuples()) + ".")
        f = M[(M.q_fdr < .05) & (M.cmh_or < 1)].sort_values("cmh_or").head(3)
        if len(f):
            li.append("<b>Cooling topics:</b> " + "; ".join(
                f"“{esc(x.label)}” (OR {x.cmh_or:.2f})" for x in f.itertuples()) + ".")
    for E_, dom in [(EA, "AI"), (ER, "Robotics")]:
        if not E_.empty:
            li.append(f"<b>New vocabulary ({dom}):</b> " + ", ".join(esc(t) for t in E_.term.head(10)) + ".")
    if not SDP.empty:
        for dom, g in SDP.groupby("domain"):
            if dom == "AI+Robotics":
                continue
            cur = (g.cur_mean * g.n_cur).sum() / g.n_cur.sum()
            base = (g.base_mean * g.n_base).sum() / g.n_base.sum()
            neg = (g.pct_neg_cur * g.n_cur).sum() / g.n_cur.sum()
            pos = (g.pct_pos_cur * g.n_cur).sum() / g.n_cur.sum()
            li.append(f"<b>Sentiment, {dom}:</b> net {cur:+.3f} (baseline {base:+.3f}); {neg:.0%} negative vs "
                      f"{pos:.0%} positive posts this week.")
    e = ts.get("engagement", {})
    if e:
        gi = e.get("grouped_importance", {})
        top = ", ".join(list(gi)[:3])
        li.append(f"<b>Engagement:</b> a gradient-boosted model predicts top-10% posts with CV AUC {e['cv_auc']:.2f} "
                  f"(vs {e['cv_auc_without_content']:.2f} without topic/sentiment); strongest drivers: {esc(top)}.")
    return "<ul class=find>" + "".join(f"<li>{x}</li>" for x in li) + "</ul>"


CSS = """
:root{--bg:#0f1220;--card:#171b2e;--fg:#e8eaf3;--muted:#9aa3c0;--acc:#7c9cff;--up:#3ecf8e;--down:#ff6b6b;--bd:#2a3050}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header,nav,main,footer{max-width:1180px;margin:auto;padding-left:24px;padding-right:24px}header{padding-top:36px;padding-bottom:16px}
h1{margin:0 0 6px;font-size:28px}h2{margin:0 0 12px;font-size:20px}h3{font-size:16px;margin:18px 0 8px}.muted{color:var(--muted)}
nav{display:flex;gap:14px;flex-wrap:wrap}nav a{color:var(--acc);text-decoration:none;font-size:14px}main{padding-top:12px;padding-bottom:60px}
section{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:22px;margin:18px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.kpi{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px}
.kpi b{display:block;font-size:24px}.kpi span{color:var(--muted);font-size:13px}img{max-width:100%;border-radius:10px;background:#fff;margin:8px 0}
.tw{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}th,td{padding:6px 8px;border-bottom:1px solid var(--bd);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:860px){.grid2{grid-template-columns:1fr}}
.find li{margin:7px 0}.tag{display:inline-block;padding:0 7px;border-radius:9px;background:#252b48;color:var(--muted);font-size:11px}
.up{color:var(--up);font-weight:600}.down{color:var(--down);font-weight:600}a{color:var(--acc)}footer{color:var(--muted);font-size:13px;padding-bottom:40px}.small{font-size:12px}
"""


def build() -> None:
    (SITE / "fig").mkdir(parents=True, exist_ok=True)
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    for f in (O / "fig").glob("*.png"):
        shutil.copy(f, SITE / "fig" / f.name)
    for f in O.iterdir():
        if f.suffix in (".csv", ".json", ".gz"):
            shutil.copy(f, SITE / "data" / f.name)
    hist = C.HISTORY_DIR / "weekly_metrics.csv"
    if hist.exists():
        shutil.copy(hist, SITE / "data" / "weekly_metrics.csv")

    w = C.window_label()
    vs, ts = _json("volume_summary.json"), _json("text_summary.json")
    V, M, PR = _read("volume_rate_ratios.csv"), _read("topic_momentum.csv"), _read("outlook_next7.csv")
    SA, SRB, BU = _read("sov_ai_brands.csv"), _read("sov_robotics_brands.csv"), _read("volume_bursts.csv")
    EA, ER = _read("emerging_terms_AI.csv"), _read("emerging_terms_Robotics.csv")
    SDP, STP = _read("sentiment_by_domain_platform.csv"), _read("sentiment_by_topic.csv")
    TOP, TPC, TB = _read("top_posts_current_week.csv"), _read("topics.csv"), _read("topic_bursts.csv")
    VR = _read("engagement_by_topic.csv")
    override = O / "findings_override.html"
    fnd = override.read_text() if override.exists() else findings(V, M, BU, SA, SRB, EA, ER, SDP, ts, w)

    for D in (V, M):
        if not D.empty:
            D["sig"] = D.q_fdr.map(lambda q: "✔" if q < 0.05 else "")
    Vb = V[V.platform == "Bluesky"].sort_values("irr", ascending=False) if not V.empty else V
    Vo = V[V.platform != "Bluesky"].sort_values("irr", ascending=False) if not V.empty else V
    BUc = BU[BU.day >= w["baseline_start"]] if not BU.empty else BU
    e = ts.get("engagement", {})
    kpis = [("Posts analysed", f"{ts.get('n_docs', 0):,}"), ("Platforms", "4"),
            ("Volume series modelled", str(vs.get("n_series", "–"))), ("Topics discovered", str(ts.get("k", "–"))),
            ("Significant volume shifts", str(vs.get("n_sig", "–"))),
            ("Engagement model AUC", f"{e.get('cv_auc', float('nan')):.2f}")]
    repo = f" · <a href='{esc(C.REPO_URL)}'>source code</a>" if C.REPO_URL else ""
    hist_img = "<img src=fig/history.png alt='week-over-week history'>" if (O / "fig/history.png").exists() else \
        "<p class=muted>The week-over-week chart appears after the second weekly run.</p>"

    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<meta name=description content="Weekly statistical & machine-learning analysis of AI and robotics conversation on social media.">
<title>AI &amp; Robotics Social-Media Trends — week of {w['current_start']}</title><style>{CSS}</style></head><body>
<header><h1>AI &amp; Robotics on Social Media — Weekly Trends</h1>
<div class=muted>Current week <b>{w['current_start']} → {w['current_end']}</b> vs. 4-week baseline
<b>{w['baseline_start']} → {w['baseline_end']}</b> (UTC, complete days). Sources: Bluesky, Reddit, Hacker News, Mastodon.
Updated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC{repo}.</div></header>
<nav><a href=#findings>Key findings</a><a href=#volume>Volume</a><a href=#sov>Share of voice</a><a href=#topics>Topics</a>
<a href=#terms>Emerging terms</a><a href=#sentiment>Sentiment</a><a href=#engagement>Engagement</a><a href=#outlook>Outlook</a>
<a href=#history>History</a><a href=#method>Methods</a><a href=#data>Data</a></nav>
<main><div class=kpis>{''.join(f"<div class=kpi><b>{v}</b><span>{k}</span></div>" for k, v in kpis)}</div>

<section id=findings><h2>Key findings this week</h2>{fnd}
<p class="muted small">Generated automatically from the statistics below; only effects that survive
Benjamini–Hochberg FDR correction are described as rising/falling.</p></section>

<section id=volume><h2>1 · Conversation volume</h2>
<p class=muted>Negative-binomial GLM per series, <code>daily count ~ current_week + day_of_week</code>; the rate ratio
(IRR) compares the current week's daily rate with the baseline. {vs.get('n_sig', '?')} of {vs.get('n_series', '?')}
series are significant at FDR q&lt;0.05.</p>
<img src=fig/volume_bluesky.png alt="Bluesky volume"><img src=fig/volume_narratives.png alt="Narrative keywords">
<div class=grid2><div><h3>Bluesky keywords</h3>
{table(Vb, [("series", "Keyword"), ("base_daily", "Baseline/day"), ("cur_daily", "This week/day"),
            ("pct_change", "Δ rate"), ("lo", "IRR lo"), ("hi", "IRR hi"), ("q_fdr", "q"), ("sig", "sig")],
       {"base_daily": num, "cur_daily": num, "pct_change": pct, "lo": f2, "hi": f2, "q_fdr": pv}, 40)}</div>
<div><img src=fig/irr_bluesky.png alt="Rate ratios"></div></div>
<h3>Hacker News comments · Reddit posts · Mastodon tags</h3>
{table(Vo, [("platform", "Platform"), ("series", "Series"), ("base_daily", "Baseline/day"), ("cur_daily", "This week/day"),
            ("pct_change", "Δ rate"), ("q_fdr", "q"), ("sig", "sig")],
       {"base_daily": num, "cur_daily": num, "pct_change": pct, "q_fdr": pv}, 60)}
<h3>Detected spikes (robust z ≥ 3.5 vs trailing 14-day median)</h3>
{table(BUc.sort_values("day", ascending=False) if not BUc.empty else BUc,
       [("day", "Day"), ("platform", "Platform"), ("series", "Series"), ("count", "Count"), ("baseline", "Typical"),
        ("robust_z", "z")], {"count": num, "baseline": num, "robust_z": f2}, 30)}</section>

<section id=sov><h2>2 · Share of voice (Bluesky)</h2>
<p class=muted>χ² homogeneity — AI brands: χ²={vs.get('sov_ai_test', {}).get('chi2', 0):,.0f}, Cramér's V=
{vs.get('sov_ai_test', {}).get('cramers_v', 0):.3f}; robotics: χ²={vs.get('sov_robotics_test', {}).get('chi2', 0):,.0f},
V={vs.get('sov_robotics_test', {}).get('cramers_v', 0):.3f}. |adjusted residual| &gt; 2 ⇒ brand over/under-represented this week.</p>
<div class=grid2><div><img src=fig/sov_ai_brands.png alt="AI share of voice">
{table(SA.sort_values("share_change_pp", ascending=False) if not SA.empty else SA,
       [("entity", "Brand"), ("base_share", "Baseline"), ("cur_share", "This week"), ("share_change_pp", "Δ pp"),
        ("adj_resid", "adj. resid")], {"base_share": sh, "cur_share": sh, "share_change_pp": pp, "adj_resid": f2})}</div>
<div><img src=fig/sov_robotics_brands.png alt="Robotics share of voice">
{table(SRB.sort_values("share_change_pp", ascending=False) if not SRB.empty else SRB,
       [("entity", "Brand"), ("base_share", "Baseline"), ("cur_share", "This week"), ("share_change_pp", "Δ pp"),
        ("adj_resid", "adj. resid")], {"base_share": sh, "cur_share": sh, "share_change_pp": pp, "adj_resid": f2})}</div></div>
</section>

<section id=topics><h2>3 · Topics &amp; momentum</h2>
<p class=muted>{ts.get('k', '?')} topics: MiniLM sentence embeddings → k-means (k by cosine silhouette) → class-based TF-IDF
labels. Independent TF-IDF/NMF model agreement: NMI={ts.get('nmf_agreement', {}).get('NMI', 0):.2f}. Momentum is the
Cochran–Mantel–Haenszel odds ratio (this week vs baseline) stratified by platform; shares are platform-balanced.</p>
<div class=grid2><img src=fig/topic_momentum.png alt="Topic momentum"><img src=fig/topic_map.png alt="Topic map"></div>
{table(M, [("label", "Topic"), ("base_share", "Baseline"), ("cur_share", "This week"), ("share_change_pp", "Δ pp"),
           ("cmh_or", "OR"), ("q_fdr", "q"), ("sig", "sig"), ("cur_docs", "Posts this week")],
       {"base_share": sh, "cur_share": sh, "share_change_pp": pp, "cmh_or": f2, "q_fdr": pv, "cur_docs": num}, 50)}
<h3>Topic spikes (daily share, robust z &gt; 4)</h3>
{table(TB, [("day", "Day"), ("label", "Topic"), ("docs", "Posts"), ("share", "Share"), ("baseline", "Typical"),
            ("robust_z", "z")], {"share": sh, "baseline": sh, "robust_z": f2, "docs": num}, 20)}
<h3>Topic dictionary</h3>
{table(TPC, [("label", "Topic"), ("size", "Posts"), ("terms", "Top c-TF-IDF terms")], {"size": num}, 60, "small")}</section>

<section id=terms><h2>4 · Emerging language</h2>
<p class=muted>Weighted log-odds ratio with informative Dirichlet prior (Monroe, Colaresi &amp; Quinn 2008), this week vs
baseline, platform-balanced document frequencies; z &gt; 1.96 ≈ significant.</p><img src=fig/emerging_terms.png alt="Emerging terms">
<div class=grid2><div><h3>AI</h3>{table(EA, [("term", "Term"), ("z", "z"), ("cur_df", "This week"), ("base_df", "Baseline")],
                                       {"z": f2, "cur_df": num, "base_df": num}, 30)}</div>
<div><h3>Robotics</h3>{table(ER, [("term", "Term"), ("z", "z"), ("cur_df", "This week"), ("base_df", "Baseline")],
                              {"z": f2, "cur_df": num, "base_df": num}, 30)}</div></div></section>

<section id=sentiment><h2>5 · Sentiment</h2>
<p class=muted>Twitter-RoBERTa (3-class) → net score P(pos)−P(neg). Lexicon check (VADER): Spearman ρ=
{ts.get('sentiment_agreement', {}).get('spearman', 0):.2f}, Cohen's κ={ts.get('sentiment_agreement', {}).get('cohen_kappa', 0):.2f}.
95% bootstrap CIs; Mann–Whitney U with FDR.</p>
<div class=grid2><img src=fig/sentiment_daily.png alt="Daily sentiment"><img src=fig/sentiment_topics.png alt="Sentiment by topic"></div>
{table(SDP, [("domain", "Domain"), ("platform", "Platform"), ("n_cur", "Posts"), ("base_mean", "Baseline net"),
             ("cur_mean", "This week net"), ("delta", "Δ"), ("pct_neg_cur", "% neg"), ("pct_pos_cur", "% pos"), ("q_fdr", "q")],
       {"base_mean": f3, "cur_mean": f3, "delta": lambda v: f"{v:+.3f}", "pct_neg_cur": sh, "pct_pos_cur": sh,
        "q_fdr": pv, "n_cur": num}, 20)}
<h3>Sentiment by topic</h3>
{table(STP.sort_values("delta") if not STP.empty else STP,
       [("label", "Topic"), ("n", "Posts"), ("mean", "Net (window)"), ("base_mean", "Baseline"), ("cur_mean", "This week"),
        ("delta", "Δ"), ("q_fdr", "q")],
       {"n": num, "mean": f3, "base_mean": f3, "cur_mean": f3, "delta": lambda v: f"{v:+.3f}", "q_fdr": pv}, 50)}</section>

<section id=engagement><h2>6 · What drives engagement</h2>
<p class=muted>Target: post in the top 10% of engagement within its own feed. Gradient-boosted trees, 5-fold CV ROC-AUC
{e.get('cv_auc', 0):.3f} (vs {e.get('cv_auc_without_content', 0):.3f} without topic/sentiment features). Grouped permutation
importance: {esc(', '.join(f"{k} {v:.3f}" for k, v in e.get('grouped_importance', {}).items()))}.</p>
{table(VR, [("topic_label", "Topic"), ("top_decile_rate", "Share of posts in top 10%")], {"top_decile_rate": sh}, 40)}
<h3>Most-engaged posts this week (per topic)</h3>
{table(TOP, [("topic_label", "Topic"), ("platform", "Platform"), ("source", "Feed"), ("eng", "Engagement"), ("text", "Text")],
       {"eng": num}, 80, "small")}</section>

<section id=outlook><h2>7 · Next-7-day outlook</h2>
<p class=muted>Run-rate projection: NB GLM with day-of-week effects on the last 28 days, 4,000 simulations (parameter +
count noise). It assumes no new shock — spikes are, by definition, not forecastable.</p>
{table(PR, [("series", "Series"), ("current_week", "This week"), ("next7_median", "Next 7 days (median)"),
            ("pi80_lo", "80% PI lo"), ("pi80_hi", "80% PI hi")],
       {"current_week": num, "next7_median": num, "pi80_lo": num, "pi80_hi": num})}</section>

<section id=history><h2>8 · Week-over-week history</h2>{hist_img}
<p class=muted>Every weekly run appends its headline metrics to <a href=data/weekly_metrics.csv>weekly_metrics.csv</a>.</p></section>

<section id=method><h2>Methods &amp; caveats</h2>
{f"<p>Full explanation of every technique and metric, why it was chosen and how far to trust it: <a href='{esc(C.REPO_URL)}/blob/main/docs/methodology.md'>methodology.md</a>.</p>" if C.REPO_URL else ""}<ul class=find>
<li><b>Collection.</b> Bluesky public AppView search (English; daily <code>hitsTotal</code> per keyword, adaptive sub-day
windows above the 10k cap; top-200 posts/day for 8 queries). Hacker News via Algolia (story titles + comment counts per
keyword-day). Reddit via the Arctic Shift archive (full post census of {len(C.REDDIT_SUBS)} subreddits). Mastodon hashtag
timelines from mastodon.social (≤{C.MASTODON_MAX_PER_DAY} posts per tag-day, rates extrapolated when truncated).</li>
<li><b>Corpus.</b> Regex relevance filter (AI / Robotics / both), exact-duplicate removal, complete UTC days only.</li>
<li><b>Coverage gaps.</b> X/Twitter, TikTok, Instagram, LinkedIn and YouTube are not covered (no keyless API access).
Bluesky post samples are “top”-sorted, so text analyses over-weight popular posts. Reddit scores are archived soon after
posting and understate final engagement.</li>
<li><b>Statistics.</b> All multi-series tests are FDR-controlled (Benjamini–Hochberg). With ~100k posts, tiny effects can be
significant — read effect sizes, not only p-values.</li></ul></section>

<section id=data><h2>Data downloads</h2><p>{' · '.join(f"<a href='data/{f.name}'>{f.name}</a>"
                                                    for f in sorted((SITE / 'data').iterdir()))}</p></section>
</main><footer>Python · pandas · statsmodels · scikit-learn · sentence-transformers · transformers · UMAP — rebuilt weekly by
GitHub Actions{repo}.</footer></body></html>"""
    (SITE / "index.html").write_text(page)
    (SITE / "healthz").write_text("ok\n")
    (SITE / ".nojekyll").write_text("")
    print(f"site: written to {SITE}", flush=True)
