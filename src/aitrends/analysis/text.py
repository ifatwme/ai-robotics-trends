"""Text & ML analysis on the unified corpus (rolling window: current week vs 4-week baseline).

1. Topics: MiniLM embeddings -> MiniBatchKMeans (k by cosine silhouette) -> class-based TF-IDF labels;
   TF-IDF/NMF as an independent cross-check (NMI, ARI)
2. Topic momentum: Cochran-Mantel-Haenszel odds ratio (current vs baseline) stratified by platform, BH-FDR
3. Emerging terms: weighted log-odds with informative Dirichlet prior (Monroe et al. 2008), platform-balanced
4. Sentiment: Twitter-RoBERTa (primary) vs VADER (agreement), bootstrap CIs, Mann-Whitney U + FDR
5. Engagement: gradient-boosted classifier of top-decile engagement (within feed), CV ROC-AUC,
   grouped permutation importance, logistic-regression odds ratios
6. UMAP 2-D topic map
"""
from __future__ import annotations

import json
import warnings

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

from .. import config as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
O = C.OUT_DIR
RNG = C.SEED
STOP = list(ENGLISH_STOP_WORDS | {
    "http", "https", "www", "com", "amp", "just", "like", "don", "ve", "ll", "re", "didn", "doesn", "isn", "im",
    "it", "s", "t", "got", "get", "really", "know", "think", "people", "thing", "things", "going", "want", "make",
    "way", "use", "using", "used", "does", "did", "say", "said", "good", "new", "time", "need", "lot", "actually",
    "yes", "oh", "gt", "lt", "nbsp", "x200b", "reddit", "bsky", "app", "html", "jpg", "png", "gif", "jpeg", "webp",
    "preview", "redd", "width", "format", "auto", "s3", "imgur"})
TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z0-9\-\.]+\b"


def _load():
    df = pd.read_pickle(C.DATA_DIR / "corpus.pkl")
    E = np.load(C.DATA_DIR / "emb.npy")
    S = pd.read_pickle(C.DATA_DIR / "sentiment.pkl")
    assert len(df) == len(E) == len(S) and (S.id.values == df.id.values).all(), "run enrich first"
    df = pd.concat([df.reset_index(drop=True), S.drop(columns="id")], axis=1)
    return df, E


def _boot_ci(x, n=1000):
    x = np.asarray(x)
    rng = np.random.default_rng(0)
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def run() -> dict:
    df, E = _load()
    print(f"text: {len(df)} docs", flush=True)
    summary: dict = {"n_docs": int(len(df)),
                     "docs_by_platform_period": df.groupby(["platform", "period"]).size().unstack().fillna(0)
                     .astype(int).to_dict()}
    clean = df.text.str.replace(r"https?://\S+|@\S+|#", " ", regex=True).str.lower()

    # ---------------- 1. topics
    rng = np.random.default_rng(RNG)
    samp = rng.choice(len(E), size=min(12000, len(E)), replace=False)
    sil = {}
    for k in range(12, 31, 3):
        km = MiniBatchKMeans(k, random_state=RNG, batch_size=4096, n_init=5).fit(E)
        sil[k] = float(silhouette_score(E[samp], km.labels_[samp], metric="cosine"))
    best_k = max(sil, key=lambda k: (round(sil[k], 3), -k))
    km = MiniBatchKMeans(best_k, random_state=RNG, batch_size=4096, n_init=10).fit(E)
    df["topic"] = km.labels_
    summary.update(silhouette=sil, k=best_k)
    print(f"text: k={best_k} silhouette={sil[best_k]:.3f}", flush=True)

    cv = CountVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=8, max_df=0.4, token_pattern=TOKEN)
    X = cv.fit_transform(clean)
    vocab = np.array(cv.get_feature_names_out())
    tf = np.vstack([np.asarray(X[df.topic.values == t].sum(0)).ravel() for t in range(best_k)])
    A = tf.sum() / best_k
    ctfidf = (tf / np.maximum(tf.sum(1, keepdims=True), 1)) * np.log(1 + A / np.maximum(tf.sum(0), 1))
    cent = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    sim = E @ cent.T
    topics = []
    for t in range(best_k):
        top = vocab[np.argsort(-ctfidf[t])[:15]].tolist()
        idx = np.where(df.topic == t)[0]
        rep = idx[np.argsort(-sim[idx, t])[:5]]
        # label: top 3 c-TF-IDF terms, skipping unigrams already contained in a chosen bigram
        lab = []
        for w in top:
            if any(w in l.split() for l in lab) or any(l in w.split() for l in lab):
                continue
            lab.append(w)
            if len(lab) == 3:
                break
        topics.append(dict(topic=t, label=" · ".join(lab), size=len(idx), terms=", ".join(top),
                           domain_mix=df.loc[idx, "domain"].value_counts(normalize=True).round(2).to_dict(),
                           platform_mix=df.loc[idx, "platform"].value_counts(normalize=True).round(2).to_dict(),
                           examples=" ||| ".join(df.text.iloc[rep].str[:160])))
    T = pd.DataFrame(topics)
    T.to_csv(O / "topics.csv", index=False)
    lab = dict(zip(T.topic, T.label))
    df["topic_label"] = df.topic.map(lab)

    tfv = TfidfVectorizer(stop_words=STOP, min_df=8, max_df=0.4, ngram_range=(1, 2), sublinear_tf=True,
                          token_pattern=TOKEN)
    W = NMF(best_k, init="nndsvda", random_state=RNG, max_iter=300).fit_transform(tfv.fit_transform(clean))
    has = W.max(1) > 0
    summary["nmf_agreement"] = dict(NMI=float(normalized_mutual_info_score(df.topic[has], W.argmax(1)[has])),
                                    ARI=float(adjusted_rand_score(df.topic[has], W.argmax(1)[has])))

    # ---------------- 2. topic momentum
    rows = []
    for t in range(best_k):
        tabs = []
        for _, g in df.groupby("platform"):
            cur, base = g.period == "current", g.period == "baseline"
            a, b = ((g.topic == t) & cur).sum(), ((g.topic != t) & cur).sum()
            c, d = ((g.topic == t) & base).sum(), ((g.topic != t) & base).sum()
            tabs.append(np.array([[a + .5, b + .5], [c + .5, d + .5]]))
        st = StratifiedTable(tabs)
        lo, hi = st.oddsratio_pooled_confint()
        sh = df.assign(is_t=df.topic == t).groupby(["period", "platform"]).is_t.mean().groupby("period").mean()
        rows.append(dict(topic=t, label=lab[t], base_share=sh.get("baseline", 0), cur_share=sh.get("current", 0),
                         cmh_or=st.oddsratio_pooled, lo=lo, hi=hi, p=st.test_null_odds().pvalue,
                         cur_docs=int(((df.topic == t) & (df.period == "current")).sum())))
    M = pd.DataFrame(rows)
    M["q_fdr"] = multipletests(M.p, method="fdr_bh")[1]
    M["share_change_pp"] = (M.cur_share - M.base_share) * 100
    M = M.sort_values("cmh_or", ascending=False)
    M.to_csv(O / "topic_momentum.csv", index=False)

    daily = df.groupby(["day", "topic"]).size().unstack(fill_value=0)
    share = daily.div(daily.sum(1), axis=0)
    tb = []
    for t in share.columns:
        s = share[t]
        for i in range(14, len(s)):
            base = s.iloc[i - 14:i]
            med = base.median()
            mad = max((base - med).abs().median() * 1.4826, 0.003)
            z = (s.iloc[i] - med) / mad
            if z > 4 and daily[t].iloc[i] >= 25:
                tb.append(dict(day=s.index[i], topic=t, label=lab[t], share=s.iloc[i], baseline=med, robust_z=z,
                               docs=int(daily[t].iloc[i])))
    pd.DataFrame(tb, columns=["day", "topic", "label", "share", "baseline", "robust_z", "docs"]) \
        .sort_values("robust_z", ascending=False).to_csv(O / "topic_bursts.csv", index=False)

    # ---------------- 3. emerging terms
    cv2 = CountVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=15, token_pattern=TOKEN, binary=True)
    X2 = cv2.fit_transform(clean)
    v2 = np.array(cv2.get_feature_names_out())
    plats = df.platform.unique()

    def balanced(mask):
        tot = np.zeros(X2.shape[1])
        target = mask.sum() / len(plats)
        for p in plats:
            m = mask & (df.platform == p).values
            if m.sum():
                tot += np.asarray(X2[m].sum(0)).ravel() * (target / m.sum())
        return tot

    for dom, dm in [("AI", df.domain.isin(["AI", "AI+Robotics"]).values),
                    ("Robotics", df.domain.isin(["Robotics", "AI+Robotics"]).values)]:
        ya, yb = balanced(dm & (df.period == "current").values), balanced(dm & (df.period == "baseline").values)
        a0 = np.maximum(ya + yb, 0.01)
        A0, na, nb = a0.sum(), ya.sum(), yb.sum()
        delta = np.log((ya + a0) / (na + A0 - ya - a0)) - np.log((yb + a0) / (nb + A0 - yb - a0))
        L = pd.DataFrame({"term": v2, "z": delta / np.sqrt(1 / (ya + a0) + 1 / (yb + a0)), "cur_df": ya, "base_df": yb})
        L = L[L.cur_df >= 8].sort_values("z", ascending=False)
        L.head(60).to_csv(O / f"emerging_terms_{dom}.csv", index=False)
        L.tail(30).iloc[::-1].to_csv(O / f"fading_terms_{dom}.csv", index=False)

    # ---------------- 4. sentiment
    vl = np.where(df.vader > 0.05, 2, np.where(df.vader < -0.05, 0, 1))
    summary["sentiment_agreement"] = dict(spearman=float(stats.spearmanr(df.vader, df.rob_score)[0]),
                                          cohen_kappa=float(cohen_kappa_score(vl, df.rob_label)),
                                          raw_agreement=float((vl == df.rob_label).mean()))
    sr = []
    for (dom, plat), g in df.groupby(["domain", "platform"]):
        o, n = g[g.period == "baseline"].rob_score, g[g.period == "current"].rob_score
        if len(n) < 30 or len(o) < 30:
            continue
        (lo_o, hi_o), (lo_n, hi_n) = _boot_ci(o), _boot_ci(n)
        cg = g[g.period == "current"]
        sr.append(dict(domain=dom, platform=plat, n_base=len(o), n_cur=len(n), base_mean=o.mean(), base_lo=lo_o,
                       base_hi=hi_o, cur_mean=n.mean(), cur_lo=lo_n, cur_hi=hi_n, delta=n.mean() - o.mean(),
                       pct_neg_cur=(cg.rob_label == 0).mean(), pct_pos_cur=(cg.rob_label == 2).mean(),
                       p_mwu=stats.mannwhitneyu(o, n).pvalue))
    SR = pd.DataFrame(sr)
    SR["q_fdr"] = multipletests(SR.p_mwu, method="fdr_bh")[1]
    SR.to_csv(O / "sentiment_by_domain_platform.csv", index=False)
    st_rows = []
    for t, g in df.groupby("topic"):
        o, n = g[g.period == "baseline"].rob_score, g[g.period == "current"].rob_score
        st_rows.append(dict(topic=t, label=lab[t], n=len(g), mean=g.rob_score.mean(), pct_neg=(g.rob_label == 0).mean(),
                            pct_pos=(g.rob_label == 2).mean(), base_mean=o.mean(), cur_mean=n.mean(),
                            delta=n.mean() - o.mean(),
                            p_mwu=stats.mannwhitneyu(o, n).pvalue if min(len(n), len(o)) > 10 else np.nan))
    ST = pd.DataFrame(st_rows)
    ST["q_fdr"] = multipletests(ST.p_mwu.fillna(1), method="fdr_bh")[1]
    ST.sort_values("mean").to_csv(O / "sentiment_by_topic.csv", index=False)
    dsent = df.groupby(["day", "domain"]).rob_score.mean().unstack()
    dsent.to_csv(O / "sentiment_daily.csv")

    # ---------------- 5. engagement drivers
    df["eng_pct"] = df.groupby(["platform", "source"]).eng.rank(pct=True)
    df["viral"] = (df.eng_pct >= 0.9).astype(int)
    feat = pd.get_dummies(df.topic_label, prefix="topic").astype(float)
    feat["p_neg"], feat["p_pos"] = df.p_neg, df.p_pos
    feat["log_len"] = np.log1p(df.text.str.len())
    feat["has_url"] = df.text.str.contains("http").astype(float)
    feat["question"] = df.text.str.contains(r"\?").astype(float)
    feat["hour_utc"] = df.created.dt.hour.astype(float)
    feat["weekend"] = (df.created.dt.dayofweek >= 5).astype(float)
    for p in plats:
        feat[f"plat_{p}"] = (df.platform == p).astype(float)
    y = df.viral.values
    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=RNG)  # noqa: E731
    auc = roc_auc_score(y, cross_val_predict(mk(), feat.values, y, cv=skf, method="predict_proba")[:, 1])
    bcols = [c for c in feat.columns if c.startswith("plat_") or c in ("hour_utc", "weekend", "log_len", "has_url")]
    auc_b = roc_auc_score(y, cross_val_predict(mk(), feat[bcols].values, y, cv=skf, method="predict_proba")[:, 1])
    gb = mk().fit(feat.values, y)
    sub = np.random.default_rng(RNG).choice(len(feat), min(15000, len(feat)), replace=False)
    pi = permutation_importance(gb, feat.values[sub], y[sub], scoring="roc_auc", n_repeats=5, random_state=RNG)
    grp = lambda c: "topic" if c.startswith("topic_") else "platform" if c.startswith("plat_") else \
        "sentiment" if c in ("p_neg", "p_pos") else c  # noqa: E731
    imp = pd.DataFrame({"feature": feat.columns, "importance": pi.importances_mean})
    imp["group"] = imp.feature.map(grp)
    imp.to_csv(O / "engagement_permutation_importance.csv", index=False)
    Z = (feat - feat.mean()) / feat.std().replace(0, 1)
    lr = LogisticRegression(max_iter=3000).fit(Z, y)
    pd.DataFrame({"feature": feat.columns, "odds_ratio_per_sd": np.exp(lr.coef_[0])}) \
        .sort_values("odds_ratio_per_sd", ascending=False).to_csv(O / "engagement_logit_or.csv", index=False)
    df.groupby("topic_label").viral.mean().sort_values(ascending=False).rename("top_decile_rate") \
        .to_csv(O / "engagement_by_topic.csv")
    summary["engagement"] = dict(cv_auc=float(auc), cv_auc_without_content=float(auc_b),
                                 grouped_importance=imp.groupby("group").importance.sum().sort_values(ascending=False)
                                 .round(4).to_dict())

    top = (df[df.period == "current"].sort_values(["eng_pct", "eng"], ascending=False).groupby("topic_label").head(3)
           [["topic_label", "platform", "source", "created", "eng", "text", "url"]])
    top["text"] = top.text.str[:280]
    top.to_csv(O / "top_posts_current_week.csv", index=False)

    # ---------------- 6. figures
    import umap
    idx = np.random.default_rng(RNG).choice(len(E), min(15000, len(E)), replace=False)
    U = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=RNG).fit_transform(E[idx])
    plt.rcParams.update({"figure.dpi": 130, "font.size": 8})
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(U[:, 0], U[:, 1], c=df.topic.values[idx], cmap=plt.get_cmap("tab20", best_k), s=2, alpha=.5)
    for t in range(best_k):
        m = df.topic.values[idx] == t
        if m.sum():
            ax.text(np.median(U[m, 0]), np.median(U[m, 1]), lab[t][:30], fontsize=7, ha="center",
                    bbox=dict(facecolor="white", alpha=.75, lw=0, pad=1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Topic map — UMAP of MiniLM sentence embeddings (15k-post sample)")
    plt.tight_layout()
    plt.savefig(O / "fig/topic_map.png")
    plt.close()

    Ms = M.sort_values("cmh_or")
    fig, ax = plt.subplots(figsize=(8, max(5, best_k * .32)))
    col = ["tab:green" if (q < .05 and o > 1) else "tab:red" if q < .05 else "grey" for q, o in zip(Ms.q_fdr, Ms.cmh_or)]
    ax.errorbar(Ms.cmh_or, range(len(Ms)), xerr=[Ms.cmh_or - Ms.lo, Ms.hi - Ms.cmh_or], fmt="none", ecolor="lightgrey")
    ax.scatter(Ms.cmh_or, range(len(Ms)), c=col, zorder=3)
    ax.axvline(1, color="k", lw=.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(Ms)))
    ax.set_yticklabels(Ms.label.str[:45])
    ax.set_xlabel("Odds ratio, current week vs baseline (CMH, stratified by platform)")
    ax.set_title("Topic momentum — green rising, red falling (FDR q<0.05)")
    plt.tight_layout()
    plt.savefig(O / "fig/topic_momentum.png")
    plt.close()

    STs = ST.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, max(5, best_k * .32)))
    ax.barh(range(len(STs)), STs.pct_pos * 100, color="tab:green", label="% positive")
    ax.barh(range(len(STs)), -STs.pct_neg * 100, color="tab:red", label="% negative")
    ax.set_yticks(range(len(STs)))
    ax.set_yticklabels(STs.label.str[:45])
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("% of posts (Twitter-RoBERTa)")
    ax.legend()
    ax.set_title("Sentiment by topic (whole window)")
    plt.tight_layout()
    plt.savefig(O / "fig/sentiment_topics.png")
    plt.close()

    _, cur, _ = C.window()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for c in dsent.columns:
        ax.plot(pd.to_datetime(dsent.index), dsent[c].rolling(3, min_periods=1).mean(), label=c)
    ax.axvspan(pd.Timestamp(cur.date()), pd.to_datetime(dsent.index).max(), color="tab:orange", alpha=.08)
    ax.axhline(0, color="k", lw=.5)
    ax.legend()
    ax.grid(alpha=.3)
    ax.set_title("Net sentiment P(pos) − P(neg), 3-day rolling mean (shaded = current week)")
    plt.tight_layout()
    plt.savefig(O / "fig/sentiment_daily.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, dom in zip(axes, ["AI", "Robotics"]):
        L = pd.read_csv(O / f"emerging_terms_{dom}.csv").head(20)[::-1]
        ax.barh(L.term, L.z, color="tab:purple")
        ax.set_title(f"Emerging this week — {dom}\n(log-odds z vs baseline)")
    plt.tight_layout()
    plt.savefig(O / "fig/emerging_terms.png")
    plt.close()

    (O / "text_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    df[["id", "platform", "source", "created", "domain", "period", "topic", "topic_label", "rob_score", "rob_label",
        "vader", "eng", "eng_pct"]].to_csv(O / "doc_assignments.csv.gz", index=False)
    return summary
