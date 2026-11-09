# Methodology

How the pipeline turns public posts into weekly trend statistics: what is measured, which
statistical / machine-learning technique produces each number, **why that technique was chosen**,
and how much each result can be trusted.

> Example values quoted below come from the run for the week **2026-11-02 → 2026-11-08**
> (baseline 2026-10-05 → 2026-11-01, 88,487 posts). Every weekly run re-computes them; the current
> values are in `out/text_summary.json` / `out/volume_summary.json` and on the dashboard.

The pipeline has four layers, each answering a different question with its own metrics:

| Layer | Question | Main metrics |
|---|---|---|
| 1. Data mining | What was said, how often, where? | complete daily counts, post samples |
| 2. Corpus | Which posts are relevant, and in which period? | domain labels, de-duplication |
| 3. Statistics | Is a change real or noise? How big? | IRR + CI, FDR q-values, robust z, Cramér's V, CMH OR, log-odds z |
| 4. Machine learning | What topics, what tone, what gets engagement? | silhouette, NMI/ARI, net sentiment, κ, CV ROC-AUC |

---

## 0. Analysis window

Rolling: the **current week** is the 7 complete UTC days before the run date; the **baseline** is the
28 days before that. Set `WINDOW_END` (exclusive, `YYYY-MM-DD`) to reproduce a specific week.

*Why:* four weeks is long enough for a stable baseline and contains every weekday four times (so
weekday effects can be estimated), yet short enough to reflect the current state of the conversation.
Only complete days are used — a partial day looks like a drop.

---

## 1. Data mining (collection)

| Platform | Endpoint | Unit collected | Why this unit |
|---|---|---|---|
| Bluesky | `api.bsky.app/xrpc/app.bsky.feed.searchPosts` | daily `hitsTotal` per keyword (English); top-200 posts/day for 8 queries | counts measure **attention** completely and cheaply; the text sample feeds topics/sentiment |
| Hacker News | `hn.algolia.com/api/v1/search_by_date` | comment counts per query-day; story titles | comments measure developer **discussion intensity** better than story counts |
| Reddit | `arctic-shift.photon-reddit.com/api/posts/search` | every post in 20 AI/robotics subreddits | a **census** of those communities — no sampling error (reddit.com blocks datacenter IPs, hence the archive) |
| Mastodon | `mastodon.social/api/v1/timelines/tag/<tag>` | ≤120 newest posts per tag-day | adds the fediverse; the API has no totals, so daily rates are extrapolated from timestamps when a day is truncated |

**Design decisions**

* **Counts first, texts second.** Volume statistics need *complete* counts; text analyses can work from samples.
* **The Bluesky 10,000 cap.** `hitsTotal` saturates at 10k — every day would read exactly 10000 and the
  trend would vanish. High-volume terms are counted in 3-hour windows, split recursively until uncapped.
* **Polite, fault-tolerant client.** Per-host minimum request intervals; retries with back-off on 429/5xx and
  on Arctic Shift's `422 "Timeout. Maybe slow down"`. Failed units are retried and, if still failing,
  recorded (`masto_failed.json`) so the models treat them as **missing, never as zero** — a zero-filled
  outage once produced a spurious "−88%" for `#llm`.

**Known weaknesses:** Bluesky texts are a *top*-sorted sample (popular posts over-represented); Reddit
scores are archived shortly after posting (engagement understated); Mastodon is a single, bot-heavy
instance (bot accounts are dropped once every row carries the `bot` flag, to avoid biasing one period);
X/Twitter, TikTok, Instagram, LinkedIn and YouTube are not covered (no keyless access).

---

## 2. Corpus construction

* **Relevance filter** — regexes label each post **AI**, **Robotics** or **AI+Robotics**; dedicated
  subreddits/hashtags count as in-domain. *Why:* keyword search returns noise (e.g. HN stories matching
  "model" or "agent").
* **Exact-duplicate removal** within a platform (6,167 in the example week). *Why:* cross-posts and spam
  inflate counts and distort topics.
* **Period label** `baseline` / `current` from the window above.

---

## 3. Statistics — is a change real, and how big?

### 3.1 Volume change → negative-binomial GLM, incidence-rate ratio (IRR)
For every daily series (keyword on Bluesky, query on HN comments, subreddit, Mastodon tag — 83 series):

`count ~ current_week + C(day_of_week)` (NB2; quasi-Poisson fallback)

* **Metric: IRR** = current daily rate ÷ baseline daily rate, with a 95% CI (IRR 1.64 ⇒ +64%).
* **Why negative binomial, not Poisson?** Social counts are *overdispersed* (variance ≫ mean). Poisson would
  understate uncertainty and declare almost everything significant; NB estimates the extra variance.
* **Why day-of-week?** Weekends are quieter; without the control, a different weekday mix fakes a change.

### 3.2 Multiple testing → Benjamini–Hochberg FDR (q-values)
Testing 83 series at p<0.05 would yield ~4 false "trends" by chance. BH controls the expected share of
false discoveries. **Only q<0.05 is called rising/falling** — the dashboard's auto-written findings obey this
rule (19 of 83 series in the example week).

### 3.3 Spikes → robust z-score
`z = (today − median₁₄) / (1.4826·MAD₁₄)`, flagged at z ≥ 3.5 and ≥ 20 posts, with a Poisson floor on the
denominator. *Why median/MAD instead of mean/SD?* An earlier spike inflates the SD and masks the next one;
median/MAD are robust to outliers. A spike says *that* something happened — the cause is checked manually
against the most-engaged posts of that day.

### 3.4 Level shifts → change-point detection
Binary segmentation on `log(1+count)`, Gaussian cost, penalty `2·log(n)·var`. *Why:* dates **when** a
narrative stepped up (e.g. "AI bubble" on 2026-10-26); the log scale makes doubling equally visible at any volume.

### 3.5 Brand competition → share of voice, χ², Cramér's V, adjusted residuals
* **Share of voice** — a brand's share among its competitors; removes the effect of overall volume changes.
* **χ² homogeneity** — did the brand mix change between periods?
* **Cramér's V** — *how much* (0.094 for AI brands, 0.119 for robotics brands in the example week: small–moderate).
  *Why needed:* with hundreds of thousands of mentions χ² is always significant; V is the effect size.
* **Adjusted standardized residuals** (|r|>2) — which brand drove the change.

### 3.6 Topic momentum → Cochran–Mantel–Haenszel odds ratio
For each topic, a 2×2 table (in topic × current week) per platform, pooled with Mantel–Haenszel; CMH test + FDR.
*Why stratify?* Platforms differ in size and content — if Reddit's share of the corpus grew, Reddit-flavoured
topics would look like they were "rising". CMH compares within each platform, then pools. Shares shown on the
dashboard are platform-balanced (mean of within-platform shares).

### 3.7 Emerging vocabulary → weighted log-odds with informative Dirichlet prior
Monroe, Colaresi & Quinn (2008), current vs baseline, on platform-balanced document frequencies (1–2-grams).
* **Metric:** z-score per term (z > 1.96 ≈ significant).
* **Why not % change in frequency?** Rare words give huge meaningless ratios (1 → 3 = "+200%"); the prior
  shrinks rare terms towards zero, so a high z needs both a real shift and enough evidence. Platform
  balancing stops Reddit's volume from dictating the vocabulary.

### 3.8 Next-7-day outlook → day-of-week NB run-rate + simulation
NB `count ~ C(day_of_week)` on the last 28 days; 4,000 simulations of coefficients (multivariate normal) and
NB count noise → median + **80% prediction interval**. *Why no trend term?* A linear trend fitted through a
spike extrapolates absurdly (it did for "AI bubble"). The run-rate model is honest about what is forecastable;
it explicitly cannot foresee new shocks.

---

## 4. Machine learning

### 4.1 Sentence embeddings — `all-MiniLM-L6-v2`
384-dimensional, L2-normalised vectors of each post's meaning (first 600 characters). *Why:* posts with the
same meaning but different words ("Claude Code leaked my keys" / "Anthropic's coding agent exposes secrets")
land close together; MiniLM is small enough to embed ~100k posts on CPU. Cached by post id, so weekly runs
only embed new posts.

### 4.2 Topic discovery — k-means on embeddings, class-based TF-IDF labels, NMF cross-check
* **Choosing k → cosine silhouette** (how much closer a post is to its own cluster than to the next), grid
  k ∈ {12, 15, …, 30}, ties broken towards fewer topics for readability.
* **Labels → class-based TF-IDF** (1–2-grams frequent in one topic, rare in the others); duplicate labels get
  the next distinctive term.
* **Validation → an independent model** (TF-IDF + NMF, a bag-of-words method) with the same k, compared by
  **NMI** (shared information between the two partitions, 0–1) and **ARI** (chance-corrected agreement of
  post-to-topic assignments).

| Metric (example week) | Value | Interpretation |
|---|---|---|
| Silhouette (k = 15) | **0.059** | Weak separation — topics overlap, typical for short social posts |
| NMI vs NMF | **0.39** | Moderate — two very different methods find partly the same structure |
| ARI vs NMF | **0.24** | Low agreement on exact assignments |

**Reading:** topics are reliable for **broad themes and their weekly momentum**, not for classifying an
individual post. Some clusters are partly driven by bot/feed accounts.

### 4.3 Sentiment — Twitter-RoBERTa (primary) with VADER (check)
* **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (trained on tweets — suited to short social
  text), int8 dynamically quantised for CPU, 112 tokens.
* **Metric: net sentiment** = P(positive) − P(negative), averaged; plus % negative / % positive posts.
* **Uncertainty:** 95% bootstrap CIs; period differences by **Mann–Whitney U** (no normality assumption) with FDR.
* **Agreement check** against the VADER lexicon:

| Metric (example week) | Value | Interpretation |
|---|---|---|
| Spearman ρ (scores) | **0.44** | Moderate agreement on direction |
| Cohen's κ (3 classes) | **0.16** | Poor agreement on exact labels |
| Raw agreement | **0.43** | Under half the posts get the same label |

**Reading: this is the weakest component.** Tech posts are mostly neutral, news-like or sarcastic, and the
methods disagree on them. Trust **large differences between topics**; do not over-read small week-to-week
moves (e.g. +0.054 vs +0.058). Caveats: ~6,400 posts were scored by the unquantised model before the switch
to int8; there is not yet a hand-labelled sample to measure true accuracy.

### 4.4 Engagement drivers — gradient-boosted trees
* **Target:** post is in the **top 10% of engagement within its own feed** (Reddit vs Reddit, Bluesky vs
  Bluesky) — engagement scales differ by orders of magnitude between platforms.
* **Features:** topic, sentiment probabilities, length, link, question mark, hour, weekend, platform.
* **Model:** `HistGradientBoostingClassifier` (captures non-linear effects and interactions).
* **Main metric: ROC-AUC with stratified 5-fold cross-validation.** *Why AUC?* It is threshold-free and
  stays meaningful with a 10% positive class (accuracy would be 90% for always predicting "not viral").
  *Why cross-validation?* Scores on unseen posts, not on training data.
* **Baseline comparison:** a content-free model (platform, timing, length, link) — the AUC gap is the evidence
  that *what is said* matters: **0.672 vs 0.621** in the example week.
* **Drivers:** grouped **permutation importance** (drop in AUC when a feature group is shuffled) — length
  0.119, sentiment 0.083, link 0.058, topic 0.053, platform 0.048; direction from standardized
  **logistic-regression odds ratios**.
* **Reading:** AUC ≈ 0.67 is modest — engagement depends mostly on unmeasured factors (author audience, news
  timing, images). Useful for "which kinds of posts tend to do better", not for predicting a single post.

---

## 5. Confidence summary

| Question | Metric | Confidence |
|---|---|---|
| Did volume change? | NB rate ratio + 95% CI, BH q-value | **High** — complete counts, overdispersion and weekday handled |
| Was there a spike? | robust z vs 14-day median/MAD | **High** for detection; cause needs manual checking |
| Did the brand mix shift? | share of voice, χ², Cramér's V, residuals | **High** (Bluesky only) |
| Which topics grew? | CMH odds ratio across platforms | **Medium** — fuzzy topics (silhouette ≈ 0.06) |
| What is the new language? | Monroe log-odds z | **Medium–high** |
| How do people feel? | RoBERTa net sentiment, bootstrap CI | **Low–medium** — κ = 0.16 vs VADER |
| What drives engagement? | CV ROC-AUC, permutation importance | **Medium** — AUC 0.67 vs 0.62 baseline |
| What next week? | NB run-rate, 80% PI | **Medium** — cannot foresee shocks |

General rule: with ~100k posts, tiny effects become statistically significant — **read effect sizes (IRR,
Cramér's V, OR, Δ pp) first and q-values second.**

## 6. Planned improvements

1. **Hand-labelled validation set** (~300 posts) → real sentiment accuracy/F1; quantify fp32 vs int8 agreement.
2. **Consistent bot/feed-account removal** on all platforms (Mastodon flag already collected from the next full run).
3. **Better topics** — HDBSCAN / BERTopic-style reduction with a topic-coherence score.
4. **Day-block bootstrap** for share-of-voice uncertainty (mentions are not independent), and a "latest"-sorted
   Bluesky sample alongside "top".

## 7. Reproducibility

Pinned dependencies, fixed random seed (42), containerised execution, per-run `out/run_meta.json`, id-keyed
ML caches, and weekly headline metrics committed to `history/weekly_metrics.csv`.
Reproduce a week: `WINDOW_END=2026-11-09 docker compose run --rm pipeline collect all`.

### References
* Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict. *Political Analysis*, 16(4).
* Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate. *JRSS-B*, 57(1).
* Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
* Barbieri, F. et al. (2020). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. *Findings of EMNLP*.
* Hutto, C., & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *ICWSM*.
* Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP*.
