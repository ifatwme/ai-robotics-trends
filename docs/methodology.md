# Methodology

## Window
Rolling: the **current week** is the 7 complete UTC days before the run date; the **baseline** is the 28 days before
that. Set `WINDOW_END` (exclusive) to reproduce a specific week.

## Data sources
| Platform | Endpoint | Unit | Notes |
|---|---|---|---|
| Bluesky | `api.bsky.app/xrpc/app.bsky.feed.searchPosts` | daily `hitsTotal` per keyword (English) | counts saturate at 10,000 → high-volume terms are counted in 3-hour windows split recursively until uncapped; post texts = top 200/day for 8 queries |
| Hacker News | `hn.algolia.com/api/v1/search_by_date` | story titles; comment counts per query-day | failed query-days are retried until complete |
| Reddit | `arctic-shift.photon-reddit.com/api/posts/search` | every post in 20 subreddits | scores captured shortly after posting |
| Mastodon | `mastodon.social/api/v1/timelines/tag/<tag>` | ≤120 newest posts per tag-day | daily rate extrapolated from timestamps when truncated |

## Corpus
Relevance regexes assign each post to **AI**, **Robotics** or **AI+Robotics** (dedicated subreddits/hashtags count as
in-domain). Exact duplicates within a platform are removed.

## Volume
For each daily series: negative-binomial (NB2) regression `y ~ current + C(dow)`; the exponentiated `current`
coefficient is the incidence-rate ratio (quasi-Poisson fallback). Multiple testing: Benjamini–Hochberg FDR.
Bursts: robust z = (y − median₁₄) / (1.4826·MAD₁₄) with a Poisson floor; flagged at z ≥ 3.5 and y ≥ 20.
Change points: binary segmentation on log(1+y), Gaussian cost, penalty 2·log(n)·var.
Share of voice: χ² test of brand × period, Cramér's V, adjusted standardized residuals.
Outlook: NB `y ~ C(dow)` on the last 28 days; 4,000 draws of coefficients (multivariate normal) and NB noise.

## Text & ML
* **Embeddings:** `all-MiniLM-L6-v2`, L2-normalised, first 600 characters.
* **Topics:** MiniBatchKMeans for k ∈ {12,15,…,30}; k maximising cosine silhouette (12k sample), ties → smaller k.
  Labels from class-based TF-IDF (1–2-grams). Cross-check: TF-IDF + NMF with the same k (NMI/ARI reported).
* **Momentum:** per topic, 2×2 tables (topic × period) per platform → Mantel–Haenszel pooled OR, CMH test, FDR.
* **Emerging terms:** Monroe, Colaresi & Quinn (2008) weighted log-odds with informative Dirichlet prior on
  platform-balanced document frequencies.
* **Sentiment:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (int8 dynamic quantisation, 112 tokens);
  net score = P(pos) − P(neg). VADER compound as an independent check (Spearman ρ, Cohen's κ).
  Period differences: Mann–Whitney U + FDR; 95% bootstrap CIs.
* **Engagement:** target = top 10% engagement within the post's own feed; features = topic, sentiment probabilities,
  length, URL, question mark, hour, weekend, platform. HistGradientBoosting, stratified 5-fold CV ROC-AUC vs a
  no-content baseline; permutation importance (grouped); standardized logistic-regression odds ratios.

## Reproducibility
Pinned dependencies, fixed random seeds (42), containerised execution, per-run `out/run_meta.json`, and weekly
headline metrics committed to `history/weekly_metrics.csv`.
