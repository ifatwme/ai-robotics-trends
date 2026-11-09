# AI & Robotics Social-Media Trends (November 2026): Project Status Report
_Status as of 2026-11-09 ~13:20 UTC · Host: Testilonia (65.109.162.75) · Project root: `/root/aitrends`_

## 1. Scope
- **Question:** What were the social-media trends around AI and robotics in November 2026, measured with statistics and ML?
- **Window:** November is still in progress. The analysis covers **Nov 1–8 (complete days)**, with **October 1–31 as the baseline**. Nov 9 data (to 12:00 UTC) was collected but is left out of inference because it is a partial day.

## 2. Data collected (done)
| Source | Method | Volume |
|---|---|---|
| Bluesky: keyword counts | Public AppView search, `hitsTotal` per keyword-day, 35 keywords × 40 days | 1,400 series-days |
| Bluesky: "AI" uncapped counts | Adaptive sub-day windows (the API caps counts at 10k) | 632 windows, 0 still capped |
| Bluesky: posts | Top-200/day for 8 queries | 28,539 posts |
| Hacker News | Algolia: stories + comment counts, 36 queries × 40 days | 32,833 story hits, 0 failed query-days after repair |
| Reddit | Arctic Shift archive, full census of 20 AI/robotics subreddits | ~41 MB raw |
| Mastodon | mastodon.social hashtag timelines, 12 tags | ~40 MB raw |

Unified corpus after relevance filtering and de-duplication: about **98k documents** (`data/corpus.pkl`).

**Not available:** X/Twitter (no API credentials; `xurl` not set up), TikTok, Instagram, LinkedIn and YouTube. Direct Reddit access is blocked from this server, so the Arctic Shift archive was used instead. The web-search backend was down, so there is no external news context yet.

## 3. Analysis
### Done
- **ML enrichment:** MiniLM sentence embeddings computed for all documents (about 27 min on CPU).
- **Volume statistics** (`analysis_volume.py`, preliminary run on Bluesky + HN):
  - Negative-binomial GLMs with day-of-week effects; rate ratios; Benjamini–Hochberg FDR.
  - Robust burst detection.
  - Change-point detection.
  - χ² share-of-voice tests.
  - Full-month projection by simulation.

### In progress
- **Sentiment scoring** with Twitter-RoBERTa (int8-quantized for CPU speed) plus a VADER cross-check on about 91k documents. ETA is roughly 25–35 min.

### Still to run
- **Volume statistics re-run** with the Reddit and Mastodon series added.
- **Text/ML analysis** (`analysis_text.py`, written, not yet run):
  - Embedding k-means topics (k chosen by silhouette), labelled with c-TF-IDF.
  - NMF cross-check.
  - Platform-stratified CMH topic-momentum tests.
  - Log-odds emerging terms.
  - Sentiment CIs and tests.
  - Gradient-boosting engagement model (CV AUC, permutation importance).
  - UMAP topic map.

## 4. Preliminary findings (Bluesky + HN only; to be confirmed)
- **Overall AI volume is flat to slightly down.** Bluesky "AI" is about 23.2k posts/day in Nov vs 25.7k in Oct (IRR 0.93, not significant after FDR).
- **Nvidia is the November breakout.** +62% daily rate (q=0.03), share of voice up +5.0 pp, and a burst on Nov 8 (2,035 posts vs ~450 baseline).
- **Claude/Anthropic split by platform.** HN comments mentioning Claude are up +49% (q<0.001, change point Nov 2). Top HN threads include Claude Code privacy/security stories and Goldman Sachs pulling Claude Code. On Bluesky, Anthropic and OpenAI share of voice fell.
- **"AI bubble" narrative.** Change point Oct 26, peak Oct 30 (858/day), still about +62% vs the October average (q≈0.10).
- **Significant declines:** "LLM" −25%, "artificial intelligence" −27%, Sora −21%, AGI −22%, ChatGPT −12% (all q<0.05).
- **Robotics:**
  - Generic "robot" volume is stable.
  - "humanoid" is −10%.
  - Boston Dynamics is −70% after its October bursts.
  - Waymo and robotaxi gained share (+5.5 pp, +3.9 pp), with a robotaxi/self-driving burst on Nov 6.
- **November bursts:** "AI regulation" on Nov 4–6 and "data center" on Nov 4–6.

## 5. Engineering and deployment (done)
- **Code:** `app/` holds collectors, corpus builder, ML enrichment, two analysis modules and the site renderer. Paths are configurable through environment variables.
- **Containers:**
  - `aitrends-pipeline:1.0.0`: Python 3.14-slim, pinned deps, CPU torch, non-root uid 10001, tini.
  - `aitrends-web:1.0.0`: nginx-unprivileged 1.29-alpine.
- **Compose (`docker-compose.yml`):** The web service is read-only, drops all capabilities, sets no-new-privileges, has memory/CPU/PID limits, log rotation, a healthcheck and `restart: unless-stopped`. The pipeline is a one-off job (profile `pipeline`; `COLLECT=1` re-downloads the data).
- **HTTPS:** Let's Encrypt IP certificate (short-lived profile, via lego HTTP-01). HTTP redirects to HTTPS, and security headers are set. Renewal runs from the systemd timer `aitrends-cert-renew.timer` twice a day via `scripts/renew_cert.sh`.
- **Live:** `https://65.109.162.75/`, currently showing the preliminary volume results. It returns HTTP 200 from Germany, Iceland and Russia.

## 6. Issues encountered
| Issue | Resolution |
|---|---|
| Hacker News API rate limits (675 failed query-days) | Repaired with a slower, incrementally-saving script |
| Bluesky 10k count cap | Adaptive window splitting |
| `ruptures` would not build on Python 3.14 | Wrote a custom change-point detector |
| Slow sentiment scoring | Quantized the model |
| Dashboard unreachable from Iran (connection reset; the Iran probe also failed) | ISP-level filtering, not a server fault; open it through a VPN |

## 7. Next steps
1. Finish sentiment scoring, then rebuild the corpus.
2. Run the full pipeline **inside the pipeline container** (`docker compose run --rm pipeline`), after `chown 10001` on `data/ out/ site/ cache/`.
3. Inspect the topics and write human-readable labels and a key-findings narrative (`out/topic_labels.json`, `out/findings.html`).
4. Re-render and publish the site, then verify it externally.
5. Optional: add X/Twitter if you provide API credentials, and add news context once web search works again.

## 8. File map
`app/` code · `docker/` Dockerfiles, nginx.conf, run_pipeline.sh · `docker-compose.yml` · `requirements.txt` · `scripts/renew_cert.sh` · `data/` raw data and caches · `out/` results · `site/` published website · `certs/`, `tls/` certificates · `logs/` and `log_*.txt` run logs · `venv/` dev environment
