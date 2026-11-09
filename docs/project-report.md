# Project report — AI & Robotics Social-Media Trends

_Status: 2026-11-09 · Repo: https://github.com/ifatwme/ai-robotics-trends · Live: https://65.109.162.75 (VPS) and
https://ifatwme.github.io/ai-robotics-trends/ (GitHub Pages, after the first successful weekly run)_

## 1. Goal
Measure how AI and robotics are discussed on social media with real collected data, statistical inference and
machine learning, and publish the results as a weekly-updated dashboard. Methods and the rationale for every metric:
[`methodology.md`](methodology.md).

## 2. First edition — week 2026-11-02 → 2026-11-08 vs 4-week baseline (2026-10-05 → 2026-11-01)
88,487 posts (Bluesky 19.1k, Hacker News 7.8k, Mastodon 16.7k, Reddit 44.9k); 83 volume series, 19 significant
after Benjamini–Hochberg FDR.

| Finding | Evidence |
|---|---|
| Overall AI chatter **dipped** | Bluesky "AI" 23.6k vs 25.9k posts/day, −9% (q = 0.023) |
| **Nvidia** is the week's breakout | +64% Bluesky, +92% HN comments (q < 0.05); share of AI-brand voice +5.7 pp → 13.3%; spike on 11-08 (2,035 vs ~450/day) |
| **Claude** gains developer attention, **Anthropic** loses broad share | +46% HN comments (q < 0.05); Anthropic share of voice −3.3 pp |
| Generic/hype terms cool | "LLM" −27%, "AGI" −23% on Bluesky; r/singularity −27%, r/GeminiAI −41% posts |
| **Autonomous driving** is the robotics story | Waymo share of robotics-brand voice +6.0 pp → 37.1%; robotaxi & self-driving spikes on 11-06 |
| Humanoid hype fades | Boston Dynamics −69% (after October bursts) |
| Politics enters the AI conversation | Emerging terms: midterms, voters, democrats, schools, data centers, "GPT-6 cyber"; "AI regulation" spike 11-05 |
| New robotics vocabulary | Cybertruck production, ABB Robotics / SoftBank, Agility, China, home robots |
| Sentiment: robotics warmer than AI | net +0.116 robotics vs +0.054 AI (13% vs 8% positive posts); week-over-week changes small — see caveat |
| What gets engagement | GBM top-decile classifier CV AUC 0.67 vs 0.62 content-free baseline; length, sentiment, links matter most |

Spike causes are not asserted in this report: the dashboard lists the most-engaged posts per topic so they can be checked.

## 3. Confidence
High: volume changes (NB GLM + FDR), spikes, brand share of voice. Medium: topics (silhouette 0.059, NMI 0.39 vs NMF),
emerging terms, engagement model, outlook. **Low–medium: sentiment** (Cohen's κ 0.16 vs VADER).
Not covered: X/Twitter, TikTok, Instagram, LinkedIn, YouTube.

## 4. Engineering
* **Repo layout:** `src/aitrends` package (collect → corpus → enrich → volume/text → history → site), `docker/`,
  `docker-compose.yml` + `deploy/compose.server.yml`, `deploy/systemd/`, `scripts/`, `.github/workflows/`, `docs/`.
* **Containers:** `pipeline` (Python 3.14, non-root, batch job) and `web` (nginx-unprivileged, read-only, health-checked),
  hardened with dropped capabilities, no-new-privileges, resource limits and log rotation.
* **VPS:** HTTPS with a Let's Encrypt IP certificate, twice-daily renewal timer, weekly pipeline timer (Mon 03:30 UTC).
* **GitHub:** CI (lint, compose validation, both image builds, web smoke test, import of every module) and a weekly
  workflow (Mon 04:17 UTC) that builds, collects, analyses, caches ML results, commits `history/weekly_metrics.csv`
  and deploys GitHub Pages. Free on a public repository.
* **Resources per weekly run:** ~45–70 min on 4 vCPU (first run ~2 h), peak RAM 3–4 GB, ~5 GB disk.

## 5. Issues found and fixed during delivery
| Issue | Fix |
|---|---|
| Bluesky counts capped at 10,000 | adaptive sub-day windows |
| HN / Arctic Shift / Mastodon rate limiting → partial data | per-host throttling, retries on 429/5xx/"slow down" 422, failed units recorded as missing (never zero), gap-fill mode |
| Linear-trend forecast exploded on bursty series | day-of-week run-rate model |
| `config as C` shadowed patsy's `C()` in model formulas | alias renamed |
| pandas attribute collisions (`.count`, `.pct_change`) in the site renderer | explicit column access |
| Unanchored `site/` in `.gitignore` kept `src/aitrends/site/` out of git | patterns anchored; CI now imports every module |
| lego v5 CLI change broke certificate renewal | `run` handles issue + renewal |
| Dashboard unreachable from Iran | ISP filtering, verified reachable from DE/IS/RU — use a VPN |

## 6. Next steps
1. Confirm the first GitHub Pages deployment (weekly workflow on the latest commit).
2. Methodology improvements: hand-labelled sentiment validation set, consistent bot removal, better topic model,
   day-block bootstrap for share of voice (see `methodology.md` §6).
3. Optional: add X/Twitter if API credentials are provided.
