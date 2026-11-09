"""ML enrichment with persistent, id-keyed caches (only new documents are processed each week).

- Sentence embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-d, L2-normalised)
- Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest, int8 dynamically quantised for CPU
- Lexicon sentiment: VADER (used as an independent agreement check)
Caches are pruned to the current corpus so they stay bounded (~2 kB per document).
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from . import config as C

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _load_cache(path):
    return pd.read_pickle(path) if path.exists() else {}


def run() -> None:
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4)))
    df = pd.read_pickle(C.DATA_DIR / "corpus.pkl")
    ids = df.id.tolist()
    texts = df.text.str.replace(r"https?://\S+", "", regex=True).str[:600].tolist()

    # ---- embeddings
    path_e = C.DATA_DIR / "emb_cache.pkl"
    ce = _load_cache(path_e)
    todo = [i for i, x in enumerate(ids) if x not in ce]
    print(f"enrich: embeddings to compute {len(todo)} / {len(ids)}", flush=True)
    if todo:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(EMB_MODEL, device="cpu")
        t = time.time()
        E = m.encode([texts[i] for i in todo], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        for i, e in zip(todo, E):
            ce[ids[i]] = e.astype(np.float32)
        print(f"enrich: embeddings {time.time() - t:.0f}s", flush=True)
    ce = {k: ce[k] for k in ids}  # prune
    pd.to_pickle(ce, path_e)
    np.save(C.DATA_DIR / "emb.npy", np.vstack([ce[x] for x in ids]))

    # ---- sentiment
    path_s = C.DATA_DIR / "sent_cache.pkl"
    cs = _load_cache(path_s)
    todo = [i for i, x in enumerate(ids) if x not in cs]
    print(f"enrich: sentiment to compute {len(todo)} / {len(ids)}", flush=True)
    if todo:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        tok = AutoTokenizer.from_pretrained(SENT_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL).eval()
        mdl = torch.ao.quantization.quantize_dynamic(mdl, {torch.nn.Linear}, dtype=torch.qint8)
        vader = SentimentIntensityAnalyzer()
        t, B = time.time(), 64
        order = sorted(todo, key=lambda i: len(texts[i]))  # length bucketing -> less padding
        for k in range(0, len(order), B):
            idx = order[k:k + B]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=112, return_tensors="pt")
            with torch.inference_mode():
                p = torch.softmax(mdl(**enc).logits, -1).numpy()
            for i, pr in zip(idx, p):
                cs[ids[i]] = (float(pr[0]), float(pr[1]), float(pr[2]), vader.polarity_scores(texts[i])["compound"])
            if k and k % (B * 150) == 0:
                print(f"enrich: sentiment {k}/{len(order)} ({time.time() - t:.0f}s)", flush=True)
                pd.to_pickle(cs, path_s)
        print(f"enrich: sentiment {time.time() - t:.0f}s", flush=True)
    cs = {k: cs[k] for k in ids}
    pd.to_pickle(cs, path_s)
    S = pd.DataFrame([cs[x] for x in ids], columns=["p_neg", "p_neu", "p_pos", "vader"])
    S["id"] = ids
    S["rob_score"] = S.p_pos - S.p_neg
    S["rob_label"] = S[["p_neg", "p_neu", "p_pos"]].values.argmax(1)  # 0 neg, 1 neu, 2 pos
    S.to_pickle(C.DATA_DIR / "sentiment.pkl")
