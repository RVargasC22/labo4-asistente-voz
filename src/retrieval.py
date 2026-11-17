"""Recuperacion: BM25 (base lexica) y denso (e5) + fusion RRF; nDCG@10 sobre candidatos anotados."""
import re, unicodedata
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
STOP = set("de la el los las un una y o en con para por del al a que se su sus es lo".split())


def tok(s):
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return [t for t in re.findall(r"[a-z0-9]+", s) if t not in STOP]


def doc_text(r, desc=True):
    parts = [r.product_title, r.product_brand, r.product_color, r.product_bullet_point]
    if desc:
        parts.append(r.product_description)
    return " ".join(str(p) for p in parts if p and str(p) != "None")[:1500]


class Lexical:
    def __init__(self, cat, desc=True):
        self.ids = cat.product_id.values
        self.bm = BM25Okapi([tok(doc_text(r, desc)) for r in cat.itertuples()])

    def scores(self, q):
        return self.bm.get_scores(tok(q))


class Dense:
    def __init__(self, cat, model="intfloat/multilingual-e5-base", desc=True, bs=64):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model, device="cuda")
        self.m.half()
        self.m.max_seq_length = 256
        self.ids = cat.product_id.values
        docs = ["passage: " + doc_text(r, desc) for r in cat.itertuples()]
        self.E = self.m.encode(docs, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)

    def scores(self, q):
        return self.E @ self.m.encode("query: " + q, normalize_embeddings=True)


def rrf(*score_vecs, k=60):
    out = np.zeros(len(score_vecs[0]))
    for s in score_vecs:
        rank = np.empty(len(s)); rank[np.argsort(-s)] = np.arange(len(s))
        out += 1.0 / (k + rank + 1)
    return out


def ndcg10(ranked_labels):
    g = np.array([GAIN[l] for l in ranked_labels[:10]])
    dcg = (g / np.log2(np.arange(2, len(g) + 2))).sum()
    ig = np.sort(np.array([GAIN[l] for l in ranked_labels]))[::-1][:10]
    idcg = (ig / np.log2(np.arange(2, len(ig) + 2))).sum()
    return dcg / idcg if idcg > 0 else np.nan


def evaluate(score_fn, ids, qrels, split="test"):
    """score_fn(query)->scores sobre todo el catalogo; se rankean solo los candidatos anotados."""
    pos = {p: i for i, p in enumerate(ids)}
    res = []
    for qid, g in qrels[qrels.split == split].groupby("query_id"):
        s = score_fn(g["query"].iloc[0])
        sc = np.array([s[pos[p]] for p in g.product_id])
        res.append(ndcg10(g.esci_label.values[np.argsort(-sc)]))
    return float(np.nanmean(res)), len(res)
