"""Prepara catalogo, consultas y tienda simulada a partir de ESCI (small_version, locale es).

Salida en data/: catalog.parquet, queries.parquet, qrels.parquet, store_seed.json
Uso: python src/prepare_data.py --raw data_raw/esci --out data
"""
import argparse, json, os
import numpy as np
import pandas as pd

SEED = 42
N_TEST_Q, N_DEV_Q = 220, 100  # >=200 consultas de evaluacion


def build(raw, out):
    rng = np.random.default_rng(SEED)
    ex = pd.read_parquet(f"{raw}/shopping_queries_dataset_examples.parquet")
    ex = ex[(ex.product_locale == "es") & (ex.small_version == 1)]

    # Separacion por query_id: train = desarrollo, test = evaluacion (sin mezclar)
    picks = []
    for split, n in (("train", N_DEV_Q), ("test", N_TEST_Q)):
        qids = np.sort(ex[ex.split == split].query_id.unique())
        picks.append(pd.DataFrame({"query_id": rng.choice(qids, n, replace=False), "split": split}))
    qsel = pd.concat(picks, ignore_index=True)
    qrels = ex.merge(qsel[["query_id"]], on="query_id")[
        ["query_id", "query", "split", "product_id", "esci_label"]]

    pids = set(qrels.product_id)
    pr = pd.read_parquet(f"{raw}/shopping_queries_dataset_products.parquet")
    cat = pr[(pr.product_locale == "es") & pr.product_id.isin(pids)].drop(columns="product_locale")
    cat = cat.drop_duplicates("product_id").reset_index(drop=True)
    # todos los productos anotados deben estar en el catalogo
    qrels = qrels[qrels.product_id.isin(cat.product_id)]
    queries = qrels[["query_id", "query", "split"]].drop_duplicates().reset_index(drop=True)

    # Tienda simulada: precio (EUR, lognormal) y stock con semilla fija. Sin etiquetas ESCI.
    prices = np.round(np.clip(rng.lognormal(3.2, 0.9, len(cat)), 2.0, 600.0), 2)
    stock = rng.choice([0, 1, 3, 5, 10, 25, 60], len(cat), p=[.08, .07, .15, .2, .25, .15, .1])
    store = {"seed": SEED, "products": {p: {"price": float(pr_), "stock": int(s)}
                                         for p, pr_, s in zip(cat.product_id, prices, stock)},
             "carts": {}, "orders": []}

    os.makedirs(out, exist_ok=True)
    cat.to_parquet(f"{out}/catalog.parquet")
    queries.to_parquet(f"{out}/queries.parquet")
    qrels.to_parquet(f"{out}/qrels.parquet")
    json.dump(store, open(f"{out}/store_seed.json", "w"))
    print(f"catalogo={len(cat)} consultas={queries.split.value_counts().to_dict()} qrels={len(qrels)}")
    print(qrels.esci_label.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data_raw/esci")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    build(a.raw, a.out)
