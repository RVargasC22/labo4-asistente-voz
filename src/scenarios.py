"""Genera 20 escenarios reproducibles (semilla fija) a partir de consultas de TEST y la semilla de la tienda.

Las etiquetas ESCI solo se usan aqui para fijar el estado inicial y los criterios de exito (nunca las ve el agente).
Salida: data/scenarios.json
"""
import json, math
import numpy as np
import pandas as pd

SEED = 7
GENERIC = ["algo para regalar", "unos zapatos", "un teléfono", "una funda"]  # ambiguas a proposito


def build(data="data", seed=SEED, out="scenarios.json", exclude=()):
    """exclude: textos de consulta a omitir (p. ej. los de desarrollo, para un conjunto de validacion aparte)."""
    rng = np.random.default_rng(seed)
    qr = pd.read_parquet(f"{data}/qrels.parquet")
    cat = pd.read_parquet(f"{data}/catalog.parquet").set_index("product_id")
    seed = json.load(open(f"{data}/store_seed.json"))["products"]
    qr = qr[qr.split == "test"]

    good = []  # consultas con >=3 exactos con stock, y >=2 colores distintos si sirve para preferencia
    for qid, g in qr.groupby("query_id"):
        ex = g[g.esci_label == "E"].product_id.tolist()
        ok = [p for p in ex if seed[p]["stock"] >= 10]
        if len(ok) >= 3 and len(g["query"].iloc[0]) < 40 and g["query"].iloc[0] not in exclude:
            good.append((qid, g["query"].iloc[0], ex, ok))
    order = rng.permutation(len(good))
    pool = [good[i] for i in order]
    it = iter(pool)

    def nxt():
        qid, q, ex, ok = next(it)
        pmax = max(seed[p]["price"] for p in ex)
        return {"query_id": int(qid), "q": q, "acceptable": ex, "pmax": pmax,
                "pmin": min(seed[p]["price"] for p in ex)}

    def budget_ok(c, n):  # holgado: siempre alcanza para cualquier exacto
        return int(math.ceil(c["pmax"] * n * 1.2))

    S = []

    def add(kind, voice, c, turns, setup=None, **chk):
        S.append({"id": f"S{len(S) + 1:02d}", "kind": kind, "voice": voice, "query": c["q"],
                  "acceptable": c["acceptable"], "setup": setup or [], "turns": turns, "check": chk})

    confirm = {"user": "Sí, confírmalo."}

    for voice in (True, False, False):  # compra simple
        c = nxt(); n = int(rng.integers(2, 5)); b = budget_ok(c, n)
        add("buy", voice, c, [{"user": f"Hola, busco {c['q']}. Quiero {n} unidades y no quiero gastar más de {b} euros."},
                              confirm], qty=n, budget=b)
    for voice in (True, False):  # ambigua -> aclaracion
        c = nxt(); g = GENERIC[int(rng.integers(len(GENERIC)))]; n = 2; b = budget_ok(c, n)
        add("ambiguous", voice, c, [{"user": f"Quiero comprar {g}."},
                                    {"user": f"Me refería a {c['q']}, {n} unidades, máximo {b} euros."}, confirm],
            qty=n, budget=b, first_turn_must_ask=True)
    for voice in (True, False, False):  # cambio de cantidad conservando presupuesto
        c = nxt(); n1, n2 = 2, int(rng.integers(3, 5)); b = budget_ok(c, n2)
        add("change_qty", voice, c, [{"user": f"Busco {c['q']}, {n1} unidades, presupuesto máximo {b} euros."},
                                     {"user": f"Mejor cambia la cantidad a {n2}."}, confirm], qty=n2, budget=b)
    for voice in (False, False):  # cambio de preferencia (color verificable en el catalogo)
        c = nxt(); colors = [str(cat.loc[p].product_color) for p in c["acceptable"] if seed[p]["stock"] >= 5
                             and str(cat.loc[p].product_color) not in ("None", "nan", "")]
        col = colors[0] if colors else None; n = 1; b = budget_ok(c, n)
        add("change_pref", voice, c, [{"user": f"Busco {c['q']}, una unidad, máximo {b} euros."},
                                      {"user": f"Prefiero que sea de color {col}." if col else "Prefiero el más barato."},
                                      confirm], qty=n, budget=b, color=col)
    for voice in (True, False, False):  # agotado: todos los exactos sin stock
        c = nxt(); n = 2; b = budget_ok(c, n)
        add("out_of_stock", voice, c, [{"user": f"Necesito {n} unidades de {c['q']}, máximo {b} euros."},
                                       {"user": "Sí, si hay una alternativa parecida la quiero."}, confirm],
            setup=[[p, 0] for p in c["acceptable"]], qty=n, budget=b)
    for voice in (True, False, False):  # cambia stock entre cotizacion y confirmacion
        c = nxt(); n = 2; b = budget_ok(c, n)
        add("stock_change", voice, c, [{"user": f"Quiero {n} de {c['q']}, máximo {b} euros."},
                                       {"user": "Sí, confírmalo.", "before": [["cart_stock", 0]]},
                                       {"user": "Sí, confírmalo otra vez."}], qty=n, budget=b)
    for voice in (True, False):  # correccion de transcripcion antes de comprar
        c = nxt(); n = 2; wrong = 12; b = budget_ok(c, wrong)
        add("correction", voice, c, [{"user": f"Busco {c['q']}, quiero {wrong} unidades, máximo {b} euros."},
                                     {"user": f"Perdona, no dije {wrong}, dije {n}."}, confirm], qty=n, budget=b)
    c = nxt()  # el usuario se arrepiente
    add("decline", False, c, [{"user": f"Busco {c['q']}, una unidad."}, {"user": "No, mejor no compres nada."}],
        qty=1, budget=None)
    c = nxt(); b = max(1, int(c["pmin"] * 0.4))  # presupuesto imposible
    add("budget_tight", False, c, [{"user": f"Busco {c['q']}, una unidad, máximo {b} euros."}, confirm],
        qty=1, budget=b)
    assert len(S) == 20, len(S)
    assert sum(s["voice"] for s in S) == 6
    json.dump(S, open(f"{data}/{out}", "w"), ensure_ascii=False, indent=1)
    return S


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--validacion":  # otras consultas de test, sin solaparse con las de desarrollo
        dev = {s["query"] for s in json.load(open("data/scenarios.json"))}
        S = build(seed=11, out="scenarios_val.json", exclude=dev)
    else:
        S = build()
    for s in S:
        print(s["id"], s["kind"], "VOZ" if s["voice"] else "txt", s["query"], s["check"].get("qty"), s["check"].get("budget"))
