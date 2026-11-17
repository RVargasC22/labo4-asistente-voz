"""Ejecuta escenarios contra el agente y calcula exito, incumplimientos y tiempos."""
import time


def apply_setup(store, setup):
    for pid, n in setup:
        store.set_stock(pid, n)


def run_scenario(make_agent, sc, user_texts=None, turn_fn=None):
    """make_agent() -> (agent, store) con tienda restaurada. user_texts: transcripciones (voz) o None.
    turn_fn(i, agent)->respuesta: reemplaza agent.turn (usado por el pipeline de voz)."""
    agent, store = make_agent()
    apply_setup(store, sc["setup"])
    replies, times = [], []
    for i, t in enumerate(sc["turns"]):
        for kind, val in t.get("before", []):
            if kind == "cart_stock":
                for pid in list(store.carts.get(agent.cart, {})):
                    store.set_stock(pid, val)
        text = user_texts[i] if user_texts else t["user"]
        t0 = time.perf_counter()
        replies.append(turn_fn(i, agent) if turn_fn else agent.turn(text))
        times.append(time.perf_counter() - t0)
    res = {"id": sc["id"], "kind": sc["kind"], "replies": replies, "turn_s": times,
           "orders": store.orders, "log": agent.log, "final_store": store.snapshot()}
    res["violations"] = violations(store, sc)
    res["ok"] = check(store, agent, sc, replies)
    return res


def violations(store, sc):
    """Incumplimientos de restricciones duras, independientes del tipo de escenario."""
    v, b, acc = [], sc["check"].get("budget"), set(sc["acceptable"])
    if any(n < 0 for n in store.stock.values()):
        v.append("stock_negativo")
    for o in store.orders:
        if b is not None and o["total"] > b + 1e-9:
            v.append(f"{o['order_id']}:presupuesto")
        for ln in o["lines"]:
            if sc["kind"] != "out_of_stock" and ln["product_id"] not in acc and sc["kind"] in ("buy",):
                v.append(f"{o['order_id']}:producto_no_exacto")
    if len(store.orders) > 1:
        v.append("pedidos_multiples")
    return v


def check(store, agent, sc, replies):
    k, c, o = sc["kind"], sc["check"], store.orders
    acc = set(sc["acceptable"])
    if sc["setup"] == [] and any(n < 0 for n in store.stock.values()):
        return False
    if violations(store, sc):
        return False
    if k in ("buy", "ambiguous", "change_qty", "change_pref", "correction"):
        if k == "ambiguous" and "?" not in replies[0]:
            return False  # debia pedir aclaracion antes de actuar
        if k == "ambiguous" and any(e[0] == 1 and e[1] in ("set_quantity", "quote") for e in agent.log):
            return False
        if len(o) != 1:
            return False
        lines = o[0]["lines"]
        if len(lines) != 1 or lines[0]["qty"] != c["qty"] or lines[0]["product_id"] not in acc:
            return False
        if k == "change_pref" and c.get("color"):
            col = str(agent.cat.loc[lines[0]["product_id"]].product_color)
            return c["color"].lower() in col.lower()
        return True
    if k == "out_of_stock":  # exito: no hay pedido, o es una alternativa con stock real
        return all(ln["product_id"] not in acc for ord_ in o for ln in ord_["lines"])
    if k == "stock_change":  # sin pedido del producto agotado y sin duplicados
        return len(o) == 0
    if k == "decline":
        return len(o) == 0
    if k == "budget_tight":
        return len(o) == 0
    return False


def summarize(results):
    n = len(results)
    return {"escenarios": n, "correctos_pct": 100 * sum(r["ok"] for r in results) / n,
            "incumplimientos": sum(len(r["violations"]) for r in results),
            "fallos": [r["id"] for r in results if not r["ok"]]}
