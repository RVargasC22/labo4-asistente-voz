import sys, json
sys.path.insert(0, "src")
import pandas as pd
from agent import Agent, parse_action
from store import Store

cat = pd.read_parquet("data/catalog.parquet")


class Script:
    def __init__(self, outs): self.outs = list(outs)
    def generate(self, msgs): return self.outs.pop(0)


class Fixed:
    def __init__(self, ids): self.ids = ids
    def search(self, q, k=5): return self.ids[:k]


def mk(outs):
    st = Store("data/store_seed.json")
    ids = [p for p, v in st.stock.items() if v >= 5][:3]
    return Agent(Script(outs), st, Fixed(ids), cat), st, ids


def J(**k): return json.dumps(k)


def test_parse():
    assert parse_action('texto {"say": "hola"} fin') == {"say": "hola"}
    assert parse_action("nada") is None
    assert parse_action('{"tool": "rm -rf"}') is None


def test_parse_tolerates_llm_shorthand_and_chained_actions():
    # salida real de Qwen2.5-7B: forma abreviada y varias acciones encadenadas; solo cuenta la primera
    chained = '{"set_budget":{"amount":270}}{"search":{"query":"cie thunderbolt"}}'
    assert parse_action(chained) == {"tool": "set_budget", "args": {"amount": 270}}
    assert parse_action('{"view_cart": {}}') == {"tool": "view_cart", "args": {}}
    assert parse_action('{"tool": "search", "args": {"query": "x"}} {"say": "y"}') == {"tool": "search", "args": {"query": "x"}}
    assert parse_action('{"nada": {"x": 1}}') is None


def test_confirm_blocked_same_turn_then_ok():
    a, st, ids = mk([
        J(tool="search", args={"query": "x"}), J(tool="set_quantity", args={"ref": "P1", "qty": 2}),
        J(tool="quote", args={}), J(tool="confirm_order", args={}),  # mismo turno: debe rechazarse
        J(say="Total X. ¿Confirmas?"),
        J(tool="confirm_order", args={}), J(say="Hecho")])
    a.turn("quiero dos")
    assert not st.orders and "aun no ha visto" in str(a.log[-1][3])
    a.turn("si")
    assert len(st.orders) == 1


def test_cart_change_invalidates_quote():
    a, st, ids = mk([
        J(tool="search", args={"query": "x"}), J(tool="set_quantity", args={"ref": "P1", "qty": 2}),
        J(tool="quote", args={}), J(say="ok?"),
        J(tool="set_quantity", args={"ref": "P1", "qty": 3}), J(tool="confirm_order", args={}), J(say="x")])
    a.turn("dos"); a.turn("mejor tres")
    assert not st.orders and "no hay cotizacion" in str(a.log[-1][3])


def test_hallucinated_ref_and_bad_qty_rejected():
    a, st, ids = mk([J(tool="set_quantity", args={"ref": "P9", "qty": 1}),
                     J(tool="search", args={"query": "x"}), J(tool="set_quantity", args={"ref": "P1", "qty": -3}),
                     J(say="ok")])
    a.turn("hola")
    assert "error" in a.log[0][3] and "error" in a.log[2][3] and not st.carts.get("main")


def test_bad_json_fallback():
    a, _, _ = mk(['{"tool": ', '{"tool": ', '{"tool": '])
    assert "repetirlo" in a.turn("hola")


def test_plain_text_is_spoken_to_user():
    a, _, _ = mk(["Dos mochilas, total 40 euros. ¿Confirmas?"])
    assert a.turn("hola") == "Dos mochilas, total 40 euros. ¿Confirmas?"


def test_budget_must_come_from_user():
    a, _, _ = mk([])
    a.s["history"].append({"role": "user", "content": "quiero 2 mochilas"})
    assert "error" in a.run_tool("set_budget", {"amount": 50})  # importe inventado
    a.s["history"].append({"role": "user", "content": "maximo 270 euros"})
    assert a.run_tool("set_budget", {"amount": 270}) == {"presupuesto": 270.0}
    a.s["history"].append({"role": "user", "content": "maximo mil cuatrocientos euros"})
    assert "error" not in a.run_tool("set_budget", {"amount": 1400})  # dicho con palabras: no verificable


def test_quote_blocks_once_when_cart_has_several_products():
    a, _, _ = mk([])
    a.run_tool("search", {"query": "x"})
    a.s["history"].append({"role": "user", "content": "maximo 500 euros"})
    a.run_tool("set_quantity", {"ref": "P1", "qty": 1}); a.run_tool("set_quantity", {"ref": "P2", "qty": 1})
    a.run_tool("set_budget", {"amount": 500})
    assert "error" in a.run_tool("quote", {})          # primer intento: avisa del cambio de producto
    assert a.run_tool("quote", {})["ok"]               # mismo carrito otra vez: varios productos a proposito
    a.run_tool("set_quantity", {"ref": "P2", "qty": 0})
    assert len(a.run_tool("quote", {})["lines"]) == 1  # carrito de un solo producto: sin bloqueo




def test_search_after_quote_invalidates_it_so_a_later_yes_cannot_confirm_it():
    a, st, _ = mk([])
    a.run_tool("search", {"query": "x"})
    a.s["history"].append({"role": "user", "content": "maximo 500 euros"})
    a.run_tool("set_quantity", {"ref": "P1", "qty": 1}); a.run_tool("set_budget", {"amount": 500})
    assert a.run_tool("quote", {})["ok"]
    a.s["turn"] += 1                                   # el usuario ve la cotizacion y responde en el turno siguiente
    a.run_tool("search", {"query": "alternativa"})     # el agente propone alternativas
    a.s["turn"] += 1                                   # el usuario dice "si, confirmalo"
    assert "error" in a.run_tool("confirm_order", {}) and not st.orders


def test_set_quantity_over_stock_is_rejected_with_alternatives():
    a, st, ids = mk([])
    a.run_tool("search", {"query": "x"})
    st.set_stock(ids[0], 1)
    r = a.run_tool("set_quantity", {"ref": "P1", "qty": 3})
    assert "solo hay 1" in r["error"] and "P2" in r["error"] and "no cambies la cantidad" in r["error"]
    assert st.carts.get("main", {}) == {}
    assert "carrito" in a.run_tool("set_quantity", {"ref": "P2", "qty": 3})


def test_search_color_filter_and_quote_color_guard():
    a, st, ids = mk([])
    colors = [str(cat.set_index("product_id").loc[p].product_color) for p in ids]
    assert "aviso" in a.run_tool("search", {"query": "x", "color": "zzzzzz"})
    r = a.run_tool("search", {"query": "x", "color": colors[0]})
    assert r["resultados"] and all(colors[0].lower()[:5] in str(cat.set_index("product_id").loc[a.s["refs"][x["ref"]]].product_color).lower()
                                   for x in r["resultados"])
    a.run_tool("search", {"query": "x"})
    a.run_tool("set_quantity", {"ref": "P1", "qty": 1})
    a.s["history"].append({"role": "user", "content": "Prefiero que sea de color Zzzzzz."})
    assert "color zzzzzz" in a.run_tool("quote", {})["error"]


def test_quote_reports_whether_it_is_within_budget():
    a, st, ids = mk([])
    a.run_tool("search", {"query": "x"})
    a.run_tool("set_quantity", {"ref": "P1", "qty": 1})
    a.s["history"].append({"role": "user", "content": "maximo 1 euro"})
    a.run_tool("set_budget", {"amount": 1})
    assert a.run_tool("quote", {})["dentro_de_presupuesto"] is False
