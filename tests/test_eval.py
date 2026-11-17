import sys, json
sys.path.insert(0, "src")
import pandas as pd
from agent import Agent
from store import Store
from evaluate import run_scenario, summarize

cat = pd.read_parquet("data/catalog.parquet")
SC = json.load(open("data/scenarios.json"))


class Script:
    def __init__(self, o): self.o = list(o)
    def generate(self, m): return self.o.pop(0)


class Fixed:
    def __init__(self, ids): self.ids = ids
    def search(self, q, k=5): return self.ids[:k]


def J(**k): return json.dumps(k)


def test_stock_change_scenario_with_oracle_llm():
    sc = next(s for s in SC if s["kind"] == "stock_change")
    def make():
        st = Store("data/store_seed.json")
        a = Agent(Script([J(tool="search", args={"query": "x"}), J(tool="set_quantity", args={"ref": "P1", "qty": 2}),
                          J(tool="quote", args={}), J(say="Total. ¿Confirmas?"),
                          J(tool="confirm_order", args={}), J(say="No hay stock."),
                          J(say="Sigue sin stock.")]), st, Fixed(sc["acceptable"]), cat)
        return a, st
    r = run_scenario(make, sc)
    assert r["ok"] and not r["orders"] and not r["violations"]
    assert summarize([r])["correctos_pct"] == 100
