import sys, pytest
sys.path.insert(0, "src")
from store import Store, StoreError

SEED = "data/store_seed.json"


def pick(s, n):
    return [p for p, v in s.stock.items() if v >= 5][:n]


def test_reset_reproducible():
    s = Store(SEED); a = s.snapshot(); p = pick(s, 1)[0]
    s.set_qty("c", p, 2); s.confirm(s.quote("c")["quote_id"], "k")
    s.reset()
    assert s.snapshot() == a


def test_confirm_updates_stock_and_is_idempotent():
    s = Store(SEED); p = pick(s, 1)[0]; st = s.stock[p]
    s.set_qty("c", p, 2)
    q = s.quote("c")
    r1 = s.confirm(q["quote_id"], "k1"); r2 = s.confirm(q["quote_id"], "k1")
    assert r1["ok"] and not r1["duplicate"] and r2["duplicate"]
    assert len(s.orders) == 1 and s.stock[p] == st - 2


def test_stock_change_blocks_atomically():
    s = Store(SEED); a, b = pick(s, 2)
    s.set_qty("c", a, 2); s.set_qty("c", b, 3)
    q = s.quote("c"); before = dict(s.stock)
    s.set_stock(b, 1)
    r = s.confirm(q["quote_id"], "k")
    assert not r["ok"] and r["issues"][0]["type"] == "stock"
    assert s.stock[a] == before[a] and not s.orders  # sin cambios parciales


def test_budget_and_validation():
    s = Store(SEED); p = pick(s, 1)[0]
    s.set_qty("c", p, 3)
    q = s.quote("c", budget=0.01)
    assert not q["ok"] and q["issues"][0]["type"] == "budget"
    assert not s.confirm(q["quote_id"], "k")["ok"]
    for bad in (-1, 100, 1.5, "2", True):
        with pytest.raises(StoreError):
            s.set_qty("c", p, bad)
    with pytest.raises(StoreError):
        s.set_qty("c", "NOPE", 1)
