"""Tienda simulada transaccional: carrito, cotizacion, confirmacion atomica e idempotente."""
import copy, json


class StoreError(Exception):
    pass


class Store:
    def __init__(self, seed_path):
        self.seed_path = seed_path
        self.reset()

    # --- restauracion reproducible -------------------------------------------------
    def reset(self):
        s = json.load(open(self.seed_path))
        self.prices = {p: round(v["price"] * 100) for p, v in s["products"].items()}  # centimos
        self.stock = {p: v["stock"] for p, v in s["products"].items()}
        self.carts, self.orders, self.quotes = {}, [], {}
        self._qn = 0

    def snapshot(self):
        return copy.deepcopy({"stock": self.stock, "carts": self.carts, "orders": self.orders})

    def set_stock(self, pid, n):
        """Para escenarios: simula cambio externo de existencias."""
        self._check_pid(pid)
        self.stock[pid] = int(n)

    # --- consulta --------------------------------------------------------------------
    def _check_pid(self, pid):
        if pid not in self.prices:
            raise StoreError(f"producto desconocido: {pid}")

    def info(self, pid):
        self._check_pid(pid)
        return {"product_id": pid, "price": self.prices[pid] / 100, "stock": self.stock[pid]}

    # --- carrito ---------------------------------------------------------------------
    def set_qty(self, cart, pid, qty):
        """Fija la cantidad (0 elimina). Valida entero 0..99."""
        self._check_pid(pid)
        if isinstance(qty, bool) or not isinstance(qty, int) or not 0 <= qty <= 99:
            raise StoreError(f"cantidad invalida: {qty!r}")
        c = self.carts.setdefault(cart, {})
        if qty == 0:
            c.pop(pid, None)
        else:
            c[pid] = qty
        return dict(c)

    def clear(self, cart):
        self.carts[cart] = {}

    # --- cotizacion ------------------------------------------------------------------
    def _evaluate(self, cart, budget_cents):
        lines, issues, total = [], [], 0
        for pid, q in self.carts.get(cart, {}).items():
            sub = self.prices[pid] * q
            total += sub
            lines.append({"product_id": pid, "qty": q, "unit": self.prices[pid] / 100, "subtotal": sub / 100})
            if self.stock[pid] < q:
                issues.append({"type": "stock", "product_id": pid, "requested": q, "available": self.stock[pid]})
        if not lines:
            issues.append({"type": "empty"})
        if budget_cents is not None and total > budget_cents:
            issues.append({"type": "budget", "total": total / 100, "budget": budget_cents / 100})
        return lines, issues, total

    def quote(self, cart, budget=None):
        b = None if budget is None else round(budget * 100)
        lines, issues, total = self._evaluate(cart, b)
        self._qn += 1
        qid = f"Q{self._qn}"
        self.quotes[qid] = {"cart": cart, "total": total, "budget": b}
        return {"quote_id": qid, "lines": lines, "total": total / 100, "issues": issues, "ok": not issues}

    # --- confirmacion ----------------------------------------------------------------
    def confirm(self, quote_id, idem_key):
        """Revalida todo; aplica pedido e inventario de forma atomica. Misma idem_key => mismo pedido."""
        for o in self.orders:
            if o["idem_key"] == idem_key:
                return {"ok": True, "duplicate": True, "order": o}
        qt = self.quotes.get(quote_id)
        if qt is None:
            raise StoreError(f"cotizacion desconocida: {quote_id}")
        cart = qt["cart"]
        lines, issues, total = self._evaluate(cart, qt["budget"])
        if total != qt["total"]:
            issues.append({"type": "price_changed", "quoted": qt["total"] / 100, "now": total / 100})
        if issues:
            return {"ok": False, "issues": issues}  # nada se modifica
        for ln in lines:  # todas las comprobaciones pasaron: aplicar
            self.stock[ln["product_id"]] -= ln["qty"]
        order = {"order_id": f"O{len(self.orders) + 1}", "idem_key": idem_key, "cart": cart,
                 "lines": lines, "total": total / 100}
        self.orders.append(order)
        self.carts[cart] = {}
        return {"ok": True, "duplicate": False, "order": order}
