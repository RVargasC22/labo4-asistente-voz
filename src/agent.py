"""Agente de compras: un solo LLM local, protocolo JSON propio, estado explicito y validaciones en codigo.

El LLM solo decide la siguiente accion; cantidades, presupuesto, stock y confirmacion se validan en codigo.
"""
import json, re
import numpy as np
from metrics import norm

MAX_STEPS = 8
HISTORY_TURNS = 8

SYSTEM = """Eres un asistente de compras por voz en español. Hablas en frases cortas y naturales, sin listas ni símbolos, porque tu respuesta se leerá en voz alta.
En cada paso respondes SOLO con un objeto JSON, sin texto adicional. Dos formas:
{"tool": "<nombre>", "args": {...}}   para ejecutar una herramienta
{"say": "<texto>"}                    para responder al usuario (termina tu turno)
Una sola acción por respuesta: espera la OBSERVACION antes de la siguiente. Ejemplos exactos:
{"tool": "search", "args": {"query": "cable thunderbolt"}}
{"tool": "set_quantity", "args": {"ref": "P1", "qty": 3}}
{"say": "Encontré un cable. ¿Quieres tres unidades?"}

Herramientas:
- search {"query": str, "color": str opcional, "brand": str opcional}: busca productos. Devuelve referencias P1, P2... Si el usuario pide un color o una marca, pásalos también en "color"/"brand": filtran los resultados.
- set_quantity {"ref": "P1", "qty": int}: fija la cantidad en el carrito (0 quita el producto).
- set_budget {"amount": number}: fija el presupuesto máximo total en euros.
- note_preference {"text": str}: guarda una preferencia del usuario (color, marca, talla...).
- view_cart {}: muestra el carrito.
- quote {}: calcula la cotización con el presupuesto vigente.
- confirm_order {}: registra el pedido. SOLO después de que el usuario haya aceptado explícitamente la última cotización.

Orden habitual: search, set_quantity, set_budget, quote; di el total y espera la confirmación del usuario; solo entonces confirm_order.
Si el usuario cambia de producto o de variante (color, marca, talla), quita el anterior con set_quantity qty 0 antes de añadir el nuevo.
Si el usuario nombra solo una categoría muy genérica, sin marca, uso ni modelo (por ejemplo "una funda"), no busques todavía: pregunta qué necesita.
Si el usuario aclara o corrige lo que quiere, haz una nueva búsqueda con la nueva descripción en vez de seguir con el producto anterior.
"say" termina tu turno: nunca anuncies lo que vas a hacer ("Buscando..."); usa primero las herramientas y responde con el resultado.

Ejemplo de conversación completa (referencias y precios inventados):
Usuario: Quiero 2 mochilas azules, máximo 50 euros.
{"tool": "search", "args": {"query": "mochila azul"}}
OBSERVACION: {"resultados": [{"ref": "P1", "titulo": "Mochila escolar azul", "precio": 20.0, "stock": 5}]}
{"tool": "set_quantity", "args": {"ref": "P1", "qty": 2}}
OBSERVACION: {"carrito": [{"ref": "P1", "qty": 2, "precio": 20.0}]}
{"tool": "set_budget", "args": {"amount": 50}}
OBSERVACION: {"presupuesto": 50.0}
{"tool": "quote", "args": {}}
OBSERVACION: {"ok": true, "total": 40.0, "issues": []}
{"say": "Dos mochilas escolares azules, total 40 euros. ¿Confirmas el pedido?"}
Usuario: Sí, confírmalo.
{"tool": "confirm_order", "args": {}}
OBSERVACION: {"ok": true}
{"say": "Pedido confirmado."}
Reglas:
- Usa únicamente datos devueltos por las herramientas. Si un atributo no aparece, di que no lo sabes.
- Si falta información relevante (producto, cantidad, talla ambigua) o hay varias opciones razonables, pregunta con "say" antes de actuar.
- Si cambian cantidades o preferencias, conserva el resto de restricciones.
- Tras una cotización, di el total y pide confirmación; no confirmes tú solo.
- Si una herramienta devuelve un problema (stock, presupuesto), explícalo y propone una alternativa buscando otro producto, o di que no hay solución.
- Escribe los precios con cifras tal como los devuelven las herramientas, por ejemplo "12,50 euros"; no los conviertas a palabras ni redondees.
"""

# decenas/centenas/mil dichos con palabras: no se pueden cotejar con digitos
SPOKEN_AMOUNT = {"cien", "ciento", "mil", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta",
                 "ochenta", "noventa", "doscientos", "trescientos", "cuatrocientos", "quinientos", "seiscientos",
                 "setecientos", "ochocientos", "novecientos"}

TOOLS = {"search", "set_quantity", "set_budget", "note_preference", "view_cart", "quote", "confirm_order"}


_DEC = json.JSONDecoder()


def _has(value, wanted):
    """El atributo del catalogo contiene lo pedido (sin tildes ni genero: "blanca" casa con "Blanco")."""
    w = norm(str(wanted))
    return len(w) >= 3 and (w in norm(str(value)) or w[:5] in norm(str(value)))


def parse_action(text):
    """Primera accion valida en la salida del LLM, normalizada a {"tool","args"} o {"say"}; None si no hay.
    Toma solo el primer objeto JSON (el LLM a veces encadena varios) y acepta la forma abreviada {"search": {...}}."""
    i = text.find("{")
    while i != -1:
        try:
            d, _ = _DEC.raw_decode(text, i)
        except json.JSONDecodeError:
            d = None
        if isinstance(d, dict):
            if "say" in d or d.get("tool") in TOOLS:
                return d
            k = next(iter(d), None)
            if k in TOOLS and isinstance(d[k], dict):
                return {"tool": k, "args": d[k]}
        i = text.find("{", i + 1)
    return None


class Agent:
    def __init__(self, llm, store, searcher, catalog, cart="main", use_state=True):
        """llm.generate(messages)->str; searcher.search(query,k)->[product_id]; catalog: DataFrame.
        use_state=False: ablacion, no se inyecta el ESTADO ACTUAL en el prompt (solo historial)."""
        self.llm, self.store, self.searcher, self.use_state = llm, store, searcher, use_state
        self.cat = catalog.set_index("product_id")
        self.cart = cart
        self.reset_state()
        self.log = []  # operaciones ejecutadas: (turno, herramienta, args, resultado)

    def reset_state(self):
        self.s = {"turn": 0, "budget": None, "prefs": [], "refs": {}, "ref_n": 0,
                  "pending": None, "multi_ok": None, "last": {}, "history": []}

    # ---------------- herramientas ----------------
    def _ref(self, pid):
        for r, p in self.s["refs"].items():
            if p == pid:
                return r
        self.s["ref_n"] += 1
        r = f"P{self.s['ref_n']}"
        self.s["refs"][r] = pid
        return r

    def _said_amount(self, amt):
        """El importe debe provenir del usuario: aparece en sus mensajes (digitos) o lo dijo con palabras."""
        said = " ".join(m["content"] for m in self.s["history"]
                        if m["role"] == "user" and not m["content"].startswith("OBSERVACION"))
        nums = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", said)]
        return any(abs(n - amt) < 1e-6 for n in nums) or bool(SPOKEN_AMOUNT & set(norm(said).split()))

    def _said_color(self):
        """Ultimo color que pidio el usuario con la forma "color X" (None si no dijo ninguno)."""
        said = norm(" ".join(m["content"] for m in self.s["history"]
                             if m["role"] == "user" and not m["content"].startswith("OBSERVACION")))
        c = re.findall(r"\bcolor (?:de )?([a-z]+)", said)
        return c[-1] if c else None

    def _view(self, pid):
        r = self.cat.loc[pid]
        info = self.store.info(pid)
        d = {"ref": self._ref(pid), "titulo": str(r.product_title)[:110], "precio": info["price"],
             "stock": info["stock"]}
        for k, col in (("marca", "product_brand"), ("color", "product_color")):
            if r[col] and str(r[col]) != "None":
                d[k] = str(r[col])
        return d

    def _cart_view(self):
        c = self.store.carts.get(self.cart, {})
        return [{"ref": self._ref(p), "qty": q, "titulo": str(self.cat.loc[p].product_title)[:60],
                 "precio": self.store.info(p)["price"]} for p, q in c.items()]

    def run_tool(self, name, a):
        s = self.s
        try:
            if name == "search":
                color, brand = a.get("color"), a.get("brand")
                ids = self.searcher.search(str(a["query"]), 40 if (color or brand) else 5)
                s["pending"] = None  # buscar alternativas deja sin vigencia la cotizacion que el usuario vio
                if color or brand:
                    ids = [p for p in ids if (not color or _has(self.cat.loc[p].product_color, color))
                           and (not brand or _has(self.cat.loc[p].product_brand, brand))]
                    if not ids:
                        return {"resultados": [], "aviso": "ninguno de los productos mas parecidos tiene ese color o marca; "
                                                           "dilo al usuario y propon quitar el filtro"}
                res = [self._view(p) for p in ids[:5]]
                s["last"] = {r["ref"]: r["stock"] for r in res}
                return {"resultados": res}
            if name == "set_quantity":
                pid = s["refs"].get(a.get("ref"))
                if pid is None:
                    return {"error": "referencia desconocida; busca primero"}
                stock = self.store.info(pid)["stock"]
                if a["qty"] > stock:
                    otros = [r for r, st in s["last"].items() if st >= a["qty"] and r != a["ref"]]
                    return {"error": f"solo hay {stock} en stock de este producto y el usuario quiere {a['qty']}: "
                                     + (f"elige otro producto con stock suficiente ({', '.join(otros)}) o " if otros else "")
                                     + "pregunta al usuario si acepta menos unidades; no cambies la cantidad por tu cuenta"}
                self.store.set_qty(self.cart, pid, a["qty"])
                s["pending"] = None  # el carrito cambio: la cotizacion previa ya no vale
                return {"carrito": self._cart_view()}
            if name == "set_budget":
                amt = float(a["amount"])
                if amt <= 0:
                    return {"error": "presupuesto invalido"}
                if not self._said_amount(amt):
                    return {"error": "el usuario no ha dicho ese presupuesto; pregunta cuanto quiere gastar"}
                s["budget"], s["pending"] = amt, None
                return {"presupuesto": amt}
            if name == "note_preference":
                s["prefs"].append(str(a["text"])[:100])
                return {"preferencias": s["prefs"]}
            if name == "view_cart":
                return {"carrito": self._cart_view(), "presupuesto": s["budget"]}
            if name == "quote":
                cart = dict(self.store.carts.get(self.cart, {}))
                if len(cart) > 1 and s["multi_ok"] != cart:
                    s["multi_ok"] = cart  # repetir quote con el mismo carrito confirma que son varios productos a proposito
                    return {"error": f"el carrito tiene {len(cart)} productos distintos; si el usuario cambio de producto, "
                                     "quita el anterior con set_quantity qty 0 y cotiza de nuevo; si quiere varios, cotiza otra vez"}
                color = self._said_color()
                for p in cart:
                    if color and not _has(self.cat.loc[p].product_color, color):
                        return {"error": f"el usuario pidio color {color} y {self._ref(p)} es de color "
                                         f"{self.cat.loc[p].product_color}; busca con el filtro color o dile que no hay"}
                q = self.store.quote(self.cart, s["budget"])
                if s["budget"] is not None:
                    q["dentro_de_presupuesto"] = q["total"] <= s["budget"]
                if q["ok"]:
                    s["pending"] = {"quote_id": q["quote_id"], "turn": s["turn"],
                                    "cart": dict(self.store.carts[self.cart])}
                return q
            if name == "confirm_order":
                p = s["pending"]
                if p is None:
                    return {"error": "no hay cotizacion vigente; usa quote primero"}
                if p["turn"] >= s["turn"]:
                    return {"error": "el usuario aun no ha visto esta cotizacion; pide su confirmacion"}
                if p["cart"] != self.store.carts.get(self.cart):
                    s["pending"] = None
                    return {"error": "el carrito cambio desde la cotizacion; cotiza de nuevo"}
                r = self.store.confirm(p["quote_id"], "idem-" + p["quote_id"])
                if r["ok"]:
                    s["pending"] = None
                return r
        except (KeyError, ValueError, TypeError) as e:
            return {"error": f"argumentos invalidos: {e!r}"}
        except Exception as e:  # StoreError
            return {"error": str(e)}
        return {"error": f"herramienta desconocida: {name}"}

    # ---------------- bucle ----------------
    def _state_text(self):
        s = self.s
        return ("ESTADO ACTUAL: " + json.dumps({"presupuesto": s["budget"], "preferencias": s["prefs"],
                "carrito": self._cart_view(), "cotizacion_vigente": bool(s["pending"])},
                ensure_ascii=False))

    def turn(self, user_text):
        s = self.s
        s["turn"] += 1
        s["history"].append({"role": "user", "content": user_text})
        msgs = [{"role": "system", "content": SYSTEM + ("\n" + self._state_text() if self.use_state else "")}] + \
               s["history"][-HISTORY_TURNS * 4:]
        bad = 0
        for _ in range(MAX_STEPS):
            out = self.llm.generate(msgs)
            act = parse_action(out)
            if act is None and out.strip() and "{" not in out:
                act = {"say": out.strip()}  # texto plano sin JSON: el LLM habla directamente al usuario
            if act is None:
                bad += 1
                if bad > 2:
                    break
                msgs += [{"role": "assistant", "content": out},
                         {"role": "user", "content": "OBSERVACION: formato invalido. Responde con UN solo objeto JSON, por ejemplo "
                          '{"tool": "search", "args": {"query": "..."}} o {"say": "..."}.'}]
                continue
            if "say" in act:
                text = str(act["say"]).strip()
                s["history"].append({"role": "assistant", "content": json.dumps(act, ensure_ascii=False)})
                return text
            res = self.run_tool(act["tool"], act.get("args", {}) or {})
            self.log.append((s["turn"], act["tool"], act.get("args", {}), res))
            obs = json.dumps(res, ensure_ascii=False, default=str)
            msgs += [{"role": "assistant", "content": json.dumps(act, ensure_ascii=False)},
                     {"role": "user", "content": "OBSERVACION: " + obs}]
            s["history"] += msgs[-2:]
        fallback = "Perdona, no he podido procesar eso. ¿Puedes repetirlo?"
        s["history"].append({"role": "assistant", "content": json.dumps({"say": fallback}, ensure_ascii=False)})
        return fallback


class Searcher:
    """Recuperacion hibrida (denso + BM25 por RRF) sobre el catalogo; no ve etiquetas ESCI."""
    def __init__(self, lex, den, rrf_fn, mode="rrf"):
        self.lex, self.den, self.rrf, self.mode = lex, den, rrf_fn, mode

    def search(self, q, k=5):
        if self.mode == "bm25":
            sc = self.lex.scores(q)
        elif self.mode == "dense":
            sc = self.den.scores(q)
        else:
            sc = self.rrf(self.lex.scores(q), self.den.scores(q))
        return [self.lex.ids[i] for i in np.argsort(-sc)[:k]]
