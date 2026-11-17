"""Genera notebooks/Labo4_asistente_voz.ipynb (ejecutar: python tools/make_notebook.py)."""
import nbformat as nbf

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Laboratorio 4 – Asistente de compras por voz (ESCI, español)
Pipeline por turnos: audio → Whisper (ASR) → agente (RAG híbrido + Qwen2.5-7B 4 bits + herramientas validadas en código) → XTTS-v2 (voz clonada).
Todo corre en esta sesión de Colab (GPU T4), sin APIs externas de inferencia.

**Antes de ejecutar**
1. Entorno de ejecución → GPU T4.
2. Subir `labo4_src.zip` a `/content`, o dejar que la celda clone el repo privado (pide un token de GitHub de solo lectura).
3. Subir la grabación de referencia de la voz autorizada como `/content/ref.wav` (6-15 s, mono) y escribir su transcripción en `REF_TEXT`.
4. Los 6 escenarios de voz usan audios `/content/audio_in/S##_t#.wav`. Si no existen, `SYNTH_USER_AUDIO=True` los sintetiza con la voz de referencia (solo como relleno: sustituir por grabaciones reales para la entrega).
""")
code("""
import os, getpass
if os.path.exists('/content/labo4_src.zip'):
    !unzip -oq /content/labo4_src.zip -d /content
else:  # repo privado: pedir token de GitHub (solo lectura) sin guardarlo en el notebook
    tok = getpass.getpass('GitHub token (read-only): ')
    !git clone -q https://{tok}@github.com/RVargasC22/labo4-asistente-voz.git /content/repo
    !cp -r /content/repo/src /content/repo/tools /content/
!pip -q install rank_bm25 jiwer coqui-tts speechbrain bitsandbytes accelerate "transformers>=4.45" sentence-transformers pyarrow rapidfuzz
import sys, platform, torch; sys.path.insert(0, '/content/src')
print(platform.python_version(), torch.__version__, torch.cuda.get_device_name(0))
""")
md("## 1. Datos: catálogo, consultas y tienda (semilla fija)")
code("""
!mkdir -p /content/data_raw/esci /content/data
base = 'https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/'
for f in ['examples', 'products']:
    !curl -sL -o /content/data_raw/esci/shopping_queries_dataset_{f}.parquet {base}shopping_queries_dataset_{f}.parquet
!python /content/src/prepare_data.py --raw /content/data_raw/esci --out /content/data
!python /content/src/scenarios.py
""")
md("## 2. Recuperación: BM25 vs denso vs híbrido, y ablación (descripción del producto)\nLa configuración (RRF) se fija con las consultas de desarrollo (`train`); `test` se evalúa una sola vez.")
code("""
import pandas as pd, numpy as np, time, json
from retrieval import Lexical, Dense, rrf, evaluate
cat = pd.read_parquet('/content/data/catalog.parquet'); qr = pd.read_parquet('/content/data/qrels.parquet')
T = {}
t = time.time(); lex = Lexical(cat); T['indice_bm25_s'] = time.time() - t
t = time.time(); den = Dense(cat); T['carga+indice_denso_s'] = time.time() - t
t = time.time(); den_nd = Dense(cat, desc=False); T['indice_denso_sin_desc_s'] = time.time() - t
methods = {'BM25 (base)': lex.scores, 'Denso e5': den.scores,
           'Híbrido RRF': lambda q: rrf(lex.scores(q), den.scores(q)),
           'Ablación: denso sin descripción': den_nd.scores}
rows = []
for n, fn in methods.items():
    d, nd = evaluate(fn, lex.ids, qr, 'train'); te, nt = evaluate(fn, lex.ids, qr, 'test')
    rows.append({'método': n, 'nDCG@10 dev': round(d, 4), 'nDCG@10 test': round(te, 4)})
pd.DataFrame(rows)
""")
md("## 3. Carga de modelos (tiempos de carga inicial y memoria)")
code("""
from llm import LocalLLM
from voice import ASR, TTS
from agent import Agent, Searcher
from store import Store
REF_WAV, REF_TEXT = '/content/ref.wav', 'ESCRIBIR AQUI LA TRANSCRIPCION DE LA GRABACION DE REFERENCIA'
mem = lambda: round(torch.cuda.memory_allocated() / 2**30, 2)
L = {}
t = time.time(); llm = LocalLLM('Qwen/Qwen2.5-7B-Instruct'); L['llm_s'] = time.time() - t; L['gpu_gb_tras_llm'] = mem()
t = time.time(); asr = ASR('openai/whisper-medium', brands=cat.product_brand.dropna().astype(str).unique()); L['asr_s'] = time.time() - t; L['gpu_gb_tras_asr'] = mem()
t = time.time(); tts = TTS(REF_WAV); L['tts_s'] = time.time() - t; L['gpu_gb_tras_tts'] = mem()
L['embeddings+indices_s'] = T['carga+indice_denso_s']
print(L)
""")
md("""
**Justificación de modelos y memoria (T4, 15 GB):** Qwen2.5-7B-Instruct en NF4 (~5.5 GB) por su seguimiento de formato JSON y español; Whisper-medium fp16 (~1.5 GB) por exactitud en cantidades; XTTS-v2 (~2 GB) por clonación con una sola referencia en español; e5-base fp16 (<0.5 GB). Todos caben simultáneamente, por lo que no hay recargas entre turnos.
""")
code("""
searcher = Searcher(lex, den, rrf)
def make_agent_factory(searcher=searcher, use_state=True):
    def make():
        st = Store('/content/data/store_seed.json')
        return Agent(llm, st, searcher, cat, use_state=use_state), st
    return make
SC = json.load(open('/content/data/scenarios.json'))
""")
md("## 4. Escenarios de texto (transcripción correcta) – sistema completo")
code("""
from evaluate import run_scenario, summarize
# calentamiento (primera inferencia compila kernels CUDA): se mide aparte
t = time.time(); run_scenario(make_agent_factory(), SC[0]); T['primer_escenario_s'] = time.time() - t
R_text = [run_scenario(make_agent_factory(), s) for s in SC]
print(summarize(R_text))
pd.DataFrame([{'id': r['id'], 'tipo': r['kind'], 'ok': r['ok'], 'viol': r['violations'], 'pedidos': len(r['orders']),
               's_por_turno': round(np.mean(r['turn_s']), 1)} for r in R_text])
""")
md("## 5. Ablaciones sobre los mismos 20 escenarios\n(a) recuperación solo BM25; (b) sin estado inyectado en el prompt (el LLM solo ve el historial).")
code("""
R_bm25 = [run_scenario(make_agent_factory(Searcher(lex, den, rrf, mode='bm25')), s) for s in SC]
R_nostate = [run_scenario(make_agent_factory(use_state=False), s) for s in SC]
abl = pd.DataFrame([{'variante': n, **{k: v for k, v in summarize(R).items() if k != 'fallos'}, 'fallos': summarize(R)['fallos']}
                    for n, R in (('completo (RRF + estado)', R_text), ('solo BM25', R_bm25), ('sin estado en prompt', R_nostate))])
abl
""")
md("## 6. Escenarios por voz (6 escenarios; ≥3 multi-turno)\nAudios de entrada, transcripciones, operaciones y estado final de la tienda quedan en `/content/salida`.")
code("""
import os, shutil
from pipeline import run_voice_scenario
SYNTH_USER_AUDIO = True
os.makedirs('/content/audio_in', exist_ok=True); os.makedirs('/content/salida/audio_out', exist_ok=True)
VS = [s for s in SC if s['voice']]
if SYNTH_USER_AUDIO:  # relleno: sustituir por grabaciones reales
    for s in VS:
        for i, t in enumerate(s['turns']):
            p = f"/content/audio_in/{s['id']}_t{i+1}.wav"
            if not os.path.exists(p): tts.speak(t['user'], p)
R_voice = [run_voice_scenario(make_agent_factory(), s, asr, tts, '/content/audio_in', '/content/salida/audio_out', REF_TEXT) for s in VS]
print(summarize(R_voice))
""")
md("### 6.1 Errores de ASR: se repiten los casos de voz con la transcripción correcta (sección 4)")
code("""
by_id = {r['id']: r for r in R_text}
rows = []
for r in R_voice:
    for v in r['voice_turns']:
        rows.append({'id': r['id'], 'turno': v['turn'], 'wer': round(v['wer'], 3), 'error_cantidad/importe': v['num_error'],
                     'error_producto': v.get('producto_error'), 'transcripción': v['transcript']})
asr_df = pd.DataFrame(rows); display(asr_df)
pd.DataFrame([{'id': r['id'], 'ok_voz': r['ok'], 'ok_texto_correcto': by_id[r['id']]['ok']} for r in R_voice])
""")
md("### 6.2 Tiempos por etapa (fin de grabación → respuesta reproducible)\nLa primera consulta incluye calentamiento; la carga inicial de modelos se reporta aparte (sección 3).")
code("""
tm = pd.DataFrame([{**{k: v[k] for k in ('asr', 'retrieval', 'llm', 'agent_other', 'tts', 'total')}, 'id': r['id'], 'turno': v['turn']}
                   for r in R_voice for v in r['voice_turns']])
display(tm.round(2)); print('mediana (s):'); print(tm[['asr', 'retrieval', 'llm', 'agent_other', 'tts', 'total']].median().round(2))
""")
md("### 6.3 Calidad de la voz: inteligibilidad (WER de Whisper sobre la salida TTS) y semejanza con la referencia (ECAPA)")
code("""
from metrics import SpeakerSim, wer
from voice import spoken_text
ss = SpeakerSim()
q = []
for r in R_voice:
    for v in r['voice_turns']:
        hyp = asr.transcribe(v['audio_out'])
        q.append({'id': r['id'], 'turno': v['turn'], 'WER_tts': round(wer(spoken_text(v['reply']), hyp), 3),
                  'sim_voz': round(ss.sim(REF_WAV, v['audio_out']), 3), 'texto_distinto_de_ref': v['text_differs_from_ref']})
display(pd.DataFrame(q)); print(pd.DataFrame(q)[['WER_tts', 'sim_voz']].mean())
""")
md("## 7. Guardado de evidencias y análisis de aciertos / fallos")
code("""
out = {'cargas': L, 'tiempos': T, 'texto': R_text, 'bm25': R_bm25, 'sin_estado': R_nostate, 'voz': R_voice}
json.dump(out, open('/content/salida/resultados.json', 'w'), ensure_ascii=False, indent=1, default=str)
shutil.copytree('/content/audio_in', '/content/salida/audio_in', dirs_exist_ok=True)
!cd /content && zip -qr salida.zip salida
for r in R_text:
    print(f"{'OK ' if r['ok'] else 'FALLO'} {r['id']} {r['kind']} viol={r['violations']} pedidos={len(r['orders'])}")
    if not r['ok']: print('    ', r['replies'])
""")
md("""
### Análisis de aciertos y fallos
Cifras de referencia (ejecución local, GPU de 16 GB, código actual): texto 75 % (15/20), voz 83 % (5/6), 0 violaciones. Ablaciones: solo BM25 70 %, sin estado en el prompt 70 %. Con 20 escenarios una diferencia de uno o dos escenarios es ruido, así que las ablaciones no son concluyentes. Antes de las últimas mejoras (filtros de color y marca, tope de stock, campo `dentro_de_presupuesto`, regla de categoría genérica) el mismo conjunto daba 75 % en local y 70 % en Colab (T4), con 1 violación. En un segundo conjunto de 20 escenarios con consultas distintas (`python src/scenarios.py --validacion`) el código anterior daba 70 % y el actual 80 %, con 0 violaciones; ese conjunto no es independiente del todo, porque sus fallos se miraron para decidir las mejoras. Esta versión no se ha repetido en Colab. Los escenarios `out_of_stock`, `stock_change`, `decline` y `budget_tight` se superan sin pedido, así que miden más la prudencia que la compra. Las tablas describen la ejecución local con el código actual.

**Aciertos**

| Escenario | Qué hizo el agente (`r['log']`) | Por qué acierta |
|---|---|---|
| S02 `buy` | `search` → `set_quantity` P5 (mochila TOTTO infantil) × 4 → `set_budget` 811 → `quote` 204,88 €; `confirm_order` tras el "sí" | Pedido de 4 unidades de un producto aceptable, dentro del presupuesto. La elección fue del LLM: con el código anterior eligió otro producto (una mochila "adaptable a carro") y fue la única violación |
| S03 `buy` | Mismo flujo, 2 unidades, total 89,32 € | Cantidad y presupuesto respetados; total dicho = cotizado |
| S07 `change_qty` | Informó de que el primer producto no tenía stock y propuso otro; al pedir 4 unidades hizo `set_quantity` y `quote` (533,28 €) y confirmó tras el "sí" | Cantidad final correcta, producto aceptable, y la cotización se rehízo al cambiar el carrito |
| S08 `change_qty` | `search` → P4 (archivador Oxford, 4,80 €) × 2; al pedir 4, nueva cotización de 19,20 € y confirmación | Eligió un producto con stock suficiente; con el código anterior eligió uno con 1 unidad y no hubo pedido |
| S10 `change_pref` | Cotizó un producto de 12,15 €; ante "color Blanco" llamó a `search` con `color: "Blanco"` y el filtro devolvió productos blancos; cambió el carrito (el bloqueo de varios productos le hizo quitar el anterior), cotizó 31,22 € y confirmó | **El filtro de color es del código**: antes la búsqueda no devolvía ningún producto blanco y el escenario acababa sin pedido |
| S20 `budget_tight` | `quote` devuelve la incidencia de presupuesto (11,70 € > 3 €); el usuario dice "sí, confírmalo" y el LLM llama a `confirm_order` | **El código lo rechaza** ("no hay cotización vigente"): la restricción de presupuesto no depende de que el LLM obedezca |

**Fallos**

| Escenario | Qué hizo el agente | Causa | Limitación |
|---|---|---|---|
| S01 `buy` | Buscó "cable thunderbolt" (el usuario dijo "la cie") y puso 3 unidades de un disco LaCie de 138,55 € → cotización 415,65 € > 270 €. Lo explicó, pero no buscó otro producto; el "sí" del turno 2 lo rechazó el código | Eligió un producto demasiado caro para el presupuesto y no propuso alternativas; con el código anterior este escenario acertaba con otro producto | El código impide el pedido, pero no corrige la elección; la decide el LLM |
| S05 `ambiguous` | Turno 1 bien: preguntó qué teléfono. Turno 2, ante "cama nido y armario puente, 2 unidades, máximo 156 euros", preguntó si buscaba un teléfono o muebles, y en el turno 3 pidió otra vez cantidad y presupuesto que el usuario ya había dicho. No llamó a ninguna herramienta | **Efecto secundario de mi regla de categoría genérica**: pregunta de más, sin releer lo que el usuario acaba de decir | Con la regla anterior este escenario acertaba; la regla arregló el caso de ambigüedad de la validación a costa de este |
| S09 `change_pref` | Cotizó un foco de 31,60 €; ante "color Plata" llamó a `search` con el filtro y no hubo ningún producto plata entre los 40 más parecidos; lo dijo al usuario | Comportamiento honesto, pero sin pedido: el filtro solo mira los 40 productos más parecidos a la consulta | Antes decía "en color plata" de un producto "Metálico": esa afirmación falsa ya no ocurre |
| S15 `stock_change` | El producto elegido tenía 1 unidad; el tope de stock rechazó las 2 pedidas y el agente preguntó si aceptaba 1. El guion del escenario responde "sí, confírmalo" y se registró un pedido de 1 unidad (123,32 €) | El escenario está pensado para que el stock cambie después de cotizar; este flujo nunca llega a ese caso, y el criterio de éxito (sin pedido) lo cuenta como fallo | El comportamiento es defendible (el usuario aceptó 1 unidad), pero el criterio no lo contempla |
| S17 `correction` (también en voz) | Corrigió 12 → 2 unidades, cotizó 24,70 € y compró "Gible Pistola de petardos con 48 recargas", que no es el producto exacto | Eligió un producto no exacto sin avisar; en voz el ASR fue perfecto, el fallo es del agente | Sin señal de exactitud: un campo `sin_coincidir` con regla en el prompt bajó el texto a 35 % y se descartó |

**Hallazgos transversales**
1. **Importes y afirmaciones.** El LLM escribe las cifras tal como las devuelven las herramientas y `spoken_text` las convierte a palabras para el TTS (antes el LLM las deletreaba y se equivocaba). De 32 importes dichos en los 20 escenarios de texto, 31 aparecen en las salidas de las herramientas; el otro (S01, 145,65 €) es una resta correcta hecha por el LLM. Las afirmaciones falsas se redujeron con datos estructurados: `quote` devuelve `dentro_de_presupuesto` (antes S17 decía que 148,20 € excedía un presupuesto de 1446 €) y el filtro y la guarda de color evitan decir "color plata" de un producto "Metálico". Sigue habiendo afirmaciones sin comprobar: en S19 el agente dice que quita el producto del carrito pero solo llama a `view_cart`, y el criterio solo mira que no haya pedido.
2. **Límite de pasos.** Con el límite en 8 no hay ningún mensaje de rescate ("no he podido procesar eso") en los 20 escenarios de texto; con 6 pasos aparecían en 2.
3. **Dónde ayuda el código y dónde no.** Las guardas en código (presupuesto dicho por el usuario, confirmación solo tras ver la cotización, cotización invalidada al cambiar el carrito o buscar, bloqueo con varios productos, tope de stock con alternativas, color pedido) dieron 0 violaciones en los 20 escenarios de desarrollo y en los 20 de validación. El tope de stock tuvo un fallo de diseño que se vio en la validación: el primer mensaje de error sugería "una cantidad menor" y el agente compró 1 unidad en vez de las 4 pedidas sin preguntar; ahora el mensaje le exige preguntar. Las guardas no evitan que el LLM elija mal entre candidatos válidos (S01, S17) ni que sobre-pregunte (S05).
4. **Validez de la medición.** Los 20 escenarios de desarrollo sirvieron para depurar y medir, y los de validación se usaron para decidir mejoras, de modo que ninguna cifra es una estimación limpia; falta un tercer conjunto que no se haya visto. La corrección de marcas del ASR es estrecha (arregla "Schwarz Cops" → "Schwarzkopf", S06 en voz) y no corrige errores como "cie" → "Sied". Un cambio de GPU basta para mover uno o tres escenarios (en Colab, con el código anterior, S04 y S05 fallaban y S08 acertaba).
""")



nb = nbf.v4.new_notebook(cells=C)
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbf.write(nb, "notebooks/Labo4_asistente_voz.ipynb")
print("ok", len(C), "celdas")
