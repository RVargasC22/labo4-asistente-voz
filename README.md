# Laboratorio 4 – Asistente de compras por voz (ESCI, español)

Pipeline por turnos: audio -> Whisper -> agente (RAG híbrido BM25+e5, Qwen2.5-7B 4 bits, herramientas validadas en código) -> XTTS-v2 con voz clonada.

## Contenido

- `src/prepare_data.py`: catálogo (8 433 productos), consultas (100 dev / 220 test), tienda simulada con semilla 42.
- `src/retrieval.py`: BM25, denso, RRF, nDCG@10.
- `src/store.py`: carrito, cotización, confirmación atómica e idempotente.
- `src/agent.py`, `src/llm.py`: agente con estado y validaciones en código.
- `src/scenarios.py`, `src/evaluate.py`: 20 escenarios reproducibles (6 de voz) y su evaluación.
- `src/voice.py`, `src/metrics.py`, `src/pipeline.py`: ASR (con corrección de marcas), TTS (importes dichos en palabras), métricas y tiempos por etapa.
- `tools/make_notebook.py`: genera `notebooks/Labo4_asistente_voz.ipynb` (el notebook se edita aquí, no a mano).
- `tools/run_local.py`: corre el pipeline completo en local.
- `tests/`: `python -m pytest tests` (requiere `python src/prepare_data.py` y `python src/scenarios.py` antes).

Los audios, los datos crudos de ESCI, `data/` y `labo4_src.zip` no se versionan.

## Qué hace el agente en código (no depende de que el LLM obedezca)

- El presupuesto debe haberlo dicho el usuario; `confirm_order` solo vale tras una cotización que el usuario ya vio, y cambiar el carrito o buscar la invalida.
- `quote` avisa si el carrito tiene varios productos (una vez) y devuelve `dentro_de_presupuesto`.
- `search` admite filtros opcionales `color` y `brand` sobre los 40 productos más parecidos.
- `set_quantity` rechaza cantidades mayores que el stock y propone alternativas; `quote` rechaza un producto si no tiene el color que pidió el usuario.
- Límite de 8 pasos por turno.

## Ejecutar en Colab (GPU T4)

1. Entorno de ejecución: GPU T4 (con un runtime nuevo si ya se usó otro).
2. Subir `labo4_src.zip` a `/content`, o dejar que la primera celda clone el repo (pide un token de GitHub de solo lectura).
3. Subir `ref.wav` (6-15 s, mono) a `/content` y escribir su transcripción en `REF_TEXT`.
4. Subir los audios de los 6 escenarios de voz a `/content/audio_in/S##_t#.wav`; si no existen, se sintetizan con la voz de referencia (solo como relleno).
5. Ejecutar todo. Comprobación de que cargó el código actual: en la tabla de escenarios de texto, S10 debe dar 1 pedido y S02 no debe tener violación. Si no, la sesión usa un `src/` antiguo.

Para regenerar el zip desde la raíz del repo:

```
python -c "import zipfile,glob; z=zipfile.ZipFile('labo4_src.zip','w',zipfile.ZIP_DEFLATED); [z.write(f) for f in sorted(glob.glob('src/*.py'))+['tools/make_notebook.py']]"
```

## Ejecutar en local

```
python src/prepare_data.py --raw data_raw/esci --out data   # datos ESCI en data_raw/esci
python src/scenarios.py                                     # 20 escenarios de desarrollo
python src/scenarios.py --validacion                        # otros 20, consultas distintas (data/scenarios_val.json)
HF_HUB_OFFLINE=1 python tools/run_local.py [--ablaciones] [--sim-voz]
```

`run_local.py` espera `ref.wav` y `audio_in/` en `--work` (por defecto `~/labo4_local`) y escribe `resultados.json` e `informe.txt` en `--work/salida`. `--ablaciones` duplica el tiempo; `--sim-voz` mide WER del TTS y semejanza ECAPA y necesita `speechbrain/spkrec-ecapa-voxceleb` en la caché de HuggingFace.

En redes con inspección TLS fallan las descargas de HuggingFace: descargar los modelos antes y usar `HF_HUB_OFFLINE=1`.

## Resultados de referencia

| | Texto (20) | Voz (6) | Violaciones |
|---|---|---|---|
| Local, desarrollo | 75 % | 83 % | 0 |
| Local, validación (otros 20) | 80 % | | 0 |
| Colab T4, código anterior a los filtros y al tope de stock | 70 % | 67 % | 1 |

La validación con el código anterior daba 70 %. El código actual **no se ha repetido en Colab**. Con 20 escenarios y decodificación voraz, uno o tres escenarios cambian de una GPU a otra. Los de desarrollo sirvieron para depurar y los de validación se miraron para decidir mejoras, así que ninguna cifra es una estimación limpia. El análisis de aciertos y fallos está al final del notebook.
