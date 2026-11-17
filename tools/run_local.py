"""Ejecuta el pipeline del notebook en local (modelos desde la cache de HuggingFace, sin red).

Entradas en --work: ref.wav y audio_in/S##_t#.wav. Salidas en --work/salida: resultados.json, informe.txt, audio_out/.
Uso: HF_HUB_OFFLINE=1 python tools/run_local.py [--ablaciones]
"""
import os, sys, json, time, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("COQUI_TOS_AGREED", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np, pandas as pd, torch
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--work", default=os.path.expanduser("~/labo4_local"))
ap.add_argument("--ref-text", default="Ayer por la tarde salí a caminar junto al río, mientras el sol se escondía despacio detrás de las montañas. Al volver a casa, preparé un té caliente y me puse a leer un buen libro.")
ap.add_argument("--ablaciones", action="store_true", help="ademas corre BM25 y sin-estado (duplica el tiempo)")
ap.add_argument("--sim-voz", action="store_true", help="WER del TTS y semejanza ECAPA (requiere speechbrain/spkrec-ecapa-voxceleb en cache)")
a = ap.parse_args()
OUT = f"{a.work}/salida"; os.makedirs(f"{OUT}/audio_out", exist_ok=True)
REF_WAV, AUDIO_IN, DATA = f"{a.work}/ref.wav", f"{a.work}/audio_in", f"{ROOT}/data"

from retrieval import Lexical, Dense, rrf
from llm import LocalLLM
from voice import ASR, spoken_text
from agent import Agent, Searcher
from store import Store
from evaluate import run_scenario, summarize
from pipeline import run_voice_scenario


class LocalTTS:
    """Como voice.TTS, pero carga XTTS-v2 desde la cache de HuggingFace en vez de descargarlo."""
    def __init__(self, ref_wav):
        from TTS.api import TTS as Coqui
        d = snapshot_download("coqui/XTTS-v2")
        self.t = Coqui(model_path=d, config_path=f"{d}/config.json").to("cuda")
        self.ref = ref_wav

    def speak(self, text, out_path):
        self.t.tts_to_file(text=spoken_text(text), speaker_wav=self.ref, language="es", file_path=out_path)
        return out_path


mem = lambda: round(torch.cuda.memory_allocated() / 2**30, 2)
T, L = {}, {}
cat = pd.read_parquet(f"{DATA}/catalog.parquet")
t = time.time(); lex = Lexical(cat); T["indice_bm25_s"] = time.time() - t
t = time.time(); den = Dense(cat); T["carga+indice_denso_s"] = time.time() - t
t = time.time(); llm = LocalLLM("Qwen/Qwen2.5-7B-Instruct"); L["llm_s"] = time.time() - t; L["gpu_gb_tras_llm"] = mem()
t = time.time(); asr = ASR("openai/whisper-medium", brands=cat.product_brand.dropna().astype(str).unique()); L["asr_s"] = time.time() - t; L["gpu_gb_tras_asr"] = mem()
t = time.time(); tts = LocalTTS(REF_WAV); L["tts_s"] = time.time() - t; L["gpu_gb_tras_tts"] = mem()
print(L, flush=True)

searcher = Searcher(lex, den, rrf)
def make_agent_factory(searcher=searcher, use_state=True):
    def make():
        st = Store(f"{DATA}/store_seed.json")
        return Agent(llm, st, searcher, cat, use_state=use_state), st
    return make
SC = json.load(open(f"{DATA}/scenarios.json"))

t = time.time(); run_scenario(make_agent_factory(), SC[0]); T["primer_escenario_s"] = time.time() - t
R_text = [run_scenario(make_agent_factory(), s) for s in SC]
print("texto", summarize(R_text), flush=True)
R_bm25 = R_nostate = []
if a.ablaciones:
    R_bm25 = [run_scenario(make_agent_factory(Searcher(lex, den, rrf, mode="bm25")), s) for s in SC]
    R_nostate = [run_scenario(make_agent_factory(use_state=False), s) for s in SC]
    for n, R in (("bm25", R_bm25), ("sin_estado", R_nostate)): print(n, summarize(R), flush=True)

VS = [s for s in SC if s["voice"]]
R_voice = [run_voice_scenario(make_agent_factory(), s, asr, tts, AUDIO_IN, f"{OUT}/audio_out", a.ref_text) for s in VS]
print("voz", summarize(R_voice), flush=True)

json.dump({"cargas": L, "tiempos": T, "texto": R_text, "bm25": R_bm25, "sin_estado": R_nostate, "voz": R_voice},
          open(f"{OUT}/resultados.json", "w"), ensure_ascii=False, indent=1, default=str)

with open(f"{OUT}/informe.txt", "w") as f:
    for tag, R in (("TEXTO", R_text), ("VOZ", R_voice)):
        f.write(f"===== {tag}: {summarize(R)}\n")
        for r in R:
            f.write(f"\n{'OK ' if r['ok'] else 'FALLO'} {r['id']} {r['kind']} viol={r['violations']} pedidos={len(r['orders'])}\n")
            for i, rep in enumerate(r["replies"]): f.write(f"  turno {i+1} respuesta: {rep}\n")
            for vt in r.get("voice_turns", []): f.write(f"  turno {vt['turn']} ASR: {vt['transcript']!r} (esperado {vt['ground_truth']!r})\n")
            for turno, tool, args, res in r["log"]: f.write(f"  log t{turno} {tool} {args} -> {str(res)[:240]}\n")
print(f"listo: {OUT}/informe.txt", flush=True)

if a.sim_voz:
    from metrics import SpeakerSim, wer
    ss = SpeakerSim(); Q = []
    for r in R_voice:
        for v in r["voice_turns"]:
            Q.append({"id": r["id"], "turno": v["turn"], "WER_tts": wer(spoken_text(v["reply"]), asr.transcribe(v["audio_out"])),
                      "sim_voz": ss.sim(REF_WAV, v["audio_out"]), "texto_distinto_de_ref": v["text_differs_from_ref"]})
    json.dump(Q, open(f"{OUT}/calidad_voz.json", "w"), ensure_ascii=False, indent=1)
    print(pd.DataFrame(Q)[["WER_tts", "sim_voz"]].mean())
