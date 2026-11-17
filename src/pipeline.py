"""Pipeline por turnos: audio -> ASR -> agente (recuperacion + LLM) -> TTS, con tiempos por etapa."""
import os, time
from evaluate import run_scenario
from metrics import asr_errors, differs_from_reference


class Acc:
    """Acumula el tiempo de las llamadas a una funcion (para desglosar LLM vs recuperacion)."""
    def __init__(self, obj, name):
        self.t, f = 0.0, getattr(obj, name)
        def w(*a, **k):
            t0 = time.perf_counter(); r = f(*a, **k); self.t += time.perf_counter() - t0; return r
        setattr(obj, name, w)

    def pop(self):
        t, self.t = self.t, 0.0
        return t


def run_voice_scenario(make_agent, sc, asr, tts, audio_dir, out_dir, ref_text=""):
    """Entrada: audio_dir/<id>_t<i>.wav por cada turno de usuario. Guarda audios de respuesta en out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    state = {}

    def turn_fn(i, agent):
        if "llm" not in state:  # instrumenta una vez por agente
            state["llm"] = Acc(agent.llm, "generate"); state["ret"] = Acc(agent.searcher, "search")
        wav = f"{audio_dir}/{sc['id']}_t{i + 1}.wav"
        t0 = time.perf_counter(); text = asr.transcribe(wav)
        t1 = time.perf_counter(); state["llm"].pop(); state["ret"].pop()
        reply = agent.turn(text)
        t2 = time.perf_counter()
        llm_s, ret_s = state["llm"].pop(), state["ret"].pop()
        out = f"{out_dir}/{sc['id']}_r{i + 1}.wav"
        tts.speak(reply, out)
        t3 = time.perf_counter()  # la respuesta ya se puede reproducir
        gt = sc["turns"][i]["user"]
        rows.append({"turn": i + 1, "audio_in": wav, "audio_out": out, "transcript": text, "ground_truth": gt,
                     "reply": reply, "asr": t1 - t0, "llm": llm_s, "retrieval": ret_s,
                     "agent_other": (t2 - t1) - llm_s - ret_s, "tts": t3 - t2, "total": t3 - t0,
                     "text_differs_from_ref": differs_from_reference(reply, ref_text) if ref_text else None,
                     **asr_errors(gt, text, sc["query"] if i == 0 else None)})
        return reply

    res = run_scenario(make_agent, sc, turn_fn=turn_fn)
    res["voice_turns"] = rows
    return res
