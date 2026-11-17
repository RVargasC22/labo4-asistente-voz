"""ASR (Whisper) y TTS con clonacion de voz (XTTS-v2). Todo local, sin APIs externas."""
import os, re
import torch

os.environ.setdefault("COQUI_TOS_AGREED", "1")


def fix_brands(text, keys):
    """Corrige marcas largas que Whisper parte o deforma al final ("Schwarz Cops" -> "Schwarzkopf").
    keys: {marca normalizada sin espacios: marca}. Conservador: solo letras, mismas 4 primeras letras, longitud
    parecida y similitud >= 80, para no tocar palabras corrientes ("maximo" vs "Maxcio")."""
    from rapidfuzz import process, fuzz
    from metrics import norm
    words, out, i = text.split(), [], 0
    while i < len(words):
        for n in (2, 1):
            span = words[i:i + n]
            core = [re.sub(r"[^\wáéíóúñü]", "", w) for w in span]
            k = norm("".join(core)).replace(" ", "")
            if len(span) < n or len(k) < 7 or not k.isalpha() or k in keys:
                continue
            m = process.extractOne(k, [c for c in keys if c[:4] == k[:4] and abs(len(c) - len(k)) <= 2],
                                   scorer=fuzz.ratio, score_cutoff=80)
            if m:
                tail = re.match(r".*?(\W*)$", span[-1]).group(1)
                out.append(keys[m[0]] + tail); i += n
                break
        else:
            out.append(words[i]); i += 1
    return " ".join(out)


class ASR:
    def __init__(self, name="openai/whisper-medium", brands=None):
        """brands: marcas del catalogo; si se dan, se corrigen en la transcripcion (ver fix_brands)."""
        from transformers import pipeline
        from metrics import norm
        self.pipe = pipeline("automatic-speech-recognition", model=name, dtype=torch.float16, device=0)
        self.keys = {norm(b).replace(" ", ""): b for b in (brands if brands is not None else [])}

    def transcribe(self, wav_path):
        out = self.pipe(wav_path, generate_kwargs={"language": "spanish", "task": "transcribe"})
        text = out["text"].strip()
        return fix_brands(text, self.keys) if self.keys else text


def _apocope(n):
    from num2words import num2words
    return re.sub(r"uno$", "un", re.sub(r"veintiuno$", "veintiún", num2words(n, lang="es")))


def _euros(m):
    e, c = int(m.group(1)), m.group(2)
    out = f"{_apocope(e)} euro" + ("" if e == 1 else "s")
    if c and int(c.ljust(2, "0")):
        cents = int(c.ljust(2, "0"))
        out += f" con {_apocope(cents)} " + ("céntimo" if cents == 1 else "céntimos")
    return out


def spoken_text(t):
    """Normaliza para TTS: importes a palabras (lo hace el codigo, no el LLM) y simbolos a palabras."""
    t = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", t)  # separador de miles: 1.446 -> 1446
    t = re.sub(r"(\d+)(?:[.,](\d{1,2})(?!\d))?\s*(?:€|euros?\b)", _euros, t)
    t = t.replace("€", " euros").replace("%", " por ciento")
    return re.sub(r"\s+", " ", t).strip()


class TTS:
    def __init__(self, ref_wav, name="tts_models/multilingual/multi-dataset/xtts_v2"):
        from TTS.api import TTS as Coqui
        self.t = Coqui(name).to("cuda")
        self.ref = ref_wav

    def speak(self, text, out_path):
        self.t.tts_to_file(text=spoken_text(text), speaker_wav=self.ref, language="es", file_path=out_path)
        return out_path
