"""Metricas de voz: WER, errores en cantidades/importes/productos, semejanza de voz, texto != referencia."""
import re, unicodedata

NUM_WORDS = {"cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
             "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
             "catorce": 14, "quince": 15, "veinte": 20}


def norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def numbers(s):
    """Multiconjunto ordenado de numeros (digitos o palabras) en el texto."""
    out = []
    for t in norm(s).split():
        if t.isdigit():
            out.append(int(t))
        elif t in NUM_WORDS and t not in ("un", "una", "uno"):  # 'un/una' son articulos casi siempre
            out.append(NUM_WORDS[t])
    return sorted(out)


def wer(ref, hyp):
    import jiwer
    return jiwer.wer(norm(ref), norm(hyp))


def asr_errors(ref, hyp, query=None):
    """Errores de transcripcion: cantidades/importes (numeros) y producto (palabras de la consulta perdidas)."""
    e = {"wer": wer(ref, hyp), "num_error": numbers(ref) != numbers(hyp)}
    if query:
        qt = set(norm(query).split()); ht = set(norm(hyp).split())
        e["producto_error"] = bool(qt - ht)
    return e


def differs_from_reference(text, ref_text):
    """Los textos generados deben ser distintos de la transcripcion de la grabacion de referencia."""
    return norm(text) != norm(ref_text)


class SpeakerSim:
    """Coseno entre embeddings ECAPA (SpeechBrain) de la referencia y la salida TTS."""
    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        self.m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                run_opts={"device": "cuda"})

    def emb(self, path):
        import torchaudio, torch
        w, sr = torchaudio.load(path)
        w = torchaudio.functional.resample(w.mean(0, keepdim=True), sr, 16000)
        return self.m.encode_batch(w.to("cuda")).squeeze()

    def sim(self, a, b):
        import torch
        return float(torch.nn.functional.cosine_similarity(self.emb(a), self.emb(b), dim=0))
