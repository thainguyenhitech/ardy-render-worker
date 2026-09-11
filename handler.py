"""HANDLER RunPod serverless — worker TTS GPU của cụm (xem bootstrap.sh). Chạy trong venv_qwen; Chatterbox ở tiến
trình con cb_server.py (127.0.0.1:8813). Worker KHÔNG có trạng thái bền: mỗi yêu cầu mang theo mẫu giọng (base64,
≤10 s) + lời mẫu; prompt clone được cache theo id trong lúc worker còn ấm.

input: {op: "health"} · {op: "doc", engine: "qwen"|"chatterbox", text, ngon_ngu: "fr", mau_b64, loi_mau,
        ngon_ngu_mau: "vi", id_giong, speed?}  → {audio_b64 (wav PCM16), sr, giay, tieng_s, engine, gpu}
"""
import base64
import hashlib
import io
import os
import time
import traceback

import numpy as np
import soundfile as sf
import torch

import runpod

QWEN_ID = os.getenv("QWEN_TTS_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
QWEN_MODEL = {"1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"}
VOL = os.getenv("VOL", "/runpod-volume")
NGON_NGU_QWEN = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "de": "German", "fr": "French",
                 "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian"}
NGON_NGU_CB = ("ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt",
               "ru", "sv", "sw", "tr", "zh")
_qwen: dict[str, object] = {}         # id model → Qwen3TTSModel (1,7B + 0,6B cùng nạp được trên 24 GB)
_prompt: dict[str, object] = {}
_tt = {"qwen": {}, "chatterbox": None, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def _nap_qwen(mid: str = QWEN_ID):
    if mid not in _qwen:
        t0 = time.time()
        from qwen_tts import Qwen3TTSModel
        _qwen[mid] = Qwen3TTSModel.from_pretrained(mid, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        _tt["qwen"][mid] = {"nap_s": round(time.time() - t0, 1)}
    return _qwen[mid]


def _duoi_log(ten: str, n: int = 60) -> str:
    try:
        return "\n".join(open(f"{VOL}/{ten}", encoding="utf-8", errors="replace").read().splitlines()[-n:])
    except Exception as e:  # noqa: BLE001
        return f"(không đọc được {ten}: {e})"


def _health_con(cong: int):
    import json
    import urllib.request
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{cong}/health", timeout=3))
    except Exception as e:  # noqa: BLE001
        return {"san_sang": False, "loi": str(e)[:100]}


def _cb_health():
    return _health_con(8813)


def _cho_con(cong: int, ten: str, toi_da: float = 120.0):
    """Engine con (Cosy/Chatterbox) nạp ~30 s sau khi handler đã lên: câu tới sớm thì CHỜ chứ không trả 503 ngay
    (cụm sẽ lùi về engine Mac chậm hơn, phí cold start đã trả — test 11/09)."""
    t0 = time.time()
    while time.time() - t0 < toi_da:
        h = _health_con(cong)
        if h.get("san_sang"):
            return None
        if h.get("loi") and "refused" not in str(h.get("loi")) and "đang nạp" not in str(h.get("loi")):
            return {"loi": f"{ten}: {h['loi']}"}
        time.sleep(3)
    return {"loi": f"{ten} chưa sẵn sàng sau {toi_da:.0f} s"}


NGON_NGU_COSY = ("en", "zh", "de", "es", "fr", "it", "ru", "ko")
_cosy_da_nho: set[str] = set()


def _doc_cosy(inp):
    """CosyVoice 3 qua chính cosy_worker.py của cụm (venv_cosy, 127.0.0.1:8814, CUDA): ghi nhớ mẫu theo hash rồi đọc."""
    import json
    import urllib.request
    ma = inp["ngon_ngu"]
    if ma not in NGON_NGU_COSY:
        return {"loi": f"cosy không đọc {ma!r}"}
    if (cho := _cho_con(8814, "cosy")):
        return cho
    au, sr, h = _mau(inp)
    if h not in _cosy_da_nho:
        os.makedirs("/tmp/cosy_mau", exist_ok=True)
        p = f"/tmp/cosy_mau/{h}.wav"
        sf.write(p, au, sr)
        nn = inp.get("ngon_ngu_mau", "vi")
        r = urllib.request.Request("http://127.0.0.1:8814/ghi_nho", headers={"Content-Type": "application/json"},
                                   data=json.dumps({"id": h, "wav": p, "loi": inp.get("loi_mau", "") if nn in NGON_NGU_COSY else "",
                                                    "ngon_ngu": nn}).encode())
        urllib.request.urlopen(r, timeout=300).read()
        _cosy_da_nho.add(h)
    t0 = time.time()
    r = urllib.request.Request("http://127.0.0.1:8814/doc", headers={"Content-Type": "application/json"},
                               data=json.dumps({"text": inp["text"], "giong": h, "ngon_ngu": ma, "speed": float(inp.get("speed") or 1.0)}).encode())
    wav = urllib.request.urlopen(r, timeout=600).read()
    giay = time.time() - t0
    w, out_sr = sf.read(io.BytesIO(wav), dtype="float32")
    return _goi(np.asarray(w, np.float32), int(out_sr), giay, "cosy", tra_audio=inp.get("tra_audio", True))


_mau_url: dict[str, str] = {}


def _lay_mau_b64(inp) -> str:
    """Mẫu giọng: `mau_b64` trực tiếp, hoặc `mau_url` (tệp text base64 công khai — gọi qua MCP không kèm nổi 400 KB)."""
    if inp.get("mau_b64"):
        return inp["mau_b64"]
    u = inp["mau_url"]
    if u not in _mau_url:
        import urllib.request
        _mau_url[u] = urllib.request.urlopen(u, timeout=60).read().decode().strip()
    inp["mau_b64"] = _mau_url[u]
    return inp["mau_b64"]


def _mau(inp):
    raw = base64.b64decode(_lay_mau_b64(inp))
    au, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if au.ndim > 1:
        au = au.mean(1)
    return au, sr, hashlib.sha1(raw).hexdigest()[:16]


def _doc_qwen(inp):
    mid = QWEN_MODEL.get(str(inp.get("qwen_model", "")), QWEN_ID)
    m = _nap_qwen(mid)
    ma = inp["ngon_ngu"]
    if ma not in NGON_NGU_QWEN:
        return {"loi": f"qwen không đọc {ma!r}"}
    au, sr, h = _mau(inp)
    # mẫu tiếng Việt: Qwen không học tiếng Việt → lời mẫu vô nghĩa với nó; chỉ lấy đặc trưng giọng (x-vector).
    # LỜI MẪU PHẢI KHỚP ĐÚNG ĐOẠN ÂM THANH: gửi lời của mẫu 17 s kèm âm thanh cắt 6 s là Qwen đọc nốt phần lời thừa
    # (đo 11/09: câu 4 s ra 15 s, whisper nghe ra lời mẫu).
    chi_xvec = bool(inp.get("xvec")) or inp.get("ngon_ngu_mau", "vi") not in NGON_NGU_QWEN or not inp.get("loi_mau")
    k = f"{mid}:{inp.get('id_giong', '')}:{h}:{int(chi_xvec)}"
    if k not in _prompt:
        _prompt[k] = m.create_voice_clone_prompt(ref_audio=(au, sr), ref_text=None if chi_xvec else inp["loi_mau"],
                                                 x_vector_only_mode=chi_xvec)
    t0 = time.time()
    with torch.inference_mode():
        wavs, out_sr = m.generate_voice_clone(text=inp["text"], language=NGON_NGU_QWEN[ma], voice_clone_prompt=_prompt[k])
    w = np.asarray(wavs[0], np.float32)
    r = _goi(w, out_sr, time.time() - t0, "qwen", tra_audio=inp.get("tra_audio", True))
    r.update(model=mid.split("/")[-1], xvec=chi_xvec)
    return r


def _doc_cb(inp):
    import json
    import urllib.request
    if inp["ngon_ngu"] not in NGON_NGU_CB:
        return {"loi": f"chatterbox không đọc {inp['ngon_ngu']!r}"}
    if (cho := _cho_con(8813, "chatterbox")):
        return cho
    _lay_mau_b64(inp)
    r = urllib.request.Request("http://127.0.0.1:8813/doc", headers={"Content-Type": "application/json"},
                               data=json.dumps({k: inp[k] for k in ("text", "ngon_ngu", "mau_b64") if k in inp}
                                               | {"id_giong": inp.get("id_giong", "")}).encode())
    d = json.load(urllib.request.urlopen(r, timeout=300))
    if "loi" in d:
        return d
    return {**d, "engine": "chatterbox", "gpu": _tt["gpu"]}


def _goi(w, sr, giay, engine, tra_audio=True):
    ra = {"sr": sr, "giay": round(giay, 3), "tieng_s": round(len(w) / sr, 3), "engine": engine, "gpu": _tt["gpu"],
          "rtf": round(giay / max(len(w) / sr, 1e-3), 3)}
    if tra_audio:
        buf = io.BytesIO()
        sf.write(buf, w, sr, format="WAV", subtype="PCM_16")
        ra["audio_b64"] = base64.b64encode(buf.getvalue()).decode()
    return ra


def handler(job):
    inp = job.get("input") or {}
    op = inp.get("op", "doc")
    try:
        if op == "health":
            return {**_tt, "chatterbox": _cb_health(), "cosy": _health_con(8814), "qwen_nap": sorted(_qwen)}
        if op == "log":               # log bootstrap/worker con trên volume (stream log RunPod hay nghẽn)
            return {"bootstrap": _duoi_log("bootstrap.log"), "cb_server": _duoi_log("cb_server.log", 60),
                    "cosy_worker": _duoi_log("cosy_worker.log", 60)}
        if op == "nap":               # hâm: nạp model Qwen (cả 1,7B lẫn 0,6B nếu yêu cầu)
            for mid in inp.get("qwen_models") or [QWEN_ID]:
                _nap_qwen(QWEN_MODEL.get(mid, mid))
            return {**_tt, "chatterbox": _cb_health(), "cosy": _health_con(8814)}
        if op == "doc":
            e = inp.get("engine", "qwen")
            return ({"qwen": _doc_qwen, "chatterbox": _doc_cb, "cosy": _doc_cosy}.get(e) or (lambda i: {"loi": f"engine {e!r}?"}))(inp)
        return {"loi": f"op {op!r}?"}
    except Exception as e:  # noqa: BLE001
        return {"loi": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-1500:]}


runpod.serverless.start({"handler": handler})
