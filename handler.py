"""Handler RunPod serverless — worker TTS GPU của cụm `tts_server/` (11/09, user: "set luôn runpod chạy 2 model vienue
và cosy, xoá các model còn lại và không cần dùng model chạy local trên mac mini"). Hai engine, mỗi cái một tiến trình con:
  vieneu (tiếng Việt, vieneu_server.py, 8815) · cosy (ngoại ngữ, cosy_worker.py của cụm, 8814).
input: {op: "health"} · {op: "log"} · {op: "doc", engine: "vieneu"|"cosy", text, ngon_ngu, id_giong, mau_sha1|mau_b64,
        loi_mau, ngon_ngu_mau, speed?} → {audio_b64 (wav PCM16), sr, giay, tieng_s, engine, gpu, mau_cache}
       {op: "ghi_nho", engine, id_giong, mau_*, loi_mau} · {op: "xoa_giong", id_giong} · {op: "emb", mau_*} → {emb}
MẪU GIỌNG UPLOAD MỘT LẦN: `mau_sha1` trỏ tệp trên volume ($VOL/mau/<sha>.b64), thiếu → {loi: "thieu_mau"} để cụm gửi
lại kèm `mau_b64` (đường lên máy user ~25–100 KB/s, 640 KB mỗi câu là 6–38 s trong khi GPU đọc 1–4 s).
Engine con CHƯA NẠP → chờ ≤120 s (worker "ready" là handler lên, con còn nạp 30–60 s — trả 503 ngay là cụm rơi về dự
phòng dù đã trả tiền cold start)."""
import base64
import hashlib
import io
import json
import os
import time
import traceback
import urllib.request

import numpy as np
import runpod
import soundfile as sf
import torch

VOL = os.getenv("VOL", "/runpod-volume")
MAU_DIR = f"{VOL}/mau"            # mẫu giọng đã upload, theo sha1 — sống qua cold start (volume)
CONG = {"vieneu": int(os.getenv("VIENEU_CONG", "8815")), "cosy": int(os.getenv("COSY_CONG", "8814"))}
NGON_NGU_COSY = ("en", "zh", "de", "es", "fr", "it", "ru", "ko")
_tt = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
_cosy_da_nho: set[str] = set()
_mau_url: dict[str, str] = {}


def _duoi_log(ten: str, n: int = 40) -> str:
    try:
        return "\n".join(open(f"{VOL}/{ten}", encoding="utf-8", errors="replace").read().splitlines()[-n:])
    except Exception as e:  # noqa: BLE001
        return f"(không đọc được {ten}: {e})"


def _json(url: str, d: dict | None = None, timeout: float = 600) -> dict:
    r = urllib.request.Request(url, headers={"Content-Type": "application/json"},
                               data=None if d is None else json.dumps(d).encode())
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def _health_con(engine: str) -> dict:
    try:
        return _json(f"http://127.0.0.1:{CONG[engine]}/health", timeout=3)
    except Exception as e:  # noqa: BLE001
        return {"san_sang": False, "loi": str(e)[:100]}


def _cho_con(engine: str, toi_da: float = 120.0):
    """Chờ engine con sẵn sàng; lỗi nạp thật (không phải 'chưa lên'/'đang nạp') thì trả ngay."""
    t0 = time.time()
    while time.time() - t0 < toi_da:
        h = _health_con(engine)
        if h.get("san_sang"):
            return None
        loi = str(h.get("loi") or "")
        if loi and "refused" not in loi and "đang nạp" not in loi and "Connection" not in loi:
            return {"loi": f"{engine}: {loi}"}
        time.sleep(3)
    return {"loi": f"{engine} chưa sẵn sàng sau {toi_da:.0f} s"}


# ---------------------------------------------------------------- mẫu giọng
def _lay_mau_b64(inp) -> str:
    """`mau_b64` (kèm `mau_sha1` thì LƯU lên volume) · chỉ `mau_sha1` (đã lưu) · `mau_url` (tệp text b64 công khai).
    Thiếu trên volume → KeyError('thieu_mau') để cụm gửi lại kèm b64."""
    sha = inp.get("mau_sha1")
    if inp.get("mau_b64"):
        if sha:
            p = f"{MAU_DIR}/{sha}.b64"
            if not os.path.exists(p):
                os.makedirs(MAU_DIR, exist_ok=True)
                with open(p + ".tmp", "w") as f:
                    f.write(inp["mau_b64"])
                os.replace(p + ".tmp", p)
        return inp["mau_b64"]
    if sha:
        p = f"{MAU_DIR}/{sha}.b64"
        if not os.path.exists(p):
            raise KeyError("thieu_mau")
        with open(p) as f:
            inp["mau_b64"] = f.read()
        return inp["mau_b64"]
    u = inp["mau_url"]
    if u not in _mau_url:
        _mau_url[u] = urllib.request.urlopen(u, timeout=60).read().decode().strip()
    inp["mau_b64"] = _mau_url[u]
    return inp["mau_b64"]


def _mau(inp):
    """→ (audio float32 mono, sr, hash16)."""
    raw = base64.b64decode(_lay_mau_b64(inp))
    au, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if au.ndim > 1:
        au = au.mean(1)
    return au, sr, hashlib.sha1(raw).hexdigest()[:16]


def _mau_wav(inp) -> tuple[str, str]:
    """Ghi mẫu ra /tmp/mau/<hash>.wav (một lần) → (đường dẫn, hash)."""
    au, sr, h = _mau(inp)
    os.makedirs("/tmp/mau", exist_ok=True)
    p = f"/tmp/mau/{h}.wav"
    if not os.path.exists(p):
        sf.write(p, au, sr)
    return p, h


def _goi(w, sr, giay, engine, tra_audio=True):
    ra = {"sr": sr, "giay": round(giay, 3), "tieng_s": round(len(w) / sr, 3), "engine": engine, "gpu": _tt["gpu"],
          "rtf": round(giay / max(len(w) / sr, 1e-3), 3)}
    if tra_audio:
        buf = io.BytesIO()
        sf.write(buf, w, sr, format="WAV", subtype="PCM_16")
        ra["audio_b64"] = base64.b64encode(buf.getvalue()).decode()
    return ra


# ---------------------------------------------------------------- CosyVoice 3 (ngoại ngữ)
def _cosy_ghi_nho(inp) -> str:
    p, h = _mau_wav(inp)
    if h not in _cosy_da_nho:
        nn = inp.get("ngon_ngu_mau", "vi")
        _json(f"http://127.0.0.1:{CONG['cosy']}/ghi_nho",
              {"id": h, "wav": p, "loi": inp.get("loi_mau", "") if nn in NGON_NGU_COSY else "", "ngon_ngu": nn}, timeout=300)
        _cosy_da_nho.add(h)
    return h


def _doc_cosy(inp):
    ma = inp["ngon_ngu"]
    if ma not in NGON_NGU_COSY:
        return {"loi": f"cosy không đọc {ma!r}"}
    if (cho := _cho_con("cosy")):
        return cho
    h = _cosy_ghi_nho(inp)
    t0 = time.time()
    r = urllib.request.Request(f"http://127.0.0.1:{CONG['cosy']}/doc", headers={"Content-Type": "application/json"},
                               data=json.dumps({"text": inp["text"], "giong": h, "ngon_ngu": ma,
                                                "speed": float(inp.get("speed") or 1.0)}).encode())
    wav = urllib.request.urlopen(r, timeout=600).read()
    giay = time.time() - t0
    w, out_sr = sf.read(io.BytesIO(wav), dtype="float32")
    return _goi(np.asarray(w, np.float32), int(out_sr), giay, "cosy", tra_audio=inp.get("tra_audio", True))


# ---------------------------------------------------------------- VieNeu (tiếng Việt)
def _vieneu_ghi_nho(inp) -> dict:
    p, _ = _mau_wav(inp)
    return _json(f"http://127.0.0.1:{CONG['vieneu']}/ghi_nho",
                 {"id": inp["id_giong"], "wav": p, "ten": inp.get("ten", ""), "luu": bool(inp.get("luu", True))}, timeout=300)


def _doc_vieneu(inp):
    if (cho := _cho_con("vieneu")):
        return cho
    gid = str(inp.get("id_giong") or "")
    t0 = time.time()
    d = _json(f"http://127.0.0.1:{CONG['vieneu']}/doc", {"text": inp["text"], "giong": gid})
    if d.get("loi") == "thieu_giong":
        # giọng user chưa có trên worker này (volume mới/mất): cụm gửi kèm mẫu thì enroll ngay rồi đọc
        if not (inp.get("mau_b64") or inp.get("mau_sha1") or inp.get("mau_url")):
            return {"loi": "thieu_giong", "giong": gid}
        _vieneu_ghi_nho(inp)
        d = _json(f"http://127.0.0.1:{CONG['vieneu']}/doc", {"text": inp["text"], "giong": gid})
    if "loi" in d:
        return {"loi": f"vieneu: {d['loi']}"}
    w, sr = sf.read(io.BytesIO(base64.b64decode(d["audio_b64"])), dtype="float32")
    return _goi(np.asarray(w, np.float32), int(sr), time.time() - t0, "vieneu", tra_audio=inp.get("tra_audio", True))


# ---------------------------------------------------------------- handler
def handler(job):
    inp = job.get("input") or {}
    op = inp.get("op", "doc")
    try:
        if op == "health":
            return {**_tt, "vieneu": _health_con("vieneu"), "cosy": _health_con("cosy"), "mau_cache": True}
        if op == "log":               # log bootstrap/worker con trên volume (stream log RunPod hay nghẽn)
            return {"bootstrap": _duoi_log("bootstrap.log"), "vieneu_server": _duoi_log("vieneu_server.log", 60),
                    "cosy_worker": _duoi_log("cosy_worker.log", 60)}
        e = inp.get("engine", "vieneu")
        if e not in ("vieneu", "cosy"):
            return {"loi": f"engine {e!r}? (chỉ vieneu | cosy)"}
        if op == "doc":
            r = {"vieneu": _doc_vieneu, "cosy": _doc_cosy}[e](inp)
            if isinstance(r, dict):
                r["mau_cache"] = True          # cờ: worker này hiểu mau_sha1 → cụm thôi gửi b64 mỗi câu
            return r
        if op == "ghi_nho":
            if (cho := _cho_con(e)):
                return cho
            return {"ok": True, **(_vieneu_ghi_nho(inp) if e == "vieneu" else {"hash": _cosy_ghi_nho(inp)}), "mau_cache": True}
        if op == "xoa_giong":
            if e == "vieneu":
                _json(f"http://127.0.0.1:{CONG['vieneu']}/xoa", {"id": inp["id_giong"]}, timeout=30)
            return {"ok": True}
        if op == "emb":
            if (cho := _cho_con("vieneu")):
                return cho
            p, _ = _mau_wav(inp)
            return _json(f"http://127.0.0.1:{CONG['vieneu']}/emb", {"wav": p}, timeout=120)
        return {"loi": f"op {op!r}?"}
    except KeyError as e:
        if "thieu_mau" in str(e):
            return {"loi": "thieu_mau", "mau_cache": True}
        return {"loi": f"KeyError: {e}", "trace": traceback.format_exc()[-1500:]}
    except Exception as e:  # noqa: BLE001
        return {"loi": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-1500:]}


runpod.serverless.start({"handler": handler})
