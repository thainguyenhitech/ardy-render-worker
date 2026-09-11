"""VieNeu-TTS v3 Turbo trên CUDA — tiến trình con của worker RunPod (venv_main, 127.0.0.1:8815). User 11/09: "set luôn
runpod chạy 2 model vienue và cosy … không cần dùng model chạy local trên mac mini". Giao diện tối giản, JSON:
  GET  /health            → {san_sang, loi, nap_s, giong: [id…], gpu}
  POST /doc {text, giong} → {audio_b64 (wav PCM16 48 kHz), sr, giay}  | {loi: "thieu_giong"} nếu chưa có giọng
  POST /ghi_nho {id, wav, loi?, luu?} → add_voice; `luu` (giọng user) ghi vào $VOL/vieneu_giong.json — sống qua cold start
  POST /xoa {id}          · POST /emb {wav} → {emb: [192 số]} (thước độ giống của tool clone — không nạp VieNeu ở Mac)
Giọng nhân vật (bản clone từ Kokoro, backend/app/config/custom_voices.json) COPY sẵn trong image."""
import base64
import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import Body, FastAPI

app = FastAPI()
_v = None
_tt: dict = {"san_sang": False, "loi": None, "nap_s": None, "giong": []}
VOL = Path(os.getenv("VOL", "/runpod-volume"))
CUSTOM = Path(os.getenv("VIENEU_CUSTOM", "/app/custom_voices.json"))
GIONG_USER = VOL / "vieneu_giong.json"
_khoa = threading.Lock()
_user: set[str] = set()


def _ids() -> list[str]:
    return sorted(vid for _, vid in _v.list_preset_voices())


def _ghi_user() -> None:
    presets = {}
    for gid in _user:
        e = _v._preset_voices.get(gid)
        if e is None:
            continue
        emb, codes = e.get("speaker_emb"), e.get("codes")
        presets[gid] = {"description": e.get("description", ""), "gender": e.get("gender", ""), "style": e.get("style"),
                        "speaker_emb": [round(float(x), 6) for x in np.asarray(emb).reshape(-1)] if emb is not None else None,
                        "codes": np.asarray(codes, dtype=int).tolist() if codes is not None else None}
    GIONG_USER.parent.mkdir(parents=True, exist_ok=True)
    tmp = GIONG_USER.with_suffix(".tmp")
    tmp.write_text(json.dumps({"presets": presets}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, GIONG_USER)


def _nap() -> None:
    global _v
    t0 = time.time()
    try:
        from vieneu import Vieneu
        _v = Vieneu(mode="v3turbo", backend="pytorch", device="cuda")
        if CUSTOM.exists():
            _v._load_voices_from_file(CUSTOM)
        if GIONG_USER.exists():
            _v._load_voices_from_file(GIONG_USER)
            _user.update(json.loads(GIONG_USER.read_text(encoding="utf-8")).get("presets", {}))
        mac = "am_michael" if "am_michael" in _v._preset_voices else _ids()[0]
        _v.infer("Xin chào, rất vui được gặp bạn.", voice=mac)          # hâm CUDA/KV cache
        _tt.update(san_sang=True, nap_s=round(time.time() - t0, 1), giong=_ids())
    except Exception as e:  # noqa: BLE001
        _tt["loi"] = f"{type(e).__name__}: {str(e)[:300]}"


@app.on_event("startup")
def _khoi_dong():
    threading.Thread(target=_nap, daemon=True).start()


@app.get("/health")
def health():
    return _tt


@app.post("/doc")
def doc(d: dict = Body(...)):
    if not _tt["san_sang"]:
        return {"loi": _tt["loi"] or "vieneu đang nạp"}
    gid = str(d["giong"])
    if gid not in _v._preset_voices:
        return {"loi": "thieu_giong", "giong": gid}
    t0 = time.time()
    with _khoa:
        au = np.asarray(_v.infer(str(d["text"]), voice=gid), np.float32).reshape(-1)
    sr = int(getattr(_v, "sample_rate", 48000))
    buf = io.BytesIO()
    sf.write(buf, au, sr, format="WAV", subtype="PCM_16")
    return {"audio_b64": base64.b64encode(buf.getvalue()).decode(), "sr": sr, "giay": round(time.time() - t0, 3),
            "tieng_s": round(len(au) / sr, 3)}


@app.post("/ghi_nho")
def ghi_nho(d: dict = Body(...)):
    if not _tt["san_sang"]:
        return {"loi": _tt["loi"] or "vieneu đang nạp"}
    gid = str(d["id"])
    with _khoa:
        _v.add_voice(gid, str(d["wav"]), denoise=bool(d.get("denoise", True)), description=str(d.get("ten", "")), save=False)
        if d.get("luu", True):
            _user.add(gid)
            _ghi_user()
        _tt["giong"] = _ids()
    return {"ok": True, "giong": _tt["giong"]}


@app.post("/xoa")
def xoa(d: dict = Body(...)):
    gid = str(d["id"])
    with _khoa:
        if _v is not None and gid in _v._preset_voices:
            _v.remove_voice(gid)
        if gid in _user:
            _user.discard(gid)
            _ghi_user()
        if _v is not None:
            _tt["giong"] = _ids()
    return {"ok": True}


@app.post("/emb")
def emb(d: dict = Body(...)):
    if not _tt["san_sang"]:
        return {"loi": _tt["loi"] or "vieneu đang nạp"}
    with _khoa:
        e = np.asarray(_v.encode_reference(str(d["wav"]), denoise=False)[0], np.float32).ravel()
    return {"emb": [float(x) for x in e]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("VIENEU_CONG", "8815")), log_level="warning")
