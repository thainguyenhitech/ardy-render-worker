"""Chatterbox Multilingual v3 — tiến trình con của worker RunPod (venv_cb, 127.0.0.1:8813). Giao diện tối giản:
POST /doc {text, ngon_ngu, mau_b64, id_giong} → wav b64 24 kHz. Mẫu giọng ghi ra /tmp theo hash để cache."""
import base64
import hashlib
import io
import os
import threading
import time
from pathlib import Path

import soundfile as sf
import torch
import uvicorn
from fastapi import Body, FastAPI

app = FastAPI()
_m = None
_tt = {"san_sang": False, "loi": None, "nap_s": None}
MAU = Path("/tmp/cb_mau")
MAU.mkdir(exist_ok=True)


def _nap():
    global _m
    t0 = time.time()
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        _m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        _tt.update(san_sang=True, nap_s=round(time.time() - t0, 1))
    except Exception as e:  # noqa: BLE001
        _tt["loi"] = f"{type(e).__name__}: {str(e)[:200]}"


@app.on_event("startup")
def _khoi_dong():
    threading.Thread(target=_nap, daemon=True).start()


@app.get("/health")
def health():
    return _tt


@app.post("/doc")
def doc(d: dict = Body(...)):
    if not _tt["san_sang"]:
        return {"loi": _tt["loi"] or "chatterbox đang nạp"}
    raw = base64.b64decode(d["mau_b64"])
    p = MAU / (hashlib.sha1(raw).hexdigest()[:16] + ".wav")
    if not p.exists():
        p.write_bytes(raw)
    t0 = time.time()
    with torch.inference_mode():
        # cfg_weight=0: tác giả khuyên khi mẫu khác ngôn ngữ đích (chống lây ngữ điệu mẫu sang câu)
        wav = _m.generate(d["text"], language_id=d["ngon_ngu"], audio_prompt_path=str(p),
                          cfg_weight=float(d.get("cfg_weight", 0.0)), exaggeration=float(d.get("exaggeration", 0.5)))
    au = wav.squeeze(0).cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, au, _m.sr, format="WAV", subtype="PCM_16")
    return {"audio_b64": base64.b64encode(buf.getvalue()).decode(), "sr": int(_m.sr), "giay": round(time.time() - t0, 3),
            "tieng_s": round(len(au) / _m.sr, 3)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("CB_CONG", "8813")), log_level="warning")
