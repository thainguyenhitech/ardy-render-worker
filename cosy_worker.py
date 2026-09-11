"""TIẾN TRÌNH CosyVoice của cụm TTS — venv RIÊNG (Python 3.10), chỉ nghe 127.0.0.1, do `cong.py` khởi động.

Vì sao tiến trình riêng: CosyVoice ghim torch 2.3.1 / transformers 4.51.3 / numpy 1.26.4 — không sống chung
venv với VieNeu 3.6 (torch 2.13, transformers 5.16). Cổng 8810 là cửa DUY NHẤT backend/client thấy.

Việc của nó: NHỚ giọng từ một đoạn mẫu (`/ghi_nho`, lưu ra đĩa — khởi động lại không phải trích lại) và
ĐỌC câu ngoại ngữ bằng giọng đó (`/doc`). Ngôn ngữ câu TRÙNG ngôn ngữ mẫu → zero-shot (dùng cả lời mẫu,
giống nhất); KHÁC → cross-lingual (chỉ lấy âm sắc — mẫu tiếng Việt vẫn đọc được tiếng Anh). CosyVoice 3
không có tiếng Việt; tiếng Nhật đòi katakana nên không nhận.

    tts_server/.venv_cosy/bin/python tts_server/cosy_worker.py --cong 8811
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

COSY = Path(os.getenv("COSY_DIR", str(Path.home() / "project" / "CosyVoice")))
MODEL = Path(os.getenv("COSY_MODEL", str(COSY / "pretrained_models" / "Fun-CosyVoice3-0.5B")))
sys.path[:0] = [str(COSY), str(COSY / "third_party" / "Matcha-TTS")]

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import Response  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s cosy %(message)s")
log = logging.getLogger("cosy")

# BỘ CHUẨN HOÁ VĂN BẢN wetext tải FST qua API modelscope — API đó trả 403 từ máy này (11/09) và thư
# viện THỬ LẠI MÃI: nạp model đứng im, CPU 0 %. Đường tải thẳng `…/resolve/master/…` thì được, nên FST
# tải sẵn về COSY/pretrained_models/wetext (xem cai_dat.sh) và chặn snapshot_download trỏ về đó.
WETEXT = Path(os.getenv("COSY_WETEXT", str(COSY / "pretrained_models" / "wetext")))


def _wetext_cuc_bo() -> None:
    if not (WETEXT / "en" / "tn" / "tagger.fst").exists():
        log.warning("thiếu FST wetext ở %s — CosyVoice đọc KHÔNG chuẩn hoá số/ký hiệu", WETEXT)
        return
    import wetext.wetext as _w
    _w.snapshot_download = lambda *a, **k: str(WETEXT)

TIEN_TO = "You are a helpful assistant.<|endofprompt|>"   # CosyVoice 3 bắt buộc (example.py)
NGON_NGU = ("en", "zh", "de", "es", "fr", "it", "ru", "ko")
KHO = Path(__file__).resolve().parent / "du_lieu"
TEP_GIONG = KHO / "cosy_giong.pt"          # {id: model_input của frontend} — đặc trưng giọng đã trích
TEP_META = KHO / "cosy_giong.json"         # {id: {ngon_ngu mẫu, co_loi}}

app = FastAPI(title="cụm TTS — CosyVoice")
_tt: dict = {"san_sang": False, "loi": None, "nap_s": None}
_m = None
_meta: dict[str, dict] = {}
_khoa = threading.Lock()                  # mô hình không an toàn đa luồng: mỗi lúc một lần suy diễn


def _nap() -> None:
    global _m, _meta
    try:
        _wetext_cuc_bo()
        from cosyvoice.cli.cosyvoice import AutoModel
        t0 = time.time()
        _m = AutoModel(model_dir=str(MODEL))
        # THIẾT BỊ: model.py gán cứng 'cuda' không thì 'cpu'. Đo 11/09 trên CPU M4: RTF 3,3–11,6 (chậm hơn
        # nói thật) — GPU Apple (MPS) phải tự dời; phép toán MPS chưa có thì PYTORCH_ENABLE_MPS_FALLBACK=1.
        tb = os.getenv("COSY_THIET_BI", "cpu")
        if tb != "cpu":
            d = torch.device(tb)
            _m.model.device = d
            for mod in (_m.model.llm, _m.model.flow):
                mod.to(d)
            # vocoder HiFT tự ép một nhánh sang float64 (generator.py inference) — MPS không có float64:
            # giữ HiFT ở CPU, chuyển mel về CPU trước khi vào nó
            _hift = _m.model.hift.inference
            _m.model.hift.inference = lambda speech_feat, **k: _hift(speech_feat=speech_feat.cpu(), **k)
        # SỐ BƯỚC ODE của flow (flow.py gán cứng 10): ít bước = nhanh hơn, chất lượng giảm dần.
        buoc = int(os.getenv("COSY_BUOC", "10"))
        if buoc != 10:
            _goc_fwd = _m.model.flow.decoder.forward

            def _fwd(*a, **k):
                if "n_timesteps" in k:
                    k["n_timesteps"] = buoc
                return _goc_fwd(*a, **k)
            _m.model.flow.decoder.forward = _fwd
        _tt.update(thiet_bi=tb, buoc=buoc)
        if TEP_GIONG.exists():
            _m.frontend.spk2info.update(torch.load(TEP_GIONG, map_location="cpu"))
        if TEP_META.exists():
            _meta = json.loads(TEP_META.read_text(encoding="utf-8"))
        with _khoa:   # hâm: lần đọc đầu biên dịch/khởi tạo — đừng để rơi vào câu đầu của user
            for _ in _m.inference_cross_lingual(TIEN_TO + "Hello there.", "", zero_shot_spk_id=next(iter(_meta), ""))\
                    if _meta else ():
                pass
        _tt.update(san_sang=True, nap_s=round(time.time() - t0, 1))
        log.info("CosyVoice sẵn sàng sau %.1f s · %d giọng · sample_rate %d", _tt["nap_s"], len(_meta), _m.sample_rate)
    except Exception as e:  # noqa: BLE001
        _tt["loi"] = f"{type(e).__name__}: {str(e)[:200]}"
        log.exception("nạp CosyVoice hỏng")


def _luu() -> None:
    KHO.mkdir(parents=True, exist_ok=True)
    torch.save({k: _m.frontend.spk2info[k] for k in _meta if k in _m.frontend.spk2info}, TEP_GIONG)
    TEP_META.write_text(json.dumps(_meta, ensure_ascii=False, indent=1), encoding="utf-8")


@app.on_event("startup")
def _khoi_dong() -> None:
    threading.Thread(target=_nap, name="nap-cosy", daemon=True).start()


@app.get("/health")
def health() -> dict:
    return {**_tt, "ngon_ngu": list(NGON_NGU), "giong": sorted(_meta),
            "sample_rate": getattr(_m, "sample_rate", None)}


def _can_san_sang() -> None:
    if not _tt["san_sang"]:
        raise HTTPException(503, _tt["loi"] or "CosyVoice đang nạp")


@app.post("/ghi_nho")
def ghi_nho(d: dict = Body(...)) -> dict:
    """{id, wav: đường dẫn, loi: lời đoạn mẫu ("" nếu không có), ngon_ngu: ngôn ngữ đoạn mẫu}"""
    _can_san_sang()
    gid, wav = str(d["id"]), str(d["wav"])
    loi = str(d.get("loi") or "").strip()
    # MẪU NGẮN: cả LLM (ngữ cảnh token mẫu) lẫn flow (mel mẫu) tỉ lệ với độ dài mẫu — đo 11/09 mẫu 17 s
    # RTF 3,4–7,7, mẫu 8 s 2,9–4,1. Cắt về COSY_MAU_S; đã cắt thì lời mẫu không còn khớp → bỏ lời
    # (đọc chế độ cross-lingual, chỉ lấy âm sắc).
    toi_da = float(os.getenv("COSY_MAU_S", "10"))
    au, sr = sf.read(wav, dtype="float32")
    if au.ndim > 1:
        au = au.mean(axis=1)
    if len(au) > toi_da * sr:
        (KHO / "cosy_mau").mkdir(parents=True, exist_ok=True)
        wav = str(KHO / "cosy_mau" / f"{gid}.wav")
        sf.write(wav, au[: int(toi_da * sr)], sr)
        loi = ""
    t0 = time.time()
    with _khoa:
        _m.add_zero_shot_spk(TIEN_TO + loi, wav, gid)
        _meta[gid] = {"ngon_ngu": (d.get("ngon_ngu") or "vi")[:2], "co_loi": bool(loi)}
        _luu()
    return {"id": gid, "giay": round(time.time() - t0, 2)}


@app.delete("/giong/{gid}")
def xoa(gid: str) -> dict:
    _can_san_sang()
    with _khoa:
        _meta.pop(gid, None)
        _m.frontend.spk2info.pop(gid, None)
        _luu()
    return {"xoa": gid}


@app.post("/doc")
def doc(d: dict = Body(...)) -> Response:
    """{text, giong, ngon_ngu, speed} → wav PCM16 mono ở sample_rate của mô hình (header X-Sample-Rate)."""
    _can_san_sang()
    gid, ma = str(d["giong"]), str(d.get("ngon_ngu") or "en")[:2]
    if gid not in _meta:
        raise HTTPException(404, f"chưa nhớ giọng {gid!r}")
    if ma not in NGON_NGU:
        raise HTTPException(422, f"CosyVoice không đọc {ma!r}")
    text, speed = str(d["text"]).strip(), float(d.get("speed") or 1.0)
    cung = _meta[gid]["co_loi"] and _meta[gid]["ngon_ngu"] == ma
    t0 = time.time()
    with _khoa:
        gen = (_m.inference_zero_shot(text, "", "", zero_shot_spk_id=gid, stream=False, speed=speed) if cung else
               _m.inference_cross_lingual(TIEN_TO + text, "", zero_shot_spk_id=gid, stream=False, speed=speed))
        au = torch.cat([o["tts_speech"] for o in gen], dim=1).squeeze(0).numpy().astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, au, _m.sample_rate, format="WAV", subtype="PCM_16")
    dai = len(au) / _m.sample_rate
    tong = time.time() - t0
    log.info("đọc [%s] %s %.2fs tiếng trong %.2fs (RTF %.2f) · %s", ma, "zero-shot" if cung else "cross-lingual",
             dai, tong, tong / max(dai, 1e-6), text[:50])
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"X-Sample-Rate": str(_m.sample_rate), "X-RTF": f"{tong / max(dai, 1e-6):.3f}"})


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--cong", type=int, default=int(os.getenv("COSY_CONG", "8811")))
    a = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=a.cong, log_level="warning")
