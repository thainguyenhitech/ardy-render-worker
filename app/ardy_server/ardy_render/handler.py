"""Worker RunPod serverless (queue) của tool `ardy_render`.

Input:
    {"cau": "A person waves." | ["...", ...], "giay": 6, "ghi_r2": true, "tra_clip": false}
    {"op": "health"}
Output:
    {"ket_qua": [{cau, tag, tep, giay, seed, dat, sinh_s, loi?, meta?, r2?, clip?}], "gpu", "tong_s"}

Handler lên NGAY, model nạp ở luồng nền (CLAUDE.md §8c: RunPod thu hồi worker chưa "ready" sau ~9 phút; job tới lúc
đang nạp thì chờ tối đa ARDY_RENDER_CHO_NAP_S rồi mới báo lỗi, không trả 503 để bên gọi phải tự thử lại).
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # ardy_server/ → import ardy_render

from ardy_render.nguon import NguonTrong  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("ardy_render.handler")

CHO_NAP_S = float(os.getenv("ARDY_RENDER_CHO_NAP_S", "900"))
# backend GOM nhiều câu một job (endpoint idle 60 s — job lẻ là khởi động nguội mỗi câu); ~3–4 s/câu trên GPU 24 GB,
# 100 câu ≈ 6 phút — endpoint phải đặt executionTimeout dư cho cỡ lô này
MAX_CAU = int(os.getenv("ARDY_RENDER_MAX_CAU", "100"))
NGUON = NguonTrong()
threading.Thread(target=NGUON.nap, name="nap_model", daemon=True).start()


def _gpu() -> str | None:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def handler(job: dict) -> dict:
    inp = job.get("input") or {}
    if inp.get("op") == "health":
        return {"san_sang": NGUON.san_sang, "loi": NGUON.loi, "nap_s": NGUON.nap_s, "gpu": _gpu()}
    caus = inp.get("cau")
    if isinstance(caus, str):
        caus = [caus]
    if not isinstance(caus, list) or not caus or not all(isinstance(c, str) and c.strip() for c in caus):
        return {"loi": "cần 'cau': chuỗi hoặc danh sách chuỗi không rỗng"}
    if len(caus) > MAX_CAU:
        return {"loi": f"tối đa {MAX_CAU} câu mỗi job"}
    t0 = time.time()
    while not NGUON.san_sang and NGUON.loi is None and time.time() - t0 < CHO_NAP_S:
        time.sleep(1.0)
    if not NGUON.san_sang:
        return {"loi": NGUON.loi or f"model chưa nạp xong sau {CHO_NAP_S:.0f} s", "nap_s": NGUON.nap_s}
    from ardy_render import render
    from ardy_render.r2 import KhoR2
    giay = float(inp.get("giay") or 6.0)
    ghi = bool(inp.get("ghi_r2", True))
    tra_clip = bool(inp.get("tra_clip", False))
    kho = KhoR2.mo() if ghi else None
    if ghi and kho is None:
        return {"loi": "ghi_r2 bật mà worker thiếu biến R2 (R2_BUCKET, RCLONE_CONFIG_R2_*)"}
    ra = []
    for c in caus:
        kq = render.render_cau(c, NGUON, giay)
        try:
            if kho is not None:
                if kq["dat"]:
                    kq["r2"] = kho.day(kq)
                else:
                    kq["r2_loi"] = kho.ghi_loi(kq)
        except Exception as e:  # noqa: BLE001
            kq["loi_r2"] = f"{type(e).__name__}: {str(e)[:160]}"
            log.exception("đẩy R2 hỏng: %s", c[:60])
        if not tra_clip:
            kq.pop("clip", None)
        ra.append(kq)
        log.info("%s %.1fs %s · %s", "ĐẠT" if kq["dat"] else "TRƯỢT", kq.get("sinh_s") or 0, kq.get("loi") or "", c[:60])
    return {"ket_qua": ra, "gpu": _gpu(), "tong_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
