"""NGUỒN SINH TRONG TIẾN TRÌNH cho tool `ardy_render`: câu tiếng Anh → vector LLM2Vec → Engine ARDY → khung thô.

Cắm vào `ardy_kho.dat_nguon` để công thức clip chuẩn (cắt · lối ra · trộn · retarget · cổng) chạy nguyên xi như lúc
dựng kho ~40.000 clip qua HTTP (`ardy_service` `POST /motion`). Ba chỗ phải khớp đường HTTP cũ, đo 14/09:

1. VECTOR theo đúng quy ước catalog (`catalog/encode.py`): mã hoá `norm_sentence(câu)` (bỏ khoảng trắng thừa, CHỮ
   THƯỜNG), lấy `feat[:, 0]`, lưu qua fp16. Engine sống (`encode_text`) mã hoá câu giữ hoa/thường — KHÔNG dùng đường đó.
2. CÂU CỐ ĐỊNH của công thức (idle, lối ra) dùng vector CATALOG nướng sẵn (`co_dinh/`), không mã hoá lại: kho cũ khớp
   câu lối ra "…relaxed at the sides." về câu catalog "…relaxed at THEIR sides." (0,993) — mã hoá sống câu gốc là đổi
   lối ra của mọi clip so với kho.
3. CHUỖI GỌI như service: `normalize_history` → `brake` (hãm vận tốc 0,5) → job {prompt, embedding fp16, duration_s,
   fps, format array, history_frames 16, target_pose, cfg_constraint} → `Engine.generate_clip`. ardy_kho xin
   chunk ≥ độ dài nên service luôn sinh MỘT miếng — ở đây cũng một lượt.

ARDY KHÔNG TẤT ĐỊNH (đo 14/09 trên 8090: cùng câu, cùng 6 s, hai phiên mới → lệch góc TB 10,5°, max 89,5° — lấy nhiễu
`torch.randn` không gieo seed). Nguồn này GIEO SEED theo (câu chuẩn hoá, độ dài, có đích hay không) trước mỗi lần sinh:
cùng yêu cầu → cùng clip, lặp lại được để soi lỗi. Clip kho cũ vốn không lặp lại được nên "cùng chuẩn" nghĩa là cùng
công thức + cùng cổng + phân bố số đo tương đương, không phải trùng từng khung.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[2]          # gốc repo (trong image: /app)
for _p in (GOC / "ardy_server", GOC / "backend" / "luong_clip" / "b1_render", GOC / "backend", GOC / "talkinghead_backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

log = logging.getLogger("ardy_render.nguon")

MODEL = os.getenv("ARDY_MODEL", "core8")
DIM = 4096
CO_DINH = Path(__file__).with_name("co_dinh")      # <sentence_hash>.f16 + chi_muc.json (câu công thức → hash catalog)
DAI_MIEN_TOI_DA = 10.0                              # service kẹp chunk ≤ 10 s; ardy_kho chỉ xin 6/7/8 s và lối ra 3 s


def norm_sentence(s: str) -> str:
    from catalog.store import norm_sentence as _n
    return _n(s)


class NguonTrong:
    """Engine ARDY + bộ mã hoá LLM2Vec trong cùng tiến trình. `nap()` chậm (GPU ~90 s, Mac ~150 s) — gọi ở luồng nền."""

    def __init__(self, thiet_bi: str | None = None) -> None:
        # Engine đọc env lúc import module: tắt nguồn chữ của nó (tool tự mã hoá theo quy ước catalog) + tắt làm nóng
        os.environ.setdefault("ARDY_TEXT_EMBEDDINGS", "none")
        os.environ.setdefault("ARDY_WARMUP", "0")
        if thiet_bi:
            os.environ["ARDY_DEVICE"] = thiet_bi
        self.engine = None
        self.enc = None
        self._info: dict | None = None
        self._co_dinh: dict[str, np.ndarray] = {}
        self._khoa = threading.Lock()              # seed là trạng thái TOÀN CỤC của torch — một lần sinh một lúc
        self.san_sang = False
        self.loi: str | None = None
        self.nap_s: dict = {}

    # ---- nạp ----------------------------------------------------------------------------------------------------
    def nap(self) -> None:
        try:
            t0 = time.time()
            from server.ardy_stream_server import Engine
            self.engine = Engine()
            self.engine.text_encoder = False       # get_model không được tự nạp encoder thứ hai
            self.engine.get_model(MODEL)
            self.nap_s["ardy"] = round(time.time() - t0, 1)
            t1 = time.time()
            from ardy.model.load_model import load_text_encoder
            # Encoder 8B cùng thiết bị Engine (GPU); trên Mac Engine chạy CPU (MPS chậm hơn 32 %) còn encoder nên ở
            # MPS — `ARDY_ENCODER_DEVICE` tách riêng
            dev = os.getenv("ARDY_ENCODER_DEVICE") or self.engine.device
            self.enc = load_text_encoder(mode="local", device=dev)
            self.nap_s["encoder_thiet_bi"] = dev
            self.nap_s["encoder"] = round(time.time() - t1, 1)
            self._nap_co_dinh()
            # LÀM NÓNG TRƯỚC KHI NHẬN JOB: đo lại 15/09 trên L4 chỉ 1,2 s — phần ~90 s của job đầu mỗi worker là NẠP
            # encoder (76–84 s) mà job tới sớm phải chờ, không phải khởi tạo kernel như đoán ban đầu. Giữ vì rẻ và để
            # `nap_s["lam_nong"]` chứng minh đường sinh chạy được trước khi báo `san_sang`.
            t2 = time.time()
            self.vector("A person waves the right hand.")
            self.motion({"text": "A person waves the right hand.", "seconds": 1.0, "fps": 60})
            self.nap_s["lam_nong"] = round(time.time() - t2, 1)
            self.san_sang = True
            log.info("ardy_render sẵn sàng: %s trên %s", self.nap_s, self.engine.device)
        except Exception as e:  # noqa: BLE001
            self.loi = f"{type(e).__name__}: {e}"
            log.exception("ardy_render: nạp hỏng")

    def _nap_co_dinh(self) -> None:
        p = CO_DINH / "chi_muc.json"
        if not p.exists():
            raise FileNotFoundError(f"thiếu {p} — chạy `python -m ardy_render.dung_co_dinh` để chép vector câu cố định")
        for cau, h in json.loads(p.read_text(encoding="utf-8")).items():
            v = np.frombuffer((CO_DINH / f"{h}.f16").read_bytes(), dtype=np.float16).astype(np.float32)
            if v.shape != (DIM,):
                raise ValueError(f"vector cố định {h} sai kích thước {v.shape}")
            self._co_dinh[norm_sentence(cau)] = v

    # ---- skeleton: y hệt `Backend.skeleton` của ardy_service (retarget ăn đúng số làm tròn đó) -------------------
    def info(self) -> dict:
        if self._info is None:
            model = self.engine.get_model(MODEL)
            sk = model.skeleton
            rest = sk.neutral_joints.detach().cpu().numpy()
            rest = rest - rest[sk.root_idx]
            self._info = {
                "model": MODEL,
                "model_fps": float(model.motion_rep.fps),
                "joints": list(sk.bone_order_names),
                "parents": [int(p) for p in sk.joint_parents.tolist()],
                "rest_positions": [[round(float(x), 6) for x in r] for r in rest],
                "root_index": int(sk.root_idx),
                "units": "m", "up_axis": "Y", "forward_axis": "+Z",
                "quaternion_order": "xyzw",
                "rest_pose": "T-pose with identity joint rotations",
            }
        return self._info

    # ---- vector ---------------------------------------------------------------------------------------------------
    def vector(self, cau: str) -> np.ndarray:
        """float32 [4096] đã qua fp16 như catalog lưu. Câu công thức lấy vector catalog nướng sẵn."""
        k = norm_sentence(cau)
        v = self._co_dinh.get(k)
        if v is not None:
            return v
        import torch
        with torch.no_grad():
            feat, lengths = self.enc([k])
        if feat.shape[1] != 1 or int(max(lengths)) != 1:
            raise RuntimeError(f"encoder trả {tuple(feat.shape)}; catalog chỉ hỗ trợ 1 vector mỗi câu")
        return feat[0, 0].float().cpu().numpy().astype(np.float16).astype(np.float32)

    @staticmethod
    def seed(cau: str, giay: float, co_dich: bool) -> int:
        h = hashlib.sha256(f"{norm_sentence(cau)}|{float(giay):.3f}|{int(bool(co_dich))}".encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    # ---- sinh: đúng thân `POST /motion` mà ardy_kho gửi ------------------------------------------------------------
    def motion(self, than: dict) -> dict:
        from ardy.tools import seed_everything
        from ardy_service.app import HISTORY_FRAMES, brake, normalize_history

        cau = str(than["text"])
        giay = float(than["seconds"])
        if giay > DAI_MIEN_TOI_DA:
            raise ValueError(f"độ dài {giay} s > {DAI_MIEN_TOI_DA} s: service cũ sẽ cắt nhiều miếng, tool chưa hỗ trợ")
        fps = int(than.get("fps") or 60)
        vec = self.vector(cau)
        job: dict = {"model": MODEL, "prompt": cau,
                     "embedding": base64.b64encode(vec.astype(np.float16).tobytes()).decode(),
                     "duration_s": giay, "fps": fps, "format": "array"}
        hist = normalize_history(than.get("history"), fps)
        if hist:
            job["history"] = brake(hist)
            job["history_fps"] = fps
            job["history_frames"] = HISTORY_FRAMES
        if than.get("target_pose"):
            job["target_pose"] = than["target_pose"]
            if than.get("cfg_constraint") is not None:
                job["cfg_constraint"] = float(than["cfg_constraint"])
        sd = self.seed(cau, giay, bool(than.get("target_pose")))
        with self._khoa:
            seed_everything(sd)
            clip = self.engine.generate_clip(job)
        clip.update({"text": cau, "matched": cau, "exact": True, "similarity": 1.0, "seed": sd})
        return {"clips": [clip]}
