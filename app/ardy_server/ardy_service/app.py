#!/usr/bin/env python3
"""Motion service: câu chữ + frame đầu vào -> motion streaming.

Gộp kho catalog và worker ARDY thành MỘT địa chỉ. Bên gọi không cần biết vector 4096 chiều, không
cần biết hãm quán tính, không cần biết cách nối hai clip — chỉ gửi câu và tư thế đang đứng.

    POST /motion          một lần trả hết, JSON
    POST /motion/stream   NDJSON, mỗi clip một dòng ngay khi sinh xong
    WS   /motion          gửi cue lúc nào cũng được, nhận clip khi có

ARDY chạy ngay trong tiến trình này (mặc định) hoặc ở máy khác (`ARDY_URL`). Kho catalog nạp một
lần vào RAM: 40.524 câu tốn 62 MB, tra nguyên văn 0,17 ms, tra ngữ nghĩa 9,9 ms.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from catalog import CatalogService

from .settle import HOLD as SETTLE_HOLD, Settle

log = logging.getLogger("motion")
ROOT = Path(__file__).resolve().parent.parent

# ---- cấu hình ---------------------------------------------------------------------------
CATALOG_URL = os.getenv("ARDY_CATALOG", f"file://{ROOT}/ardy-catalog")
# Trống = nhúng ARDY vào chính tiến trình này (một cục, một cổng). Có = gọi worker ở máy khác.
ARDY_URL = os.getenv("ARDY_URL", "").rstrip("/")
ENCODER_URL = os.getenv("ENCODER_URL", "").rstrip("/")  # dịch vụ mã hoá thường trú, tuỳ chọn
MODEL = os.getenv("ARDY_MODEL", "core8")

# `CHUNK_SECONDS` là ĐỘ HẠT của luồng, KHÔNG phải độ dài động tác. Động tác dài bao nhiêu là do
# chính chuyển động quyết định (xem `settled()`), service chỉ cắt luồng thành từng miếng để bên gọi
# phát được ngay miếng đầu. Miếng phải dài hơn thời gian sinh ra nó thì luồng mới đuổi kịp thời
# gian thực: trên M4 sinh tốn ~0,72 lần độ dài miếng, nên 2,5 giây dư ~0,6 giây mỗi mắt xích.
CHUNK_SECONDS = float(os.getenv("MOTION_CHUNK_SECONDS", "2.5"))
DEFAULT_SECONDS = float(os.getenv("MOTION_SECONDS", "2.5"))  # chỉ dùng khi bên gọi ép `seconds`
DEFAULT_FPS = int(os.getenv("MOTION_FPS", "20"))
HISTORY_FRAMES = int(os.getenv("MOTION_HISTORY_FRAMES", "16"))
ZV_DECAY = float(os.getenv("MOTION_ZV_DECAY", "0.5"))
SESSION_TTL = float(os.getenv("MOTION_SESSION_TTL", "900"))
# Trần AN TOÀN, không phải tham số thiết kế: động tác tuần hoàn (nhảy, chạy) không bao giờ lắng
# nên nếu bên gọi không đóng luồng thì phải có chỗ dừng, nếu không worker chạy mãi.
MAX_SECONDS = float(os.getenv("MOTION_MAX_SECONDS", "60"))
# Ngưỡng nhận biết động tác đã lắng nằm ở `ardy_service/settle.py` (dùng chung với demo).
MIN_SIMILARITY = float(os.getenv("MOTION_MIN_SIMILARITY", "0.75"))

# THỜI LƯỢNG TỰ NHIÊN CỦA TỪNG CÂU — đo sẵn cho cả kho 40.524 câu (09/09, GPU thuê ~10 giờ,
# `tools/khao_sat_thoi_luong.py`). Bên gọi cần biết TRƯỚC động tác dài bao nhiêu để xếp lịch
# chèn vào lời nói; không có bảng này thì phải đoán rồi cắt ngang cử chỉ — đo được KHÔNG câu nào
# ngắn hơn 4 giây, trung vị 5, trong khi bên gọi vẫn xin 2,5–3 giây.
# {câu: (dài giây, có tự lắng)}; câu chưa đo → không có khoá, `/catalog/lookup` trả None.
_THOI_LUONG: dict[str, tuple[float, bool]] = {}


def _nap_thoi_luong() -> None:
    """Nạp `thoi_luong.jsonl` cạnh kho catalog (chỉ với kho file://; R2 thì bỏ qua)."""
    import json as _json
    from pathlib import Path as _Path
    if not CATALOG_URL.startswith("file://"):
        return
    p = _Path(CATALOG_URL[len("file://"):]) / "thoi_luong.jsonl"
    if not p.exists():
        log.info("không có bảng thời lượng ở %s — bên gọi sẽ phải tự đoán", p)
        return
    n = 0
    for ln in p.read_text(encoding="utf-8").splitlines():
        try:
            r = _json.loads(ln)
            _THOI_LUONG[r["cau"]] = (float(r["dai"]), bool(r.get("lang")))
            n += 1
        except Exception:
            pass
    log.info("bảng thời lượng: %d câu", n)


# ---- xử lý tư thế đầu vào ----------------------------------------------------------------
def normalize_history(raw, fps: int) -> Optional[List[dict]]:
    """Nhận tư thế đầu vào ở mọi dạng hợp lý và trả về đúng dạng worker cần.

    Bên gọi thường chỉ có MỘT khung — tư thế nhân vật đang đứng. Một khung thì không có vận tốc,
    nên nhân bản đủ `HISTORY_FRAMES` là đúng ý nghĩa vật lý: đứng yên. Chấp nhận cả ba dạng:

        {"root": [x,y,z], "quats": [[x,y,z,w], ...]}          một khung trần
        [{"root": ..., "quats": ...}, ...]                     nhiều khung trần
        [{"t": 0.05, "data": {"root": ..., "quats": ...}}, ...] dạng worker trả về
    """
    if not raw:
        return None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        return None

    frames = []
    for f in raw:
        if not isinstance(f, dict):
            raise ValueError("mỗi khung lịch sử phải là một object")
        d = f.get("data") if isinstance(f.get("data"), dict) else f
        if "root" not in d or "quats" not in d:
            raise ValueError("khung lịch sử thiếu 'root' hoặc 'quats'")
        frames.append({"root": list(d["root"]), "quats": [list(q) for q in d["quats"]]})

    if len(frames) == 1:  # một khung = đứng yên, nhân bản cho đủ cửa sổ
        frames = frames * HISTORY_FRAMES
    frames = frames[-HISTORY_FRAMES:]
    return [{"t": i / fps, "data": f} for i, f in enumerate(frames)]


def brake(frames: List[dict], decay: float = ZV_DECAY) -> List[dict]:
    """Zero-velocity padding: giữ nguyên khung CUỐI, kéo các khung trước lại gần nó.

    Không có bước này thì ARDY đọc được vận tốc của động tác cũ trong lịch sử và lao tiếp theo
    hướng đó vài khung đầu của động tác mới. Đo được: đi -> giậm chân vào với 17,5 độ mỗi khung,
    hãm ở 0.5 còn 8,9 độ.
    """
    if not frames or decay >= 1.0:
        return frames
    lq = np.asarray(frames[-1]["data"]["quats"], dtype=np.float32)
    lr = np.asarray(frames[-1]["data"]["root"], dtype=np.float32)
    out, n = [], len(frames)
    for i, f in enumerate(frames):
        w = decay ** (n - 1 - i)
        q = np.asarray(f["data"]["quats"], dtype=np.float32)
        # cùng bán cầu với khung cuối, nếu không phép nội suy đi vòng xa
        q = np.where(np.sum(q * lq, axis=-1, keepdims=True) < 0, -q, q)
        b = lq + (q - lq) * w
        b /= np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
        r = lr + (np.asarray(f["data"]["root"], dtype=np.float32) - lr) * w
        out.append({"t": f["t"], "data": {"root": [round(float(x), 5) for x in r],
                                          "quats": [[round(float(x), 5) for x in v] for v in b]}})
    return out


def tail_of(clip: dict, n: int = HISTORY_FRAMES) -> List[dict]:
    """n khung cuối của một clip, dạng lịch sử cho clip kế tiếp."""
    fps = clip.get("fps") or DEFAULT_FPS
    return [
        {"t": i / fps, "data": {"root": f["data"]["root"], "quats": f["data"]["quats"]}}
        for i, f in enumerate(clip["frames"][-n:])
    ]


# ---- chỗ sinh chuyển động ----------------------------------------------------------------
class Backend:
    """Nơi gọi ARDY. Nhúng trong tiến trình, hoặc qua HTTP tới worker ở máy khác."""

    def __init__(self) -> None:
        self.embedded = not ARDY_URL
        self.engine = None
        self.http: Optional[httpx.AsyncClient] = None
        if self.embedded:
            # Worker phải chạy không trạng thái: vector đi kèm từng yêu cầu, không nạp bộ mã hoá
            # 16 GB. Đặt trước khi import vì module đọc env lúc nạp.
            os.environ.setdefault("ARDY_TEXT_EMBEDDINGS", "none")
            os.environ.setdefault("ARDY_WARMUP", "0")  # làm nóng cần câu chữ, mà worker không có
            from server.ardy_stream_server import Engine  # noqa: PLC0415

            self.engine = Engine()
            threading.Thread(target=self.engine.load, name="ardy-loader", daemon=True).start()
            threading.Thread(target=self.engine.run_forever, name="ardy-gen", daemon=True).start()
            log.info("ARDY nhúng trong tiến trình, thiết bị %s", self.engine.device)
        else:
            self.http = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0))
            log.info("ARDY ở %s", ARDY_URL)

    async def ready(self) -> tuple[bool, Optional[str]]:
        if self.embedded:
            return bool(self.engine.ready), self.engine.error
        try:
            r = await self.http.get(f"{ARDY_URL}/health")
            d = r.json()
            return bool(d.get("ready")), d.get("error")
        except Exception as e:  # noqa: BLE001
            return False, str(e)

    async def skeleton(self) -> dict:
        if self.embedded:
            model = await asyncio.to_thread(self.engine.get_model, MODEL)
            sk = model.skeleton
            # parents + rest_positions: BẮT BUỘC cho project tự retarget sang rig khác (Mixamo/RPM).
            # Xoay trả về là LOCAL so với khớp cha nên không có cây cha thì không dựng được xoay
            # thế giới; rest_positions cho tỉ lệ chi để quy đổi quãng dịch chuyển của gốc.
            # `server/ardy_stream_server.py` vẫn trả hai trường này — service phải bằng vai, nếu
            # không project dùng service buộc phải chép cứng (đúng cái §4 dặn đừng làm).
            rest = sk.neutral_joints.detach().cpu().numpy()
            rest = rest - rest[sk.root_idx]
            return {
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
        r = await self.http.get(f"{ARDY_URL}/skeleton")
        r.raise_for_status()
        return r.json()

    async def generate(self, job: dict) -> dict:
        if self.embedded:
            return await asyncio.to_thread(self.engine.generate_clip, job)
        r = await self.http.post(f"{ARDY_URL}/generate", json=job)
        r.raise_for_status()
        return r.json()

    async def aclose(self) -> None:
        if self.http is not None:
            await self.http.aclose()


# ---- phiên: nhớ khung cuối để bên gọi không phải gửi lại ----------------------------------
class Sessions:
    """Nhớ đuôi của clip cuối cùng theo `session_id`.

    Bên gọi có thể tự giữ khung cuối rồi gửi lên mỗi lần, nhưng đa số chỉ muốn gửi câu tiếp theo.
    Cho một `session_id` là service tự nối chuỗi.
    """

    def __init__(self) -> None:
        self._d: dict[str, tuple[float, List[dict]]] = {}
        self._lock = threading.Lock()

    def get(self, sid: Optional[str]) -> Optional[List[dict]]:
        if not sid:
            return None
        with self._lock:
            v = self._d.get(sid)
            return v[1] if v else None

    def put(self, sid: Optional[str], frames: List[dict]) -> None:
        if not sid:
            return
        with self._lock:
            self._d[sid] = (time.time(), frames)

    def drop(self, sid: Optional[str]) -> None:
        if not sid:
            return
        with self._lock:
            self._d.pop(sid, None)

    def sweep(self) -> None:
        cut = time.time() - SESSION_TTL
        with self._lock:
            for k in [k for k, (t, _) in self._d.items() if t < cut]:
                del self._d[k]

    def __len__(self) -> int:
        return len(self._d)


# ---- service ------------------------------------------------------------------------------
class MotionService:
    def __init__(self) -> None:
        self.catalog = CatalogService(CATALOG_URL, min_similarity=MIN_SIMILARITY)
        self.backend = Backend()
        self.sessions = Sessions()
        self.enrich_http = httpx.AsyncClient(timeout=30.0) if ENCODER_URL else None
        self._enriching: set[str] = set()
        self.stats = {"cues": 0, "exact": 0, "nearest": 0, "missing": 0}

    # -- tra vector ------------------------------------------------------------------
    def lookup(self, text: str):
        hit = self.catalog.lookup(text)
        if hit.exact:
            self.stats["exact"] += 1
        elif hit.vector is not None:
            self.stats["nearest"] += 1
        else:
            self.stats["missing"] += 1
        return hit

    async def enrich(self, text: str) -> None:
        """Đẩy câu lạ sang dịch vụ mã hoá để lượt sau khớp nguyên văn. Không chặn lượt này."""
        if not self.enrich_http or text in self._enriching:
            return
        self._enriching.add(text)
        try:
            r = await self.enrich_http.post(f"{ENCODER_URL}/encode", json={"sentences": [text]})
            if r.status_code == 200:
                await asyncio.to_thread(self.catalog.reload_if_changed)
                log.info("đã mã hoá thêm %r", text[:60])
        except Exception as e:  # noqa: BLE001
            log.warning("mã hoá %r hỏng: %s", text[:40], e)
        finally:
            self._enriching.discard(text)

    # -- sinh một clip ---------------------------------------------------------------
    async def one(self, text: str, seconds: float, fps: int, history: Optional[List[dict]],
                  steps: Optional[int] = None, target_pose: Optional[dict] = None,
                  cfg_cstr: Optional[float] = None) -> dict:
        hit = self.lookup(text)
        if hit.vector is None:
            raise HTTPException(503, "kho catalog rỗng: chạy catalog/encode.py trước")
        if not hit.exact and hit.similarity < self.catalog.min_similarity:
            asyncio.create_task(self.enrich(text))

        job: dict = {
            "model": MODEL,
            # Gửi câu ĐÃ KHỚP chứ không phải câu gốc: vector và câu chữ phải cùng một câu, nếu
            # không classifier-free guidance kéo theo hai hướng khác nhau.
            "prompt": hit.sentence,
            "embedding": base64.b64encode(hit.vector.astype(np.float16).tobytes()).decode(),
            "duration_s": seconds,
            "fps": fps,
            "format": "array",
        }
        # SỐ BƯỚC KHỬ NHIỄU: worker nhận `steps` (kẹp ≤ num_base_steps) nhưng service chưa hề
        # truyền xuống. Đây là knob tốc độ trực tiếp — thời gian sinh tỉ lệ THẲNG với số bước.
        if steps:
            job["steps"] = int(steps)
        if history:
            job["history"] = brake(history)
            job["history_fps"] = fps
            job["history_frames"] = HISTORY_FRAMES
        # NEO ĐẦU CUỐI: `target_pose` ghim KHUNG CUỐI của miếng này vào tư thế bên gọi giao.
        # Chỉ đặt ở miếng CUỐI của một câu (xem vòng sinh) — ghim từng miếng thì mỗi 2,5 giây
        # nhân vật lại phải về tư thế neo, tức cắt vụn động tác.
        if target_pose:
            job["target_pose"] = target_pose
            # TRỌNG SỐ HƯỚNG DẪN của kênh ràng buộc, tách khỏi trọng số văn bản. Phải chỉnh
            # được: đo 09/09 ở mức mặc định 2,0, ghim khung cuối kéo SẬP biên độ động tác
            # (vẫy tay 0,70 → 0,04 m) — đúng cảnh báo của bài "Less is More: Improving Motion
            # Diffusion Models with Sparse Keyframes": ràng buộc quá tay thì mất chuyển động.
            if cfg_cstr is not None:
                job["cfg_constraint"] = float(cfg_cstr)
        clip = await self.backend.generate(job)
        clip["text"] = text
        clip["matched"] = hit.sentence
        clip["exact"] = bool(hit.exact)
        clip["similarity"] = round(float(hit.similarity), 3)
        self.stats["cues"] += 1
        return clip

    # -- sinh cả chuỗi, đưa ra từng clip ngay khi xong -------------------------------
    async def chain(self, req: dict) -> AsyncIterator[dict]:
        """Stream chuyển động cho tới khi động tác KẾT THÚC, rồi đóng.

        Không có tham số "độ dài clip". ARDY tự hồi quy: nó sinh 8 khung một cửa sổ và sinh mãi,
        không phát tín hiệu kết thúc nào. Nên độ dài do CHUYỂN ĐỘNG quyết định, đo bằng
        `settled()`:

          - Động tác chuyển thế ("sits down on the floor") lắng hẳn: mức chuyển động rơi từ 3,26
            xuống 0,2x rồi phẳng lì. Luồng đóng ngay khi đo được, không cắt ngang.
          - Động tác tuần hoàn ("dances the cha-cha", "runs in a circle") không bao giờ lắng: giữ
            3-5 độ mỗi khung suốt 12 giây liền. Luồng chạy cho tới khi BÊN GỌI đóng kết nối, hoặc
            chạm trần an toàn `MAX_SECONDS`.

        Bên gọi ép độ dài bằng `seconds` nếu thật sự cần, và tắt hẳn phép đo bằng `until: "forever"`.
        """
        texts = req.get("texts") or ([req["text"]] if req.get("text") else [])
        if not texts:
            raise HTTPException(400, "cần 'text' hoặc 'texts'")
        if not all(isinstance(t, str) and t.strip() for t in texts):
            raise HTTPException(400, "'texts' phải là danh sách câu không rỗng")
        fps = int(req.get("fps") or DEFAULT_FPS)
        chunk = max(0.5, min(float(req.get("chunk_seconds") or CHUNK_SECONDS), 10.0))
        steps = int(req["steps"]) if req.get("steps") else None
        ceiling = max(chunk, min(float(req.get("max_seconds") or MAX_SECONDS), 600.0))
        # `seconds` = ép cứng độ dài, bỏ qua phép đo. `until` = "settle" (mặc định) | "forever".
        forced = float(req["seconds"]) if req.get("seconds") else None
        until = str(req.get("until") or ("fixed" if forced else "settle")).lower()
        sid = req.get("session_id")
        if req.get("reset"):
            self.sessions.drop(sid)

        try:
            history = normalize_history(req.get("history"), fps)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if history is None:
            history = self.sessions.get(sid)  # nối tiếp lượt trước của cùng phiên
        # TƯ THẾ ĐÍCH cho khung CUỐI: bên gọi giao tư thế mà clip phải kết thúc vào (thường là
        # khung nghỉ/idle nó sẽ nối tiếp). ARDY tự lái cả chuỗi để hạ cánh đúng đó — đo 09/09:
        # mối nối clip↔clip max 107° → 4,7°, tb 13,9° → 0,7°.
        neo_cuoi = req.get("target_pose")
        if neo_cuoi is not None and not isinstance(neo_cuoi, dict):
            raise HTTPException(400, "'target_pose' phải là {'quats': [...], 'root': [...]}")

        index = 0
        for text in texts:
            elapsed = 0.0
            watch = Settle(fps)  # đo xem động tác đã kết thúc chưa
            budget = min(forced, ceiling) if forced is not None else ceiling
            while elapsed + 1e-3 < budget:
                secs = min(chunk, budget - elapsed)
                # ghim tư thế đích CHỈ ở miếng cuối cùng của câu (biết được khi độ dài đã chốt)
                mieng_cuoi = elapsed + secs + 1e-3 >= budget
                clip = await self.one(text, secs, fps, history, steps,
                                      neo_cuoi if (neo_cuoi and mieng_cuoi) else None,
                                      req.get("cfg_constraint"))
                history = tail_of(clip)
                self.sessions.put(sid, history)
                elapsed += len(clip["frames"]) / (clip.get("fps") or fps)
                quiet = watch.feed(clip)
                done = until not in ("forever", "fixed") and quiet
                clip["index"] = index
                clip["elapsed_s"] = round(elapsed, 2)
                # Mức chuyển động của riêng miếng này, để bên gọi tự quyết nếu muốn luật khác.
                clip["energy_deg"] = round(watch.last, 3)
                clip["settled"] = bool(done)
                index += 1
                yield clip
                if done or elapsed + 1e-3 >= budget:
                    break

    async def aclose(self) -> None:
        await self.backend.aclose()
        if self.enrich_http is not None:
            await self.enrich_http.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="ARDY motion service")
    svc = MotionService()
    _nap_thoi_luong()

    @app.on_event("shutdown")
    async def _stop() -> None:
        await svc.aclose()

    @app.get("/health")
    async def health() -> JSONResponse:
        ready, err = await svc.backend.ready()
        svc.sessions.sweep()
        return JSONResponse({
            "ready": ready,
            "error": err,
            "catalog": len(svc.catalog),
            "ardy": "nhúng trong tiến trình" if svc.backend.embedded else ARDY_URL,
            "encoder": ENCODER_URL or None,
            "sessions": len(svc.sessions),
            "stats": svc.stats,
        })

    @app.get("/skeleton")
    async def skeleton() -> JSONResponse:
        return JSONResponse(await svc.backend.skeleton())

    @app.post("/catalog/nearest")
    async def catalog_nearest(req: Request) -> JSONResponse:
        """Câu gần nghĩa nhất trong DANH SÁCH ỨNG VIÊN bên gọi đưa (kho clip render sẵn của 8010) — cùng MiniLM +
        xếp lại từ khoá như `/catalog/lookup`. Body: {"text": …, "candidates": [...], "top": 3}."""
        body = await req.json()
        text = str(body.get("text") or "")
        cands = body.get("candidates") or []
        top = int(body.get("top") or 3)
        if not text or not isinstance(cands, list):
            raise HTTPException(400, "thiếu 'text' hoặc 'candidates'")
        ra = await asyncio.to_thread(svc.catalog.nearest_in, text, cands, top)
        return JSONResponse({"query": text, "top": ra, "matched": ra[0]["sentence"] if ra else None,
                             "similarity": ra[0]["similarity"] if ra else 0.0})

    # ---- KHO CLIP: TÁCH ĐƯỜNG DẪN KHỎI NỘI DUNG (user 13/09) --------------------------------------
    # "kiến trúc cần tách biệt giữa path và kho clip, khi search dùng mô hình mini search path để lấy
    # địa chỉ clip". `/catalog/nearest` bắt bên gọi gửi CẢ KHO làm ứng viên — đo 13/09 ở 21.545 câu:
    # 1,18 MB mỗi truy vấn (3,29 MB ở 60.000 câu), tra nguội 13,8 s, và cache bên gọi khoá theo SỐ
    # LƯỢNG câu nên mỗi clip mới xoá sạch cache. Càng làm giàu kho thì tra càng đắt — sai chiều.
    # `/kho/tim` giữ chỉ mục đường dẫn (câu → tệp clip) + véc-tơ dựng sẵn ngay tại service: bên gọi
    # gửi ĐÚNG CÂU, nhận về ĐỊA CHỈ clip.
    def _kho_duong():
        from catalog.kho_duong import kho_duong  # noqa: PLC0415
        return kho_duong(lambda xs: svc.catalog.encoder(xs) if svc.catalog.encoder else None)

    @app.post("/kho/tim")
    async def kho_tim(req: Request) -> JSONResponse:
        """Địa chỉ clip gần nghĩa nhất trong kho render sẵn. Body: {"text": …, "top": 1}."""
        body = await req.json()
        text = str(body.get("text") or "")
        if not text:
            raise HTTPException(400, "thiếu 'text'")
        kd = await asyncio.to_thread(_kho_duong)
        ra = await asyncio.to_thread(kd.tim, text, int(body.get("top") or 1))
        return JSONResponse({"query": text, "top": ra,
                             "tep": ra[0]["tep"] if ra else None,
                             "matched": ra[0]["cau"] if ra else None,
                             "similarity": ra[0]["similarity"] if ra else 0.0})

    @app.post("/kho/capnhat")
    async def kho_capnhat() -> JSONResponse:
        """Mã hoá thêm NHỮNG CÂU MỚI của kho vào chỉ mục đường dẫn (tăng dần, gọi sau mỗi đợt render)."""
        kd = await asyncio.to_thread(_kho_duong)
        return JSONResponse(await asyncio.to_thread(kd.cap_nhat))

    @app.get("/kho/trang_thai")
    async def kho_trang_thai() -> JSONResponse:
        return JSONResponse((await asyncio.to_thread(_kho_duong)).bao())

    @app.get("/catalog/lookup")
    async def catalog_lookup(q: str) -> JSONResponse:
        """Xem trước câu sẽ khớp vào đâu, không sinh clip. Dùng để chỉnh câu cho LLM."""
        hit = svc.catalog.lookup(q, record_miss=False)
        tl = _THOI_LUONG.get(hit.sentence)
        return JSONResponse({"query": q, "matched": hit.sentence, "exact": bool(hit.exact),
                             "similarity": round(float(hit.similarity), 3),
                             "found": hit.vector is not None,
                             # THỜI LƯỢNG TỰ NHIÊN đo sẵn cho cả kho (xem `_nap_thoi_luong`):
                             # bên gọi biết TRƯỚC động tác dài bao nhiêu để xếp lịch, thay vì
                             # đoán rồi cắt ngang. None = câu chưa đo.
                             "duration_s": None if tl is None else tl[0],
                             "settled": None if tl is None else tl[1]})

    @app.post("/motion")
    async def motion(req: dict) -> JSONResponse:
        """Trả hết một lần. Dùng khi bên gọi không cần phát sớm."""
        clips = [c async for c in svc.chain(req)]
        return JSONResponse({"clips": clips, "session_id": req.get("session_id")})

    @app.post("/motion/stream")
    async def motion_stream(req: dict) -> StreamingResponse:
        """NDJSON: mỗi dòng một object JSON, kết thúc bằng dòng `{"type": "end"}`.

        Đọc theo dòng ở phía client là phát được clip đầu ngay khi nó xong, không phải chờ cả chuỗi.
        """
        async def gen() -> AsyncIterator[bytes]:
            head = {"type": "skeleton", **(await svc.backend.skeleton())}
            yield (json.dumps(head, ensure_ascii=False) + "\n").encode()
            n, why, secs = 0, "hết câu", 0.0
            try:
                async for clip in svc.chain(req):
                    n += 1
                    secs = clip.get("elapsed_s", 0.0)
                    if clip.get("settled"):
                        why = "động tác đã lắng"
                    yield (json.dumps({"type": "clip", **clip}, ensure_ascii=False) + "\n").encode()
            except asyncio.CancelledError:
                # Bên gọi đóng kết nối giữa chừng — đúng cách dừng một động tác tuần hoàn.
                log.info("bên gọi đóng luồng sau %.1f giây", secs)
                raise
            except HTTPException as e:
                yield (json.dumps({"type": "error", "message": e.detail}) + "\n").encode()
            except Exception as e:  # noqa: BLE001
                log.exception("stream hỏng")
                yield (json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"}) + "\n").encode()
            yield (json.dumps({"type": "end", "clips": n, "seconds": secs, "reason": why,
                               "session_id": req.get("session_id")},
                              ensure_ascii=False) + "\n").encode()

        return StreamingResponse(gen(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.websocket("/motion")
    async def motion_ws(sock: WebSocket) -> None:
        """Gửi cue lúc nào cũng được, nhận clip khi sinh xong.

        client -> {"type":"cue", "text":"...", "seconds":2.5, "history":[...]}
                  {"type":"reset"}  {"type":"ping"}
        server -> {"type":"session"|"clip"|"error"|"pong", ...}
        """
        await sock.accept()
        sid = "ws-" + uuid.uuid4().hex[:8]
        lock = asyncio.Lock()  # sinh tuần tự: chuỗi sẽ đứt nếu hai cue chạy chồng nhau
        try:
            await sock.send_json({"type": "session", "session_id": sid,
                                  **(await svc.backend.skeleton())})
            while True:
                msg = json.loads(await sock.receive_text())
                t = msg.get("type", "cue")
                if t == "ping":
                    await sock.send_json({"type": "pong"})
                    continue
                if t == "reset":
                    svc.sessions.drop(sid)
                    await sock.send_json({"type": "status", "state": "reset"})
                    continue
                if t != "cue":
                    await sock.send_json({"type": "error", "message": f"type lạ: {t!r}"})
                    continue
                req = {**msg, "session_id": sid}
                req.pop("type", None)
                try:
                    n, secs = 0, 0.0
                    async with lock:
                        async for clip in svc.chain(req):
                            n += 1
                            secs = float(clip.get("elapsed_s") or 0.0)
                            await sock.send_json({"type": "clip", **clip})
                    # DẤU KẾT THÚC CUE (09/09): một cue trả NHIỀU miếng, bên gọi phải biết khi
                    # nào hết. Suy ra từ `elapsed_s`/`settled` được nhưng mong manh — miếng cuối
                    # của động tác tuần hoàn không có `settled`, còn `elapsed_s` thì phụ thuộc
                    # cách làm tròn. Đường HTTP stream vốn đã có `end`; WS thiếu.
                    await sock.send_json({"type": "end", "clips": n, "seconds": round(secs, 3)})
                except HTTPException as e:
                    await sock.send_json({"type": "error", "message": e.detail})
                except Exception as e:  # noqa: BLE001
                    log.exception("ws cue hỏng")
                    await sock.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
        except WebSocketDisconnect:
            pass
        finally:
            svc.sessions.drop(sid)

    return app
