"""KHO CLIP ARDY RENDER SẴN — runtime chỉ NẠP, không sinh (12/09 tối, user chốt "phiên bản chỉ dùng clip render sẵn").

Kho do `backend/luong_clip/b1_render/ardy_kho.py` dựng: mọi clip mở màn từ idle và kết ở idle (xem docstring tool),
chỉ mục `kho.json` = {tag: [{"tep", "prompt", "dai_s", "toan_than", …}]}. Ở đây:
  · `lay(cue)`: tìm theo TAG (`cue.ten`) rồi theo CÂU (`cue.prompt`, cùng bảng `ardy_cue.don` hai bên nên trùng);
    nhiều biến thể thì XOAY VÒNG theo từng tag (hai lần liên tiếp không lặp cùng bản). Trả BẢN SAO SÂU — bộ đệm
    neo sàn tại chỗ, ống ghi thêm trường vào clip.
  · Không nối chuỗi: clip nào cũng bắt đầu ở idle, ống nối bằng tầng bậc 5 + helper bước chân như mọi clip.
Bật/tắt: `ARDY_KHO_DUNG` (mặc định 1 — chỉ có tác dụng khi kho.json tồn tại) · `ARDY_KHO_LIVE=1` = thiếu tag thì
sinh sống như cũ (mặc định 0: thiếu là bỏ cue + ghi sổ khớp `kho_thieu` để render bổ sung).
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

KHO = Path(os.getenv("ARDY_KHO", Path.home() / "project/motion_trainer/ardy_kho"))
DUNG = os.getenv("ARDY_KHO_DUNG", "1") == "1"
LIVE = os.getenv("ARDY_KHO_LIVE", "0") == "1"
RAM_MAX = int(os.getenv("ARDY_KHO_RAM", "64"))     # ô clip giữ trong RAM (~0,5–1 MB/ô)
# TỰ BỔ SUNG (12/09 tối): tag LLM tự đặt không có trong kho (raise_left_hand, stretch_arms_yawn…) → render ngay bằng
# chính tool `ardy_kho.render_tag` (≈8 s, trong luồng sinh của bộ đệm), ghi vào kho; câu hiện tại có thể lỡ (như ARDY
# sống trước đây) nhưng từ lần sau là clip sẵn. Kho tự giàu theo cách người dùng thật nói. Tắt: ARDY_KHO_TU_BO_SUNG=0.
TU_BO_SUNG = os.getenv("ARDY_KHO_TU_BO_SUNG", "1") == "1"
# KHỚP NGHĨA vào kho: cosine MiniLM tối thiểu để dùng clip kho thay cho tag LLM. Kho 411 câu thưa hơn catalog 41k nên
# cosine thấp hơn tra catalog (đo 12/09: "crosses the arms over the chest" 0,90 · "raises the left hand high" 0,69–0,87 ·
# "does a cartwheel" → "turns around" 0,53 = sai nghĩa). Sàn 0,70 chỉ chặn sai nghĩa; dưới sàn câu chạy nền thuần và
# tag được render bổ sung để lần sau khớp đúng.
NGUONG_NGHIA = float(os.getenv("ARDY_KHO_NGUONG_NGHIA", "0.70"))
DICH_VU = os.getenv("ARDY_SERVICE", "http://localhost:8090").rstrip("/")
_TOOL = Path(__file__).resolve().parents[2] / "backend" / "luong_clip" / "b1_render" / "ardy_kho.py"


# TOÀN THÂN THEO NỘI DUNG — một nguồn sự thật cho tool render lẫn runtime. Ngưỡng là DI CHUYỂN THẬT (đi/chạy/xoay),
# không phải dồn trọng tâm: đo 12/09 tối, khoanh tay hông trôi 12 cm + vặn 27° bị gán toàn thân → ống coi chân là nội
# dung → lộ chân trượt 74 cm/s của ARDY (cử chỉ giữ chân thì ống ghim chân, không thấy).
TOAN_THAN_DI_M = float(os.getenv("ARDY_KHO_TT_DI_M", "0.25"))
TOAN_THAN_QUAY_DO = float(os.getenv("ARDY_KHO_TT_QUAY_DO", "60"))


def toan_than_noi_dung(clip: dict) -> bool:
    """Clip có dời chỗ (hông đi > TOAN_THAN_DI_M) hoặc quay (yaw > TOAN_THAN_QUAY_DO) theo nội dung."""
    import math
    hp = clip.get("hips_pos") or []
    h = (clip.get("bones") or {}).get("Hips") or []
    if len(hp) > 1:
        xs = [float(f[1]) for f in hp]
        zs = [float(f[3]) for f in hp]
        if max(max(xs) - min(xs), max(zs) - min(zs)) > TOAN_THAN_DI_M:
            return True
    if len(h) > 1:
        try:
            from loi.goc_troi import _euler_sang_q, _yaw_cua
            y = [_yaw_cua(_euler_sang_q(f[1], f[2], f[3])) for f in h[::3]]
            u = [y[0]]                                   # bung góc (tránh quấn ±180) rồi lấy biên độ
            for a in y[1:]:
                d = a - u[-1]
                u.append(u[-1] + math.atan2(math.sin(d), math.cos(d)))
            if math.degrees(max(u) - min(u)) > TOAN_THAN_QUAY_DO:
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


FPS_RA = 60           # hợp đồng khung gửi client (CLAUDE.md §4 bẫy 2: mọi tầng trên lưới 60 fps)


def nang_luoi_60(clip: dict) -> dict:
    """Clip kho render ở fps GỐC ARDY (20, `ardy_kho.FPS` 15/09 — theo tài liệu chính thức) → lưới đều 60 fps. Sửa TẠI
    CHỖ, trả clip. Clip đã 60 fps (kho cũ) trả nguyên.

    SPLINE BẬC BA C² (scipy, natural) qua CHÍNH các khung gốc — trên véc-tơ xoay LIÊN TỤC từng xương + `hips_pos`. Không
    dùng `ong.luoi_chuan`: nó nội suy `hips_pos` TUYẾN TÍNH, và `dong_thoi_gian._mau_track` slerp tuyến tính giữa hai khung —
    đưa clip 20 fps thẳng vào đó là tái tạo đúng lỗi gãy khúc mỗi 3 khung của service cũ (vọt gia tốc cổ chân 91–93 % một
    pha, đo 15/09). Khung gốc giữ nguyên, Euler ghi ra nhánh liên tục."""
    import math
    import numpy as np
    fps = float(clip.get("fps") or FPS_RA)
    if fps >= FPS_RA - 1e-6:
        return clip
    b = clip.get("bones") or {}
    if not b:
        return clip
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        logger.warning("kho ARDY: thiếu scipy — không nâng lưới %s fps lên 60", fps)
        return clip
    from loi.dong_thoi_gian import _q_sang_euler
    from loi.goc_troi import _euler_sang_q
    t_cuoi = max(float(tr[-1][0]) for tr in b.values() if tr)
    moc = np.arange(int(math.floor(t_cuoi * FPS_RA + 1e-6)) + 1) / FPS_RA

    def _log(q):
        w = float(np.clip(q[0], -1.0, 1.0)); v = np.asarray(q[1:], float); s = float(np.linalg.norm(v))
        return (2.0 * math.atan2(s, w) / s) * v if s > 1e-9 else np.zeros(3)

    def _exp(r):
        a = float(np.linalg.norm(r))
        return (math.cos(a / 2), *((math.sin(a / 2) / a) * r)) if a > 1e-9 else (1.0, 0.0, 0.0, 0.0)

    for ten, tr in b.items():
        if len(tr) < 2:
            continue
        tt = np.array([float(f[0]) for f in tr])
        # spline trên THÀNH PHẦN quaternion (bán cầu liên tục) rồi chuẩn hoá — không trên véc-tơ xoay: ép bán cầu làm chuỗi mang
        # w < 0 → |r| ≈ 2π, spline giữa hai véc-tơ như vậy khác trục đi qua phép quay rất lớn (ardy_kho._nang_tho, 15/09 tối)
        Qs, q_truoc = [], None
        for f in tr:
            q = np.asarray(_euler_sang_q(f[1], f[2], f[3]), float)
            if q_truoc is not None and float(np.dot(q, q_truoc)) < 0:
                q = -q                                               # cùng bán cầu khung trước
            q_truoc = q
            Qs.append(q)
        V = CubicSpline(tt, np.array(Qs), axis=0, bc_type="natural")(np.clip(moc, tt[0], tt[-1]))
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        gan, ra = None, []
        for tm, v in zip(moc, V):
            e = _q_sang_euler(tuple(float(c) for c in v), gan)
            ra.append([round(float(tm), 4), round(float(e[0]), 5), round(float(e[1]), 5), round(float(e[2]), 5)])
            gan = e
        b[ten] = ra
    hp = clip.get("hips_pos") or []
    if len(hp) >= 2:
        ht = np.array([float(f[0]) for f in hp]); hv = np.array([f[1:4] for f in hp], float)
        H = CubicSpline(ht, hv, axis=0, bc_type="natural")(np.clip(moc, ht[0], ht[-1]))
        clip["hips_pos"] = [[round(float(tm), 4), *[round(float(x), 5) for x in h]] for tm, h in zip(moc, H)]
    clip["fps"] = FPS_RA
    clip["duration_s"] = round(float(moc[-1]), 4)
    return clip


class Kho:
    def __init__(self, duong: Path = KHO) -> None:
        self.duong = Path(duong)
        self._mt = None
        self._tag: dict[str, list[dict]] = {}
        self._cau: dict[str, list[dict]] = {}
        self._dem: dict[str, int] = {}
        self._ram: OrderedDict[str, dict] = OrderedDict()
        self._khoa = threading.Lock()
        self._dang_bo_sung: dict[str, threading.Event] = {}
        self._nghia_cache: dict = {}

    # -- chỉ mục ---------------------------------------------------------------------------
    def _nap(self) -> None:
        p = self.duong / "kho.json"
        try:
            mt = p.stat().st_mtime
        except OSError:
            self._mt, self._tag, self._cau = None, {}, {}
            return
        if mt == self._mt:
            return
        try:
            cm = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            logger.warning("kho ARDY: kho.json không đọc được", exc_info=True)
            return
        tag, cau = {}, {}
        for t, ds in cm.items():
            ds = [m for m in ds if isinstance(m, dict) and m.get("tep")]
            if not ds:
                continue
            tag[t] = ds
            for m in ds:
                if m.get("prompt"):
                    cau.setdefault(m["prompt"], []).append(m)
        self._mt, self._tag, self._cau = mt, tag, cau
        logger.info("kho ARDY: %d tag · %d clip · %s", len(tag), sum(len(v) for v in tag.values()), self.duong)

    def co(self) -> bool:
        self._nap()
        return bool(self._tag)

    def co_tag(self, tag: str) -> bool:
        self._nap()
        return tag in self._tag

    def tags(self) -> list[str]:
        self._nap()
        return list(self._tag)

    # -- lấy clip ----------------------------------------------------------------------------
    def _doc(self, tep: str) -> dict | None:
        # Ô RAM KHOÁ THEO (tên, mtime): tệp cùng tên bị GHI ĐÈ tại chỗ (đồng bộ R2 tải bản mới, lô sửa kho như
        # `lam_tron_kho` 15/09) thì đọc lại, không phục vụ bản cũ tới khi tiến trình chết. stat ~µs, rẻ hơn deepcopy.
        p = self.duong / tep
        try:
            mt = p.stat().st_mtime_ns
        except OSError:
            logger.warning("kho ARDY: không thấy %s", tep)
            return None
        with self._khoa:
            o = self._ram.get(tep)
            if o is not None and o[0] == mt:
                self._ram.move_to_end(tep)
                return o[1]
        try:
            c = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            logger.warning("kho ARDY: không đọc được %s", tep, exc_info=True)
            return None
        try:
            c = nang_luoi_60(c)          # kho render ở fps gốc ARDY (20) → lưới 60 fps một lần lúc nạp, rồi mới vào ô RAM
        except Exception:  # noqa: BLE001
            logger.warning("kho ARDY: nâng lưới 60 fps hỏng %s — dùng nguyên tệp", tep, exc_info=True)
        with self._khoa:
            self._ram[tep] = (mt, c)
            while len(self._ram) > RAM_MAX:
                self._ram.popitem(last=False)
        return c

    def tim_nghia(self, prompt: str, tim=None) -> tuple[str | None, float]:
        """(câu kho gần nghĩa nhất, cosine) qua `/kho/tim` — CHỈ GỬI CÂU, service tra chỉ mục đường dẫn của nó.

        TÁCH ĐƯỜNG DẪN KHỎI KHO CLIP (13/09, user: *"kiến trúc cần tách biệt giữa path và kho clip, khi search
        dùng mô hình mini search path để lấy địa chỉ clip"*). Bản cũ gửi CẢ KHO làm ứng viên cho `/catalog/nearest`
        — đo trên 21.545 câu: **1,18 MB mỗi truy vấn** (3,29 MB ở 60.000 câu), tra nguội **13,8 s**, và khoá cache
        là `(prompt, SỐ LƯỢNG câu)` nên mỗi clip mới xoá sạch cache; riêng 13/09 kho tăng 15.471 clip. Tức càng
        làm giàu kho thì tra càng đắt — đúng chiều ngược với mục tiêu.

        Nay service giữ chỉ mục đường dẫn + véc-tơ dựng sẵn: gói gửi ~200 byte, truy vấn **18 ms** kể cả lần đầu,
        độ chính xác không đổi (20/20 bộ thử, similarity y hệt). Khoá cache nay CHỈ theo câu, nên kho lớn thêm
        không còn thổi bay cache.

        `tim(text, candidates)` tiêm được (kiểm) — giữ nguyên chữ ký cũ để bài kiểm không phải sửa.
        """
        self._nap()
        if not prompt:
            return None, 0.0
        c = self._nghia_cache.get(prompt)
        if c is not None:
            return c
        ra: tuple[str | None, float] = (None, 0.0)
        try:
            if tim is None:
                import httpx
                d = httpx.post(f"{DICH_VU}/kho/tim", json={"text": prompt, "top": 1}, timeout=10.0).json()
                top = d.get("top") or []
                if top:
                    ra = (str(top[0]["cau"]), float(top[0]["similarity"]))
            else:
                top = tim(prompt, list(self._cau))
                if top:
                    ra = (str(top[0]["sentence"]), float(top[0]["similarity"]))
        except Exception as e:  # noqa: BLE001
            logger.warning("kho ARDY: tìm nghĩa %r hỏng: %s", prompt[:40], str(e)[:80])
        self._nghia_cache[prompt] = ra
        if len(self._nghia_cache) > 512:
            self._nghia_cache.pop(next(iter(self._nghia_cache)))
        return ra

    _BEN = frozenset({"left", "right"})

    def _tag_gan(self, ten: str) -> str | None:
        """Tag kho GẦN NHẤT khi LLM tự đặt biến thể của tag đã có (đo 12/09: `thumbs_up_raise`, `jump_up_twice`,
        `run_in_a_circle`): tag kho mà MỌI từ của nó nằm trong tag LLM, chọn cái nhiều từ nhất. Không đổi BÊN: tag LLM
        có left/right mà tag kho không có (hoặc ngược lại) thì không gần (`raise_left_hand` ≠ `raise_hand`)."""
        tu = set(t for t in ten.split("_") if t)
        if not tu:
            return None
        ben = tu & self._BEN
        tot, diem = None, 0
        for t in self._tag:
            tt = set(x for x in t.split("_") if x)
            if not tt or not tt <= tu or (tt & self._BEN) != ben:
                continue
            if len(tt) > diem:
                tot, diem = t, len(tt)
        return tot

    def lay(self, cue) -> dict | None:
        """Bản sao sâu clip cho `cue` (có `.ten`, `.prompt`) hoặc None khi kho không có."""
        self._nap()
        ten = getattr(cue, "ten", "") or ""
        prompt = getattr(cue, "prompt", "") or ""
        ds = self._tag.get(ten) or self._cau.get(prompt)
        if not ds:
            # KHỚP NGHĨA (12/09, user: "LLM không biết catalog có gì… tag matching câu trong catalog, offline thì
            # matching clip bằng model search mini"): câu của tag → MiniLM + xếp lại từ khoá trên các câu đã render
            # trong kho (`/catalog/nearest` của service). Đủ gần thì dùng ngay; tag đúng vẫn render bổ sung ở nền.
            gan, sim = self.tim_nghia(prompt)
            if gan is None or sim < NGUONG_NGHIA:
                if gan is not None:
                    logger.info("kho ARDY: %r (%s) gần nhất kho là %r %.2f < %.2f — không dùng", ten, prompt[:40], gan[:50], sim, NGUONG_NGHIA)
                gan2 = self._tag_gan(ten)          # đường lùi khi service tắt / kho quá xa: biến thể tên tag
                if gan2 is None:
                    return None
                logger.info("kho ARDY: %r không có, dùng tag gần theo tên %r", ten, gan2)
                ds = self._tag[gan2]
            else:
                ds = self._cau.get(gan)
                if ds is None:
                    # CHỈ MỤC ĐƯỜNG DẪN (service 8090) ĐI TRƯỚC `kho.json` mà runtime đang nạp — kho vừa tự bổ sung
                    # clip, hoặc service trỏ kho khác. 13/09: kiem_kho_ardy nhận câu của kho thật 38k câu cho kho tạm
                    # 3 clip → `self._cau[gan]` ném KeyError, SẬP đường nạp động tác. Nạp lại một lần; vẫn thiếu thì trượt.
                    self._mt = None
                    self._nap()
                    ds = self._cau.get(gan)
                if ds is None:
                    logger.warning("kho ARDY: chỉ mục đường dẫn trả %r mà kho runtime không có — bỏ cue", gan[:50])
                    return None
                logger.info("kho ARDY: %r (%s) → khớp nghĩa %r %.2f", ten, prompt[:40], gan[:50], sim)
            if TU_BO_SUNG:
                self.bo_sung_nen(cue)
        khoa = ten or ds[0].get("prompt", "")
        with self._khoa:
            i = self._dem.get(khoa, 0)
            self._dem[khoa] = i + 1
        m = ds[i % len(ds)]
        c = self._doc(m["tep"])
        if c is None:
            return None
        ra = copy.deepcopy(c)
        ra["_kho"] = {"tep": m["tep"], "loai": m.get("loai"), "dai_s": m.get("dai_s")}
        # cờ toàn thân tính từ NỘI DUNG clip lúc nạp (một nguồn sự thật `toan_than_noi_dung`), không tin chỉ mục —
        # đổi ngưỡng không phải render/đánh chỉ mục lại
        ra["_kho_toan_than"] = toan_than_noi_dung(c)
        return ra

    def bo_sung(self, cue, render=None) -> dict | None:
        """Render tag thiếu vào kho rồi trả clip (bản sao sâu) — chạy ĐỒNG BỘ trong luồng gọi (luồng sinh của bộ đệm).
        `render(tag, prompt, kho)` tiêm được (kiểm); mặc định là tool `ardy_kho.render_tag`. Cùng tag đang render ở
        luồng khác thì chờ luồng đó rồi lấy từ kho.

        KHO PRODUCTION TRÊN R2 (14/09): có `ARDY_RENDER_ENDPOINT` thì KHÔNG render tại chỗ — gửi câu sang worker RunPod
        `ardy_render` (không chờ); clip đạt lên R2 rồi về kho qua `dong_bo_r2`. Câu hiện tại lỡ động tác như trước."""
        ten = getattr(cue, "ten", "") or ""
        prompt = getattr(cue, "prompt", "") or ""
        if not ten or not prompt:
            return None
        from loi import dong_bo_r2
        if render is None and dong_bo_r2.RENDER_ENDPOINT:
            try:
                if dong_bo_r2.gui_render(prompt):
                    logger.info("kho ARDY: xếp hàng render %r (%s) — gửi RunPod theo lô", ten, prompt[:50])
            except Exception:  # noqa: BLE001
                logger.warning("kho ARDY: xếp hàng render %r hỏng", ten, exc_info=True)
            return None
        with self._khoa:
            dang = self._dang_bo_sung.get(ten)
            if dang is None:
                dang = self._dang_bo_sung[ten] = threading.Event()
                toi_render = True
            else:
                toi_render = False
        if not toi_render:
            dang.wait(120.0)
            return self.lay(cue)
        try:
            if render is None:
                render = _tool().render_tag
            m = render(ten, prompt, self.duong)
            logger.info("kho ARDY: tự bổ sung %r → %s", ten, m.get("tep") if m else "TRƯỢT")
        except Exception:  # noqa: BLE001
            logger.warning("kho ARDY: tự bổ sung %r hỏng", ten, exc_info=True)
            m = None
        finally:
            with self._khoa:
                self._dang_bo_sung.pop(ten, None)
            dang.set()
        self._mt = None                                   # ép nạp lại chỉ mục
        return self.lay(cue) if m else None

    def bo_sung_nen(self, cue, render=None) -> bool:
        """Kích render bổ sung ở LUỒNG NỀN RIÊNG, trả ngay. Đo 12/09: render đồng bộ trong luồng sinh của bộ đệm (một
        luồng) chặn cả các cue CÓ SẴN phía sau 8 s mỗi tag thiếu → giơ ngón cái/tay lên ngực cũng "còn rỗng". Câu hiện
        tại lỡ động tác này, từ lần sau có sẵn. Trả False nếu tag đó đang render rồi."""
        ten = getattr(cue, "ten", "") or ""
        with self._khoa:
            if not ten or ten in self._dang_bo_sung:
                return False
        threading.Thread(target=self.bo_sung, args=(cue, render), name=f"kho-bo-sung-{ten}", daemon=True).start()
        return True

    def bao(self) -> dict:
        self._nap()
        return {"tag": len(self._tag), "clip": sum(len(v) for v in self._tag.values()), "ram": len(self._ram), "duong": str(self.duong)}


def _tool():
    """Nạp tool render (`backend/luong_clip/b1_render/ardy_kho.py`) như mô-đun — không phải gói, nạp theo đường dẫn."""
    import importlib.util
    import sys
    m = sys.modules.get("ardy_kho")
    if m is not None:
        return m
    spec = importlib.util.spec_from_file_location("ardy_kho", _TOOL)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ardy_kho"] = m
    spec.loader.exec_module(m)
    return m


_KHO: Kho | None = None


def kho() -> Kho:
    global _KHO
    if _KHO is None:
        _KHO = Kho()
        # KHO PRODUCTION TRÊN R2 (14/09): bật đồng bộ thì lượt ĐẦU (đối chiếu đủ) chạy ngay ở luồng nền rồi định kỳ —
        # kho cục bộ `ARDY_KHO` là bản sao của R2, `_nap` tự nạp lại khi kho.json đổi
        from loi import dong_bo_r2
        if dong_bo_r2.BAT:
            db = dong_bo_r2.DongBoR2.mo(KHO)
            if db is not None:
                db.chay_nen()
    return _KHO


def bat() -> bool:
    """Kho có được dùng không: bật cờ VÀ kho có chỉ mục."""
    return DUNG and kho().co()
