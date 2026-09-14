"""CLIENT ARDY — nói với ARDY MOTION SERVICE CHẠY NGAY TRÊN MÁY NÀY (cổng 8090).

User chốt 08/09: "server ardy chạy tốt trên mac mini này … không dùng của runpod nữa".
Service ở `~/project/ARDY_NVIDIA` (`python -m ardy_service`), tài liệu
`docs/ARDY_SERVER_GUIDE.md`. Nó gộp sẵn ba việc mà bản RunPod bắt ta tự lo:
kho catalog 40.524 câu (câu → vector), hãm quán tính lịch sử, và ARDY (vector → khung).

Đường đi: prompt tiếng Anh (1 hành động ≤12 từ) -> `POST /motion` -> khung 27 khớp dạng
mảng -> `ardy_retarget.tu_mang` -> `retarget` -> clip đúng định dạng kho -> cache đĩa theo
(model, prompt, dài, fps) -> provider trộn vào câu bằng `gru/blend.tron` như clip Kimodo.

VÌ SAO BỎ RUNPOD (giữ lại đây làm sổ, đừng dựng lại): mỗi PATCH endpoint = version mới =
thay worker (cold start 100–150 s); worker "running" ≠ sẵn sàng (/ping 204 khi đang nạp);
idleTimeout tắt container giữa lúc đang test; ba tiến trình backend trên cùng máy giành nhau
bật/tắt worker; và $1,10/h khi bật. Service cục bộ không có cái nào trong số đó: không khoá,
không cold start, không đồng hồ tự tắt, không tệp trạng thái chung — nên toàn bộ máy móc đó
đã XOÁ khỏi tệp này.

Đổi lại, sinh clip ăn CPU CỦA CHÍNH MÁY NÀY (Mac M4: clip 2,5 s mất ~1,8–2,0 s, ~0,72× thời
gian thực — guide §7 giải thích vì sao CPU nhanh hơn MPS ở mô hình 326M). Nó chạy ở TIẾN
TRÌNH RIÊNG nên không giữ GIL của backend (bài học §13 CLAUDE.md), nhưng vẫn tranh CPU với
LLM/TTS — vé được đặt TRƯỚC suy diễn transformer và lấy sau, đúng như bản RunPod.
"""
from __future__ import annotations

import concurrent.futures as _cf
import hashlib
import json
from collections import OrderedDict
import logging
import math
import os
import threading
import time
from pathlib import Path

from app.modules.body_motion import ardy_kenh as _kenh

logger = logging.getLogger(__name__)
# httpx ghi INFO mỗi request: vòng kiểm tra sức khoẻ 10 s/lần làm log backend không bao giờ
# đứng yên — nhiễu, và từng làm công cụ khảo sát tưởng backend luôn bận nên chờ mãi.
logging.getLogger("httpx").setLevel(logging.WARNING)

DICH_VU = os.getenv("ARDY_SERVICE", "http://localhost:8090").rstrip("/")
CACHE_DIR = Path(os.getenv("ARDY_CACHE", Path.home() / "project/motion_trainer/ardy_cache"))
FPS = 60.0            # service lấy mẫu lại từ 20 fps gốc của model lên mức này (guide §2)
MODEL = os.getenv("ARDY_MODEL", "core8")
# BẬT/TẮT: service cục bộ luôn "có", tắt hẳn ARDY bằng ARDY_SERVICE="" (khoá cũ RUNPOD_API_KEY
# đã bỏ). Giữ tên `BAT` cho main.py hỏi trước khi trả trạng thái/nhận demo.
BAT = bool(DICH_VU)


def _khoa(prompt: str, dai_s: float, fps: float, neo: bool = False, noi: str = "") -> str:
    # `neo` VÀO KHOÁ: clip có neo hai đầu là clip KHÁC (bắt đầu và kết thúc ở tư thế nền), lẫn
    # với bản không neo trong cùng một ô cache là lúc bật lúc tắt tuỳ ai gọi trước.
    # `noi` = dấu vân của khung cuối motion TRƯỚC khi nối chuỗi (12/09) — clip nối vào đuôi khác là clip khác.
    return hashlib.sha1(
        (f"{MODEL}|{prompt.strip().lower()}|{dai_s:.2f}|{fps:.0f}|{int(bool(neo))}"
         + (f"|noi:{noi}" if noi else "")).encode()
    ).hexdigest()[:16]


# NỐI CHUỖI (12/09, user "logic đúng khi dùng ARDY là input frame cuối motion trước"): cue sau lấy `KHUNG_NOI`
# khung THÔ cuối của clip cue trước làm `history` → ARDY mở màn từ đúng tư thế + vận tốc đó. Khác neo (tư thế đứng
# cố định): đây là đuôi thật của motion trước, nên clip toàn thân cũng nối được; nối mà clip CHẾT (history v≈0 kéo
# sập cú chạy — 09/09) thì sinh lại độc lập một lần. Mỗi clip giữ đuôi thô ở `_khung_cuoi_tho` (8 khung × 27 quat).
KHUNG_NOI = int(os.getenv("ARDY_NOI_KHUNG", "8"))
# Trần số clip giữ trong RAM (mỗi clip ~2,0 MB dict Python → 16 ô ≈ 32 MB). Cache đĩa vẫn giữ mọi clip ĐỘC LẬP.
CACHE_RAM_MAX = int(os.getenv("ARDY_CACHE_RAM", "16"))


def lich_su_tai(clip: dict, t_cat: float | None = None, n: int = KHUNG_NOI) -> list | None:
    """`KHUNG_NOI` khung thô của `clip` KẾT THÚC TẠI `t_cat` (giây, mốc cục bộ clip) — làm `history` cho cue nối chuỗi.

    12/09 chiều (user "2 motion khác hướng, nhân vật trượt xoay cả người"): nối vào khung CUỐI trong khi clip trước
    bị cue sau CẮT ở giữa (vẫy tay cắt ở 2,1/3,98 s lúc hông 17°, cue sau nối từ khung cuối hông 37°) → mối nối vênh
    68,6°, làm liên xoay cả người 20° trong 1,6 s; clip xoay/chạy thì chênh tới 180°. Nối đúng tại điểm cắt thì khung
    đầu clip sau = khung đang hiển thị. Cần `_khung_tho` (mọi khung); thiếu thì lùi về `_khung_cuoi_tho` (đuôi).
    `t_cat=None` = cuối clip. Mốc t của lịch sử dời về 0 ở khung đầu.

    MỐC THEO CHỈ SỐ KHUNG, KHÔNG THEO `t` GHI TRONG KHUNG THÔ (12/09 tối): service trả clip theo MIẾNG 2,5 s và `t`
    của mỗi miếng lại chạy từ 0 (240 khung mà t chỉ tới 1,48 s; 44/44 clip cache không đơn điệu). Lọc `t ≤ t_cat`
    khi đó lấy 8 khung ở CUỐI miếng sau — "lịch sử tại 0,43 s" của cú xoay người thật ra là tư thế ĐÃ xoay xong
    (−160°), clip vẫy tay nối vào mở màn ở −160° trong khi cú xoay mới tới −36° → ống hoà 157° trong 1,8 s = nhân vật
    trượt xoay cả người. Khung thô cách đều `fps` của clip, chỉ số khung là mốc duy nhất đáng tin (`retarget` cũng
    đánh mốc theo chỉ số). Đo lại với lịch sử đúng: ARDY nối tuyệt đối, lệch 1–2,5°/khớp, hướng trùng.
    """
    kt = clip.get("_khung_tho")
    if kt:
        fps = float(clip.get("fps") or 60.0)
        k_cuoi = len(kt) - 1
        if t_cat is not None:
            k_cuoi = max(0, min(k_cuoi, int(math.floor(float(t_cat) * fps + 1e-6))))
        fr = kt[max(0, k_cuoi - n + 1):k_cuoi + 1]
        nj = (len(fr[0]) - 4) // 4
        return [{"t": round(k / fps, 4),
                 "data": {"root": [float(v) for v in f[1:4]],
                          "quats": [[float(v) for v in f[4 + 4 * j:8 + 4 * j]] for j in range(nj)]}}
                for k, f in enumerate(fr)]
    return clip.get("_khung_cuoi_tho") or None


def _van_noi(history: list) -> str:
    """Dấu vân của lịch sử nối (làm tròn 3 chữ số) cho khoá cache."""
    try:
        return hashlib.sha1(json.dumps(
            [[round(float(v), 3) for v in (f.get("data") or {}).get("root", [])]
             + [round(float(x), 3) for q in (f.get("data") or {}).get("quats", []) for x in q]
             for f in history]).encode()).hexdigest()[:12]
    except Exception:  # noqa: BLE001
        return "?"


BIEN_DO_TOI_THIEU = float(os.getenv("ARDY_BIEN_DO_MIN", "0.08"))   # m, max(Δ bàn tay P/T, Δ đầu)
# Dưới ngưỡng này là câu KHÔNG có trong kho catalog và bị khớp bừa (guide §10 đo được
# "chay lui ra sau" rơi vào "A person is doing the cha cha dance." ở 0,442) — ghi log cảnh báo.
SIM_CANH_BAO = float(os.getenv("ARDY_SIM_CANH_BAO", "0.75"))

# NEO HAI ĐẦU: ép mọi clip ARDY bắt đầu và kết thúc ở cùng một tư thế, để clip nào cũng ghép
# được vào clip nào. LUẬT, HÀNG RÀO TỰ KIỂM VÀ CÔNG TẮC nằm ở `ardy_neo.py` — client chỉ cấp
# đường đi tới service (user 09/09: "cấu trúc nên là dạng hook/module chuẩn ... chứ không gắn
# cứng vào 1 function"). Tệp giữ tư thế neo + kết quả tự kiểm:
_NEO_TEP = CACHE_DIR / "neo_chuan.json"


def _bien_do(clip: dict) -> float:
    """Biên độ chuyển động = max quãng FK của bàn tay trái/phải/đầu (m) — 0,02 là đứng yên,
    cử chỉ thật 0,2–1,0 (đo cache 07/09)."""
    try:
        import numpy as np
        from app.modules.motion.bo_phan.do_bo_phan import vi_tri_world
        ra = 0.0
        for b in ("RightHand", "LeftHand", "Head"):
            t, p = vi_tri_world(clip, (b,))
            p = np.asarray(p).reshape(len(t), -1)[:, :3]
            if len(p):
                ra = max(ra, float(np.linalg.norm(p.max(0) - p.min(0))))
        return ra
    except Exception:
        logger.warning("ARDY: không đo được biên độ", exc_info=True)
        return 1.0


class ArdyClient:
    def __init__(self) -> None:
        # MỘT luồng sinh: clip sau cần khung cuối clip trước khi nối chuỗi, và trên máy này
        # sinh song song chỉ làm cả hai cùng chậm (nút thắt là băng thông RAM — guide §7).
        self._tho = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ardy")
        self._info: dict | None = None          # /skeleton (joints, parents, rest_positions)
        self._khoa_info = threading.Lock()
        # CACHE RAM CÓ TRẦN (12/09, user "nhiều session lưu history thì tài nguyên cạn"): một clip dict = 2,0 MB;
        # bản cũ giữ MỌI clip đã sinh/đã nạp cho tới khi tiến trình chết. Nay LRU `CACHE_RAM_MAX` ô, trượt thì nạp
        # lại từ đĩa (746 KB JSON, ~10 ms). Clip NỐI CHUỖI (khoá theo đuôi cue trước, gần như không tái dùng) chỉ
        # ở RAM, không ghi đĩa — không thì mỗi lượt nói để lại vài MB đĩa mãi mãi.
        self._cache: "OrderedDict[str, dict]" = OrderedDict()
        self._tra_cache: dict[str, tuple] = {}   # prompt → (câu khớp, thời lượng, có lắng)
        self._neo_mod = None                     # mô-đun neo hai đầu (ardy_neo.Neo)
        self._khoa_neo = threading.Lock()
        self._kenh_ws = None                     # kênh WS thường trực (ardy_kenh.Kenh)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._san_sang: bool = False
        self._san_sang_luc: float = 0.0
        self._loi: str | None = None
        self._dung_luc: float = 0.0             # lần sinh gần nhất (cho trạng thái frontend)
        self._bat_luc: float = time.time()
        if BAT:
            threading.Thread(target=self._vong_suc_khoe, name="ardy-health", daemon=True).start()

    # ---------- HTTP ----------
    def _http(self):
        import httpx
        return httpx.Client(base_url=DICH_VU, timeout=httpx.Timeout(10.0, read=180.0))

    def _suc_khoe(self, timeout: float = 5.0) -> tuple[bool, str | None]:
        try:
            import httpx
            r = httpx.get(f"{DICH_VU}/health", timeout=timeout)
            d = r.json()
            return bool(d.get("ready")), d.get("error")
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {str(e)[:80]}"

    def _vong_suc_khoe(self) -> None:
        """Nền: giữ trạng thái mới cho `trang_thai()` — cửa HTTP của frontend gọi từ event loop
        nên KHÔNG được chặn ở đó (§6 CLAUDE.md: mọi khâu nặng ra khỏi loop)."""
        while True:
            ok, loi = self._suc_khoe()
            self._san_sang, self._loi = ok, loi
            if ok:
                self._san_sang_luc = time.time()
            time.sleep(10.0 if ok else 4.0)

    def _lay_info(self) -> dict:
        """`/skeleton` một lần: joints + parents + rest_positions cho `retarget` (guide §4 dặn
        lấy động, đừng chép cứng)."""
        with self._khoa_info:
            if self._info is None:
                import httpx
                d = httpx.get(f"{DICH_VU}/skeleton", timeout=30.0).json()
                if not d.get("parents") or not d.get("rest_positions"):
                    raise RuntimeError("/skeleton thiếu parents/rest_positions — cập nhật ardy_service")
                self._info = d
                logger.info("ARDY service %s · %d khớp · model %s @ %.0f fps gốc",
                            DICH_VU, len(d.get("joints") or []), d.get("model"), d.get("model_fps") or 0)
            return self._info

    def neo(self):
        """Mô-đun NEO HAI ĐẦU (`ardy_neo`). Client chỉ CẤP đường đi, không giữ luật của neo.

        Tách ra mô-đun riêng theo yêu cầu user 09/09 ("cấu trúc nên là dạng hook/module chuẩn để
        dễ kiểm soát và sửa chữa chứ không gắn cứng vào 1 function"): luật chọn tư thế neo, hàng
        rào tự kiểm và công tắc nằm gọn ở đó; đổi cách neo không phải đụng vào client.
        """
        if self._neo_mod is not None:
            return self._neo_mod
        with self._khoa_neo:
            if self._neo_mod is None:
                from app.modules.body_motion import ardy_neo as _n
                from app.modules.body_motion import ardy_song as _song
                # thước là TỐC ĐỘ (m/s) chứ không phải hộp bao — xem `ardy_song`
                self._neo_mod = _n.Neo(self._sinh_tho,
                                       lambda c: _song.do(c)["toc_tb"], _NEO_TEP)
                self._neo_mod.khoi_dong()
        return self._neo_mod

    def _sinh_tho(self, prompt: str, giay: float, history: list | None, dich: dict | None):
        """(khung ARDY thô, clip đã retarget) — đường đi mà `ardy_neo` dùng để dựng và tự kiểm.

        Không đi qua cache và KHÔNG đi qua tầng thử-lại-khi-đứng-yên: mô-đun neo cần số đo THẬT
        của đúng lần sinh đó, thử lại dài hơn là chấm nhầm clip khác.
        """
        from app.modules.body_motion.ardy_retarget import retarget, tu_mang
        info = self._lay_info()
        than = {"text": prompt, "seconds": float(giay), "fps": int(FPS),
                "chunk_seconds": max(float(giay), 2.5)}
        if history:
            than["history"] = history
        if dich:
            than["target_pose"] = dich
        with self._http() as h:
            r = h.post("/motion", json=than)
            r.raise_for_status()
            d = r.json()
        clips = d.get("clips") or []
        frames = [f for c in clips for f in (c.get("frames") or [])]
        if not frames:
            return [], None
        clip = retarget(info, tu_mang(list(clips[0].get("joints") or info["joints"]), frames),
                        float(clips[0].get("fps") or FPS), nguon="ardy")
        return frames, clip

    def _kenh(self):
        """Kênh WS thường trực. Chỉ dùng từ luồng sinh (`self._tho`) — Kenh không đa luồng."""
        if self._kenh_ws is None:
            self._kenh_ws = _kenh.Kenh(DICH_VU)
        return self._kenh_ws

    def dat_lai_phien(self) -> None:
        """PHIÊN MỚI: xoá lịch sử tư thế bên ARDY để cue đầu xuất phát từ tư thế đứng nghỉ.

        Không xoá thì cue đầu của phiên sau nối tiếp tư thế cuối của phiên TRƯỚC — nhân vật vừa
        tải lại trang đã ở giữa một động tác nào đó.
        """
        if not _kenh.BAT or self._kenh_ws is None:
            return
        try:
            self._tho.submit(self._kenh_ws.dat_lai).result(timeout=10)
        except Exception as e:  # noqa: BLE001
            logger.warning("ARDY WS: đặt lại phiên bỏ qua (%s)", str(e)[:90])

    def _don_neo(self, toan_than: bool = False) -> tuple[list | None, dict | None]:
        """(history, target_pose) cho một đơn — hỏi thẳng mô-đun neo."""
        try:
            return self.neo().don(toan_than)
        except Exception as e:  # noqa: BLE001
            logger.warning("ARDY neo: lấy đơn hỏng (%s) — chạy không neo", str(e)[:80])
            return None, None

    # ---------- đồng bộ (gọi từ provider) ----------
    def _nho_ram(self, k: str, clip: dict) -> None:
        """Đưa vào LRU RAM, đẩy ô cũ nhất ra khi quá `CACHE_RAM_MAX`."""
        self._cache[k] = clip
        self._cache.move_to_end(k)
        while len(self._cache) > CACHE_RAM_MAX:
            self._cache.popitem(last=False)

    def _tu_cache(self, k: str) -> dict | None:
        c = self._cache.get(k)
        if c is not None:
            self._cache.move_to_end(k)
        else:
            p = CACHE_DIR / f"{k}.json"
            if p.exists():
                try:
                    c = json.loads(p.read_text(encoding="utf-8"))
                    self._nho_ram(k, c)
                except Exception:
                    return None
        return json.loads(json.dumps(c)) if c is not None else None

    def _ghi_cache(self, k: str, clip: dict, dia: bool = True) -> dict:
        self._nho_ram(k, clip)
        if dia:
            try:
                (CACHE_DIR / f"{k}.json").write_text(json.dumps(clip), encoding="utf-8")
            except Exception:
                logger.warning("ARDY: không ghi cache", exc_info=True)
        return json.loads(json.dumps(clip))

    def bao_cache(self) -> dict:
        """Số ô RAM đang giữ + ước lượng MB — cho log dọn lượt."""
        return {"o": len(self._cache), "mb_uoc": round(2.0 * len(self._cache), 1), "tran": CACHE_RAM_MAX}

    def sinh_dong_bo(self, prompt: str, dai_s: float, timeout: float = 8.0,
                     history: list | None = None, neo: bool | None = None,
                     toan_than: bool = False, moi_mieng=None,
                     tran_s: float | None = None, noi_tiep: list | None = None) -> dict | None:
        """Clip đã retarget (định dạng kho) hoặc None. Cache trước, service sau.

        `neo=False` ép KHÔNG neo (dùng để đối chứng). Mặc định hỏi `ardy_neo` — mô-đun đó chỉ
        trả đơn khi đã TỰ KIỂM đạt, nên trước lúc kiểm xong hệ thống chạy y như cũ.
        `history` truyền tay được ưu tiên: đó là tư thế THẬT bên gọi đang hiển thị, sát hơn neo.
        `noi_tiep` = đuôi khung thô của clip cue TRƯỚC (`clip["_khung_cuoi_tho"]`) → NỐI CHUỖI: làm
        `history`, bỏ neo, khoá cache riêng theo dấu vân đuôi (xem `KHUNG_NOI`).
        """
        ls = dich = None
        noi = ""
        if noi_tiep:
            history, noi = list(noi_tiep), _van_noi(noi_tiep)
        if neo is not False and not noi:
            # NEO CHẠY TRÊN CẢ HAI ĐƯỜNG TRUYỀN. Bản đầu của kênh WS bỏ neo vì tưởng lịch sử
            # tư thế theo phiên đã lo — nhưng kênh nay `reset` mỗi cue (xem `ardy_kenh.Kenh.cue`)
            # nên KHÔNG còn nguồn liên tục nào khác. Liên tục = `history` bên gọi giao, hoặc neo.
            ls, dich = self._don_neo(toan_than)
        if history is not None:
            ls = history
        # khoá cache theo CÁI THẬT SỰ DÙNG: clip có neo là clip khác, lẫn chung một ô là lúc
        # bật lúc tắt tuỳ ai gọi trước.
        # TRẦN đi vào KHOÁ CACHE: cùng câu mà trần khác nhau là clip khác độ dài.
        k = _khoa(prompt, dai_s if dai_s > 0 else -float(tran_s or 0.0), FPS,
                  ls is not None and history is None, noi=noi)
        # CACHE VẪN DÙNG khi chạy kênh WS, vì kênh gửi `reset` kèm mỗi cue nên clip ĐỘC LẬP với
        # trạng thái phiên (xem `ardy_kenh.Kenh.cue`). Bản đầu tôi tắt cache để đổi lấy nối
        # chuỗi, nhưng đo ra nối chuỗi KHÔNG hợp ở đây — transformer là nền nên hai cửa sổ
        # thường cách nhau vài giây, nối vào tư thế cũ là nối sai chỗ; đổi lại mất cache thì mấy
        # câu đầu lượt hết đường băng (khách phải chờ 6/7 câu thay vì 5/6).
        from app.modules.body_motion import ardy_so_khop as _so_khop
        c = self._tu_cache(k)
        if c is not None:
            if moi_mieng is not None:
                moi_mieng(c, True)          # trúng cache = xong ngay, kho đầy luôn
            _so_khop.ghi_clip(prompt, c, SIM_CANH_BAO)
            return c
        if not BAT:
            logger.warning("ARDY: ARDY_SERVICE rỗng — bỏ qua")
            return None
        fut = self._tho.submit(self._sinh, prompt, dai_s, ls, dich, moi_mieng, tran_s, bool(noi))
        try:
            clip = fut.result(timeout=timeout)
        except Exception as e:  # noqa: BLE001
            logger.warning("ARDY: sinh '%s' hỏng/timeout %.1fs: %s", prompt[:40], timeout, str(e)[:100])
            fut.cancel()
            return None
        if clip:
            if clip.get("_dung_yen"):
                logger.warning("ARDY: clip đứng yên sau cả lần thử lại — không cache, không trộn")
                _so_khop.ghi("chet", cau=prompt, khop=clip.get("_khop"), sim=clip.get("_similarity"),
                             exact=clip.get("_exact"))
                return None
            ra = self._ghi_cache(k, clip, dia=not noi)    # nối chuỗi: RAM thôi, không để lại đĩa
            _so_khop.ghi_clip(prompt, clip, SIM_CANH_BAO)
            if moi_mieng is not None:
                moi_mieng(ra, True)         # miếng cuối: kho đã đủ
            return ra
        return None

    # ---------- đặt vé trước, lấy sau (song song với suy diễn transformer) ----------
    def bo_cache(self, prompt: str, dai_s: float) -> None:
        """Xoá cache một clip (provider phát hiện dị dạng sau ống — không giữ bản hỏng).

        Xoá CẢ HAI biến thể khoá (có neo / không neo): bên gọi không biết lúc nấu đã dùng neo
        hay chưa (mô-đun neo bật lên giữa chừng sau khi tự kiểm xong), nên chỉ xoá một biến thể
        là bản hỏng vẫn nằm lại trong ô kia và được bưng ra lần sau.
        """
        for co_neo in (False, True):
            k = _khoa(prompt, dai_s, FPS, co_neo)
            self._cache.pop(k, None)
            try:
                (CACHE_DIR / f"{k}.json").unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    # ---------- trạng thái ----------
    def wake(self) -> None:
        """Giữ tên cũ (main.py/provider gọi lúc mở trang và đầu lượt). Với service cục bộ không
        có gì để đánh thức — chỉ làm mới trạng thái và nạp `/skeleton` sẵn cho lần sinh đầu."""
        if not BAT:
            return
        self._bat_luc = self._bat_luc or time.time()

        def _am():
            ok, loi = self._suc_khoe()
            self._san_sang, self._loi = ok, loi
            if ok:
                self._san_sang_luc = time.time()
                try:
                    self._lay_info()
                except Exception:
                    logger.warning("ARDY: nạp /skeleton hỏng", exc_info=True)
        self._tho.submit(_am)

    def tra_cuu(self, prompt: str) -> tuple[str, float | None, bool | None]:
        """(câu KHỚP trong kho, thời lượng tự nhiên giây, có tự lắng) — `/catalog/lookup`.

        Bảng thời lượng đo sẵn cho cả kho 40.524 câu (09/09). Biết TRƯỚC động tác dài bao nhiêu
        thì ống thời gian xếp cửa sổ đúng và không cắt ngang cử chỉ — đo được KHÔNG câu nào ngắn
        hơn 4 giây (trung vị 5), trong khi ta vẫn xin 2,5–3 giây.
        Rẻ (tra bảng băm ~0,2 ms + HTTP nội bộ) và có cache nên gọi thoải mái lúc lập lịch.
        """
        c = self._tra_cache.get(prompt)
        if c is not None:
            return c
        ra: tuple[str, float | None, bool | None] = (prompt, None, None)
        if BAT:
            try:
                import httpx
                import urllib.parse
                d = httpx.get(f"{DICH_VU}/catalog/lookup?q={urllib.parse.quote(prompt)}",
                              timeout=10.0).json()
                dai = d.get("duration_s")
                ra = (str(d.get("matched") or prompt),
                      float(dai) if dai is not None else None, d.get("settled"))
            except Exception as e:  # noqa: BLE001
                logger.warning("ARDY tra cứu '%s' hỏng: %s", prompt[:36], str(e)[:60])
        self._tra_cache[prompt] = ra
        return ra

    def am_truoc(self, muc: list[tuple[str, float]]) -> None:
        """Sinh sẵn (nền) các clip hay dùng để lượt nói đầu tiên không phải chờ.

        Vì sao (08/09): bật cue cho MỌI câu thì cache lạnh làm thân câu chờ tới `TF_ARDY_TIMEOUT`
        mỗi câu — đo lượt 10 câu/43 s mất 72 s để dựng. Bộ cử chỉ lấp chỗ chỉ có 10 cái và dùng
        lại mãi, hâm một lần lúc khởi động (~25 s CPU lúc máy còn rảnh) là từ đó luôn trúng cache.
        """
        if not BAT:
            return

        def _mot(prompt: str, dai: float):
            # PHẢI HÂM ĐÚNG BẢN SẼ DÙNG: khoá cache có cờ neo, hâm bản không neo là nấu một
            # đằng bưng một nẻo — cache đầy mà lượt thật vẫn trượt.
            ls, dich = self._don_neo()
            k = _khoa(prompt, dai, FPS, bool(ls))
            if self._tu_cache(k) is not None:
                return
            try:
                clip = self._sinh(prompt, dai, ls, dich)
                if clip and not clip.get("_dung_yen"):
                    self._ghi_cache(k, clip)
            except Exception as e:  # noqa: BLE001
                logger.warning("ARDY hâm '%s' hỏng: %s", prompt[:40], str(e)[:80])
        for prompt, dai in muc:
            self._tho.submit(_mot, prompt, float(dai))

    def ping_ok(self, timeout: float = 8.0) -> bool:
        """Service trả `ready: true` thật sự (đồng bộ — gọi từ thread, không từ event loop)."""
        ok, _ = self._suc_khoe(timeout=timeout)
        if ok:
            self._san_sang, self._san_sang_luc = True, time.time()
        return ok

    def trang_thai(self) -> dict:
        """Cho badge ARDY trên frontend. Giữ nguyên các khoá mà `app.js` đang đọc
        (`bat`/`san_sang`/`dang_bat`/`giay_bat`); `dich_vu` + `loi` để soi khi service chết."""
        gio = time.time()
        return {"bat": BAT, "san_sang": bool(self._san_sang),
                "dang_bat": False,          # cục bộ: hoặc chạy, hoặc không — không có cold start
                "giay_bat": None,
                "giay_ranh": round(gio - self._dung_luc, 1) if self._dung_luc else None,
                "tat_sau_s": None, "max": None,
                "dich_vu": DICH_VU, "loi": None if self._san_sang else self._loi}

    def san_sang(self) -> bool:
        return bool(self._san_sang) and time.time() - self._san_sang_luc < 60.0

    # ---------- sinh ----------
    def _sinh(self, prompt: str, dai_s: float, history: list | None,
              dich: dict | None = None, moi_mieng=None,
              tran_s: float | None = None, noi_tiep: bool = False) -> dict | None:
        from app.modules.body_motion.ardy_retarget import retarget, tu_mang
        info = self._lay_info()
        # `dai_s <= 0` = KHÔNG ÉP SỐ GIÂY (user chốt 09/09: *"không được fix số giây, phải để
        # motion tự nhiên theo ARDY đã render"*). Không gửi `seconds` thì service chạy chế độ
        # `until="settle"`: nó tự đo mức chuyển động và đóng luồng đúng lúc động tác LẮNG. Động
        # tác tuần hoàn (chạy, nhảy múa) không bao giờ lắng nên vẫn cần trần an toàn.
        than = {"text": prompt, "fps": int(FPS)}
        if float(dai_s) > 0:
            than["seconds"] = float(dai_s)
        else:
            # TRẦN RIÊNG TỪNG CUE (09/09). Trần chung 10 s là quá rộng: đo trên một lượt thật,
            # MỌI cue đều chạy tới 4,98 s hoặc 7,48 s bất kể LLM ghi `:4` hay `:6`, trong khi
            # câu nói chỉ dài 1,4–4,3 s. Động tác vì thế không bao giờ diễn nốt trước câu sau,
            # dồn ứ nhau, và 5/7 cue của lượt không kịp về mốc ("kho còn rỗng"). Trần = ngân
            # sách LLM đã cân theo BẢNG THỜI LƯỢNG ĐO SẴN, nên vẫn là độ dài tự nhiên của chính
            # câu đó — không ép `seconds`, không kéo giãn, không cắt clip đã render.
            than["max_seconds"] = float(tran_s or os.getenv("ARDY_TRAN_GIAY", "10"))
            if tran_s:
                # BỘ DÒ LẮNG KHÔNG MANG THÔNG TIN — user chốt 10/09 sau khi đo: chỉ dùng trần.
                #
                # Đo trên 10 cue, sinh một lần rồi PHÁT LẠI quyết định của `ardy_service/settle.py`
                # với nhiều mức tham số (A/B tất định trên cùng khung):
                #     đổi FLOOR 0,05 → 0,90  : kết quả KHÔNG ĐỔI một giây nào
                #     đổi HOLD  0,4  → 3,0   : chỉ nhảy giữa 2,50 / 5,00 / 12,50 / 14,00 s
                #     đỉnh năng lượng thật của cue 0,29–1,11 °/khung, mà FLOOR = 0,9
                #                                  -> 6/8 cue có đỉnh NẰM DƯỚI sàn
                # Nó chỉ trả True tại CƠ HỘI ĐÁNH GIÁ ĐẦU TIÊN (khi bộ đệm đủ HOLD khung), nên
                # độ dài clip là bội số của kích thước miếng 2,5 s chứ không phải của động tác.
                # Dấu vết để lại: bảng `ardy_thoi_luong_tag.json` có 43/54 tag đúng 4,0 s.
                #
                # Và ARDY thật sự LẶP VÔ HẠN (đo hồ sơ năng lượng 14 s: "jumps up twice" nhảy
                # lại ở 4,0 · 6,5 · 9,5 s, giữa hai lần có thung lũng lặng ~1 s). HOLD = 3,0 s
                # dài hơn khoảng nghỉ đó nên không bao giờ bắt đúng chỗ.
                #
                # Để `settle` chạy thì nó còn cắt clip xuống 5,0 s một cách tuỳ tiện khi trần là
                # 6 s. Tắt đi thì trần của LLM là thứ DUY NHẤT quyết định — đúng thứ ta muốn.
                than["until"] = "forever"
        if history:
            than["history"] = history
        if dich:
            than["target_pose"] = dich
            # MỘT MIẾNG khi có neo: service mặc định cắt 2,5 s/miếng rồi nối chuỗi, mà ràng buộc
            # khung cuối chỉ đặt được ở miếng CUỐI → miếng đầu vẫn sinh kiểu cũ rồi miếng cuối
            # sinh lại kiểu toàn chuỗi. Xin trọn một miếng thì chỉ lấy mẫu MỘT lần.
            than["chunk_seconds"] = max(float(dai_s), 2.5)
        t0 = time.time()
        # KÊNH WS THƯỜNG TRỰC (user chốt 09/09). Gửi NGUYÊN thân đơn, `history` gồm: kênh
        # `reset` mỗi cue nên lịch sử tư thế theo phiên không còn là nguồn liên tục — bỏ
        # `history` đi là bỏ luôn phép neo đầu. `chain` ưu tiên `history` trong đơn hơn lịch sử
        # phiên nên hai đường truyền cho ra cùng một clip.
        c, clips = None, []
        if _kenh.BAT:
            k = self._kenh()

            def _mieng(khung, joints, fps_ws):
                """Miếng vừa về: retarget NGAY rồi đưa lên bên gọi (kho stream).

                Retarget ở đây chứ không ở bên gọi: `info`/`tu_mang` là việc của tệp này, đẩy
                ra ngoài là hai nơi cùng biết một phép đổi không gian.
                """
                if moi_mieng is None or not khung:
                    return
                # Kho khớp prompt ra động tác ĐỔI ĐỘ CAO thì KHÔNG đưa miếng nào lên — chặn ở
                # đây chứ không đợi cuối clip, vì miếng đầu đã vào kho stream và được diễn ngay.
                # `_khop_gan_nhat` được kênh ghi TRƯỚC khi gọi hàm này (xem `ardy_kenh.cue`).
                _mk_ = getattr(k, "_khop_gan_nhat", None)
                if _mk_:
                    from app.modules.body_motion.ardy_cue import doi_do_cao as _ddc
                    if _ddc(str(_mk_[0] or "")):
                        return
                cc = retarget(info, tu_mang(list(joints or info["joints"]), khung),
                              float(fps_ws or FPS), nguon="ardy")
                cc["source"] = f"ardy:{prompt}"
                moi_mieng(cc, False)          # False = chưa xong, còn miếng nữa

            try:
                frames_ws = k.cue(than, moi_mieng=(_mieng if moi_mieng is not None else None))
                _mk = getattr(k, "_khop_gan_nhat", None)
                if frames_ws and _mk:
                    c = {"matched": _mk[0], "similarity": _mk[1], "exact": _mk[2],
                         "fps": _mk[3], "joints": _mk[4]}
                    clips = [{"frames": frames_ws}]
            except Exception as e:  # noqa: BLE001
                logger.warning("ARDY WS: cue '%s' hỏng (%s) — lùi về HTTP lần này",
                               prompt[:40], str(e)[:110])
        if c is None:
            with self._http() as h:
                r = h.post("/motion", json=than)
                r.raise_for_status()
                d = r.json()
            clips = d.get("clips") or []
            if not clips:
                logger.warning("ARDY: service không trả clip cho '%s'", prompt[:40])
                return None
            c = clips[0]
        # GHÉP MỌI MIẾNG (08/09, service cập nhật): bản mới cắt luồng thành miếng
        # `MOTION_CHUNK_SECONDS` (2,5 s) — xin 4 s thì trả 2 clip (2,48 + 1,48). Bản cũ của ta
        # chỉ đọc `clips[0]` nên MỌI cửa sổ dài hơn 2,5 s bị cắt cụt còn 2,5 s.
        # Các miếng nối liền nhau thật sự: đo mối nối lệch 3,1° (đúng một khung của động tác
        # chạy) và gốc lệch 8 mm, `root` là toạ độ TUYỆT ĐỐI nên ghép thẳng theo thứ tự.
        # `retarget` đánh mốc theo chỉ số khung nên chỉ cần đúng thứ tự và cùng fps.
        frames: list = []
        for _c in clips:
            frames.extend(_c.get("frames") or [])
        if len(frames) < 10:
            logger.warning("ARDY: chỉ %d khung cho '%s'", len(frames), prompt[:40])
            return None
        # Kho catalog quyết câu THẬT SỰ được diễn: câu lạ bị khớp về câu gần nhất (guide §10).
        khop, sim = str(c.get("matched") or prompt), float(c.get("similarity") or 0.0)
        # KHO KHỚP RA ĐỘNG TÁC ĐỔI ĐỘ CAO: bỏ, không retarget, không cache. Prompt vô hại vẫn có
        # thể bị khớp sang câu leo/xuống ("A person goes up." → "goes right upwards", sim 0,90).
        from app.modules.body_motion.ardy_cue import doi_do_cao as _ddc
        if _ddc(khop):
            logger.warning("ARDY: '%s' khớp ra '%s' — động tác ĐỔI ĐỘ CAO, cảnh chỉ có mặt sàn "
                           "phẳng: BỎ", prompt[:40], khop[:60])
            from app.modules.body_motion import ardy_so_khop as _so_khop
            _so_khop.ghi("do_cao", cau=prompt, khop=khop, sim=sim, exact=bool(c.get("exact")))
            return None
        if not c.get("exact") and sim < SIM_CANH_BAO:
            logger.warning("ARDY: '%s' KHÔNG có trong kho, khớp bừa về '%s' (%.2f) — sửa prompt",
                           prompt[:40], khop[:50], sim)
        clip = retarget(info, tu_mang(list(c.get("joints") or info["joints"]), frames),
                        float(c.get("fps") or FPS), nguon="ardy")
        clip["source"] = f"ardy:{prompt}"
        clip["_khop"] = khop
        clip["_similarity"] = round(sim, 3)
        clip["_exact"] = bool(c.get("exact"))
        clip["_noi_tiep"] = bool(noi_tiep)
        # MỌI KHUNG THÔ (gọn: [t, root×3, quats×J×4] làm tròn 4 chữ số, ~190 KB JSON) để cue sau nối tại ĐIỂM CẮT
        # bất kỳ (`lich_su_tai`); kèm đuôi thô riêng cho đường lùi. MỐC t = CHỈ SỐ KHUNG / fps — `t` trong khung
        # service trả chạy lại từ 0 ở mỗi miếng 2,5 s (xem `lich_su_tai`), không dùng được làm mốc.
        fps_tho = float(c.get("fps") or FPS)
        try:
            clip["_khung_tho"] = [[round(k / fps_tho, 4), *[round(float(v), 4) for v in f["data"]["root"]],
                                   *[round(float(v), 4) for q in f["data"]["quats"] for v in q]]
                                  for k, f in enumerate(frames)]
        except Exception:  # noqa: BLE001
            logger.warning("ARDY: không giữ được khung thô của '%s'", prompt[:40], exc_info=True)
        # ĐUÔI THÔ để cue sau nối chuỗi (xem `KHUNG_NOI`): mốc t dời về 0 ở khung đầu đuôi
        try:
            duoi = frames[-KHUNG_NOI:]
            clip["_khung_cuoi_tho"] = [{"t": round(k / fps_tho, 4),
                                        "data": {"root": list(f["data"]["root"]),
                                                 "quats": [list(q) for q in f["data"]["quats"]]}}
                                       for k, f in enumerate(duoi)]
        except Exception:  # noqa: BLE001
            logger.warning("ARDY: không giữ được đuôi thô của '%s'", prompt[:40], exc_info=True)
        from app.modules.body_motion import ardy_song as _song
        _sd = _song.do(clip)
        # NỐI CHUỖI MÀ CHẾT → sinh lại ĐỘC LẬP một lần (09/09: history đứng yên kéo sập cú chạy 0,77 → 0,01).
        if noi_tiep and not _song.song(clip):
            logger.warning("ARDY nối chuỗi '%s' KHÔNG SỐNG (%s) — sinh lại độc lập, không nối",
                           prompt[:40], _song.mo_ta(clip))
            return self._sinh(prompt, dai_s, None, None, moi_mieng, tran_s, noi_tiep=False)
        bd = _bien_do(clip)      # giữ để đối chiếu trong log — KHÔNG dùng để quyết định nữa
        logger.info("ARDY sinh '%s' %.1fs: %d khung / %d miếng trong %.2fs (service %.0f ms) ·"
                    " khớp '%s' %.2f · lắng %s · biên độ tay/đầu %.2f m", prompt[:40], dai_s,
                    len(frames), len(clips), time.time() - t0,
                    sum(float(x.get("gen_ms") or 0.0) for x in clips), khop[:40], sim,
                    clips[-1].get("settled"), bd)
        # CLIP ĐỨNG YÊN (07/09: wave/bow 1,5 s Δ tay 2 cm, cache giữ luôn bản hỏng → user
        # "không thấy vẫy tay"): không cache, thử lại MỘT lần DÀI HƠN. Bản RunPod còn nối thêm
        # câu nhấn mạnh vào prompt — nay vô nghĩa: service khớp prompt về câu trong kho trước
        # khi sinh, thêm chữ chỉ làm khớp sang câu khác.
        # CHẤM BẰNG `ardy_song`, KHÔNG BẰNG HỘP BAO (09/09). Hộp bao chấm 0,73 cho clip đi 0,03 m
        # trong 4 giây — bàn tay bị clip trước để lại ở tư thế giơ cao nên hộp rộng, trong khi nó
        # không nhúc nhích. Cổng này vì thế KHÔNG BAO GIỜ NỔ ở đúng ca cần nổ, và clip chết đi
        # thẳng xuống client. Sinh lại là phản ứng đúng: phương sai giữa các mẫu của chính ARDY
        # là 2–3 lần, nên một mẫu mới rất dễ sống.
        # TẮT THỬ LẠI BẰNG `ARDY_THU_LAI=0` (user chốt 09/09 cho bản dựng mới). Mỗi lần thử lại
        # tốn TRỌN một lượt sinh ~1,03× thời lượng clip: đo trên 8010, cùng câu 'tilts head'
        # sinh ba lần 4,0 → 5,0 → 6,0 s hết 15,2 giây rồi vẫn không kịp cho câu nào. Với luồng
        # nói thật, thà bỏ cue đó còn hơn chiếm máy 15 giây.
        if not _song.song(clip):
            if os.getenv("ARDY_THU_LAI", "1") != "1":
                logger.info("ARDY '%s' KHÔNG SỐNG (%s) — BỎ, không thử lại",
                            prompt[:40], _song.mo_ta(clip))
                clip["_dung_yen"] = True
                return clip
            if dai_s < 5.5:
                logger.warning("ARDY '%s' KHÔNG SỐNG (%s · hộp bao %.2f m) — thử lại %.1fs",
                               prompt[:40], _song.mo_ta(clip), bd, dai_s + 1.0)
                return self._sinh(prompt, dai_s + 1.0, history, dich)
            clip["_dung_yen"] = True
        return clip


_CLIENT: ArdyClient | None = None


def client() -> ArdyClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = ArdyClient()
    return _CLIENT
