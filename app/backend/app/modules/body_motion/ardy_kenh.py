"""KÊNH WS THƯỜNG TRỰC TỚI ARDY — một kết nối, giữ suốt, 1:1.

User chốt 09/09: *"thiết kế backend luôn kết nối ws với ardy để stream dữ liệu liên tục, sau
này production cũng tương tự. Chịu tải nhiều nhất chính là client mở ws với backend, còn backend
1:1 ardy"*.

CÁI KÊNH MUA ĐƯỢC: bỏ bắt tay TCP + `/skeleton` mỗi cue, và **giữ sẵn khả năng nối chuỗi** —
service giữ lịch sử tư thế theo PHIÊN (`session_id` = chính kết nối này) nên cue sau CÓ THỂ nối
tiếp tư thế cuối của cue trước mà ta không phải gửi `history` kèm. Đo 09/09: nối tiếp làm lệch
tại mối nối 118° → 1,8°.

NHƯNG NỐI CHUỖI KHÔNG PHẢI MẶC ĐỊNH (đo lại 09/09, xem `cue`): transformer là NỀN nên phần lớn
cửa sổ ARDY cách nhau vài giây — nối cue N+1 vào đuôi cue N là nối vào tư thế nhân vật KHÔNG
còn ở đó. Mặc định mỗi cue gửi kèm `reset` → clip ĐỘC LẬP với trạng thái phiên → **cache clip
vẫn dùng được**. Bên gọi bật `noi_tiep=True` khi và chỉ khi hai cửa sổ thật sự KỀ NHAU.

ĐO ĐỘ MƯỢT WS vs HTTP (3 lần mỗi bên, 09/09): NGANG NHAU (5/6–7/8 câu phải chờ). Kênh không
làm ARDY sinh nhanh hơn — nút thắt là chính phép khuếch tán. Đừng kỳ vọng nó là tầng tăng tốc.

TUẦN TỰ LÀ BẢN CHẤT, KHÔNG PHẢI HẠN CHẾ: service tự khoá (`lock` trong `motion_ws`) vì chuỗi
đứt nếu hai cue chạy chồng. Ba chỗ phía ta cũng đã tuần tự sẵn (ThreadPoolExecutor(1) của
client, ProcessPoolExecutor(1) của bếp). WS không làm nhanh hơn — nó mua tính LIÊN TỤC.

Dùng `websockets.sync.client` (chặn) chứ không phải bản asyncio: mã gọi nằm trong luồng riêng
của `ardy_client`, kéo một event loop vào đó chỉ thêm chỗ để sai.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

BAT = os.getenv("ARDY_WS", "1") == "1"
MO_TIMEOUT = float(os.getenv("ARDY_WS_MO", "10"))       # giây, mở kết nối
CUE_TIMEOUT = float(os.getenv("ARDY_WS_CUE", "180"))    # giây, chờ trọn một cue


class Kenh:
    """Một kết nối WS, mở lười, tự nối lại. KHÔNG an toàn đa luồng — dùng từ MỘT luồng."""

    def __init__(self, dich_vu: str) -> None:
        # http://host:port -> ws://host:port/motion
        self.url = dich_vu.replace("https://", "wss://").replace("http://", "ws://") + "/motion"
        self._sk = None
        self._info: dict | None = None      # skeleton, service gửi ngay khi nối
        self._khoa = threading.Lock()
        self._so_noi = 0
        self._so_cue = 0

    # ---- kết nối ----------------------------------------------------------------
    def _mo(self):
        if self._sk is not None:
            return self._sk
        from websockets.sync.client import connect
        t0 = time.time()
        sk = connect(self.url, open_timeout=MO_TIMEOUT, max_size=None)
        # Bản tin ĐẦU TIÊN là `session` kèm luôn bộ xương — không phải gọi /skeleton riêng.
        d = json.loads(sk.recv(timeout=MO_TIMEOUT))
        if d.get("type") != "session":
            sk.close()
            raise RuntimeError(f"ARDY WS: chờ 'session', nhận {d.get('type')!r}")
        self._sk, self._info = sk, d
        self._so_noi += 1
        logger.info("ARDY WS: nối %s (phiên %s, %d khớp) trong %.2fs — lần nối thứ %d",
                    self.url, d.get("session_id"), len(d.get("joints") or []),
                    time.time() - t0, self._so_noi)
        return sk

    def dong(self) -> None:
        sk, self._sk, self._info = self._sk, None, None
        if sk is not None:
            try:
                sk.close()
            except Exception:  # noqa: BLE001
                pass

    def info(self) -> dict | None:
        """Bộ xương (joints/parents/rest_positions) service gửi lúc nối. None nếu chưa nối."""
        if self._info is None:
            try:
                self._mo()
            except Exception as e:  # noqa: BLE001
                logger.warning("ARDY WS: không mở được kênh (%s)", str(e)[:120])
                return None
        return self._info

    def dat_lai(self) -> None:
        """Xoá lịch sử tư thế của phiên — nhân vật về đứng nghỉ. Gọi khi bắt đầu PHIÊN MỚI."""
        with self._khoa:
            if self._sk is None:
                return
            try:
                self._sk.send(json.dumps({"type": "reset"}))
                # nuốt các bản tin còn tồn cho tới khi thấy `status`
                han = time.time() + 5.0
                while time.time() < han:
                    d = json.loads(self._sk.recv(timeout=5.0))
                    if d.get("type") in ("status", "error"):
                        break
                logger.info("ARDY WS: đã đặt lại lịch sử tư thế của phiên")
            except Exception as e:  # noqa: BLE001
                logger.warning("ARDY WS: đặt lại hỏng (%s) — đóng kênh, lần sau nối lại",
                               str(e)[:100])
                self.dong()

    # ---- gửi một cue ------------------------------------------------------------
    def cue(self, than: dict, noi_tiep: bool = False, moi_mieng=None) -> list[dict]:
        """Gửi một cue, gom mọi miếng tới khi service báo `end`. Trả danh sách khung THÔ.

        `noi_tiep=False` (mặc định) gửi kèm `reset`: cue này xuất phát từ tư thế NGHỈ, độc lập
        với cue trước. `noi_tiep=True` để service nối tiếp tư thế cuối của cue trước.

        VÌ SAO MẶC ĐỊNH LÀ KHÔNG NỐI (đo 09/09): transformer là NỀN, cửa sổ ARDY là những hòn
        đảo giữa nó. Hai cửa sổ cách nhau vài giây transformer mà cho ARDY nối tiếp nhau thì nó
        nối vào một tư thế nhân vật KHÔNG còn ở đó — sai chỗ nối. Nối chuỗi chỉ đúng khi hai cửa
        sổ KỀ NHAU (khe < 0,15 s, xem `ardy_lich`); đó là việc của bên gọi, không phải mặc định
        của kênh. Kèm theo: cue độc lập thì CACHE LẠI DÙNG ĐƯỢC, và cache chính là thứ giữ được
        đường băng cho mấy câu đầu lượt.

        Ngoại lệ được ném ra ngoài SAU KHI đóng kênh: kênh hỏng giữa chừng thì trạng thái phiên
        không còn tin được nữa, nối lại từ đầu sạch hơn là đoán xem còn dùng được không.
        """
        with self._khoa:
            try:
                sk = self._mo()
                goi = {"type": "cue", **than}
                if not noi_tiep:
                    goi["reset"] = True      # `chain` tự xoá lịch sử phiên trước khi sinh
                sk.send(json.dumps(goi, ensure_ascii=False))
                khung: list[dict] = []
                joints = None
                han = time.time() + CUE_TIMEOUT
                while True:
                    con = han - time.time()
                    if con <= 0:
                        raise TimeoutError("quá %.0fs chưa thấy 'end'" % CUE_TIMEOUT)
                    d = json.loads(sk.recv(timeout=con))
                    t = d.get("type")
                    if t == "clip":
                        khung.extend(d.get("frames") or [])
                        joints = joints or d.get("joints")
                        self._khop_gan_nhat = (d.get("matched"), d.get("similarity"),
                                               d.get("exact"), d.get("fps"), joints)
                        # MIẾNG VỀ TỚI ĐÂU DÙNG TỚI ĐÓ (user chốt 09/09): service cắt luồng
                        # thành miếng `CHUNK_SECONDS` (2,5 s) để bên gọi phát được ngay miếng
                        # đầu — trước đây ta gom hết rồi mới trả, tức vứt đúng cái lợi đó.
                        # Clip 4 s: miếng 1 xong sau ~2,6 s mà ta đợi tới ~4,4 s.
                        if moi_mieng is not None:
                            try:
                                moi_mieng(list(khung), joints, d.get("fps"))
                            except Exception:  # noqa: BLE001
                                logger.warning("ARDY WS: xử lý miếng hỏng", exc_info=True)
                    elif t == "end":
                        self._so_cue += 1
                        return khung
                    elif t == "error":
                        raise RuntimeError("ARDY WS lỗi: %s" % str(d.get("message"))[:160])
                    # `pong`/`status` lạc vào thì bỏ qua
            except Exception:
                self.dong()
                raise

    def trang_thai(self) -> dict:
        return {"bat": BAT, "noi": self._sk is not None, "so_lan_noi": self._so_noi,
                "so_cue": self._so_cue, "url": self.url}
