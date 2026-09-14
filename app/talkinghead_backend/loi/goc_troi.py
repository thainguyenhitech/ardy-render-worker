"""GỐC TRÔI — hết động tác thì nhân vật ĐỨNG LẠI ĐÓ, không bị kéo về chỗ cũ.

User chốt 10/09: *"không cần kéo về vị trí ban đầu khi kết thúc motion, cứ cho nhân vật đứng
tại điểm kết thúc đó luôn; còn nếu chuyển động kết thúc mà nhân vật đang quay mặt hướng khác
thì cho camera di chuyển qua thẳng mặt nhân vật chứ không xoay nhân vật lại chỗ cũ."*

VẤN ĐỀ: nền transformer luôn sinh quanh THẾ ĐỨNG CHUẨN tại gốc toạ độ, còn clip ARDY mang gốc
riêng của nó (`hips_pos = root − root_khung0`, tức cũng xuất phát từ 0). Nên khi đoạn ARDY kết
thúc và nền đắp lại, nhân vật bị giật về gốc — dời bao nhiêu, xoay bao nhiêu trong lúc diễn
thì trả lại bấy nhiêu.

NGHIỆM — ĐỔI GỐC ĐỘNG (Dynamic Relative Origin): giữ một phép biến đổi tích luỹ `Goc`
(tịnh tiến ngang + xoay quanh trục đứng) rồi áp cho MỌI khung. Hết một đoạn ARDY thì gốc
CỘNG THÊM đúng phần đoạn đó vừa dời được. Từ đó nền chảy tiếp quanh chỗ mới, và đoạn ARDY sau
cũng xuất phát từ chỗ mới.

    khung ARDY   : world = Goc ⊕ (gốc riêng của clip)
    hết đoạn     : Goc ← Goc ⊕ (gốc riêng ở khung CUỐI của đoạn)
    khung nền    : world = Goc ⊕ (thế đứng chuẩn)

KHÔNG ĐỘNG TỚI PHƯƠNG ĐỨNG. Sàn là cố định; chỉ x/z và góc quay quanh y trôi theo. Trộn cả y
vào đây là phá `neo_san`.

QUAN HỆ VỚI `chuyen_tiep`: chạy TRƯỚC nó. Sau khi đổi gốc, vị trí và hướng ở mép đoạn đã liên
tục, nên tầng chuyển tiếp chỉ còn phải tắt phần TƯ THẾ — độ lệch nhỏ hơn hẳn, T ngắn hơn.

LUẬT SỐ 1 CÒN NGUYÊN: đây là phép đổi hệ quy chiếu áp đều cho cả khung, không phải trộn hai
nguồn. Mỗi khung vẫn là nội dung của đúng một nguồn, chỉ nhìn từ một gốc khác.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

from loi.dong_thoi_gian import _euler_sang_q, _q_sang_euler

logger = logging.getLogger(__name__)

BAT = os.getenv("TH_GOC_TROI", "1") != "0"


def _q_nhan(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def _q_yaw(g: float):
    return (math.cos(g / 2.0), 0.0, math.sin(g / 2.0), 0.0)


def _yaw_cua(q) -> float:
    """Góc quay quanh trục đứng của một quaternion: lấy hướng NHÌN rồi chiếu xuống mặt sàn.

    Không đọc thành phần y của Euler: với Euler XYZ nó KHÔNG phải yaw khi có pitch/roll
    (§4 bẫy 1 — Euler không phải thứ để đọc thẳng).
    """
    w, x, y, z = q
    # quay véc-tơ (0,0,1) bởi q
    fx = 2.0 * (x * z + w * y)
    fz = 1.0 - 2.0 * (x * x + y * y)
    return math.atan2(fx, fz)


@dataclass
class Goc:
    """Phép biến đổi tích luỹ: xoay `yaw` quanh trục đứng rồi tịnh tiến (dx, dz)."""

    dx: float = 0.0
    dz: float = 0.0
    yaw: float = 0.0

    def ap_diem(self, x: float, z: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.dx + c * x + s * z, self.dz - s * x + c * z)

    def hop(self, x: float, z: float, yaw: float) -> "Goc":
        """Gốc mới = gốc này ⊕ trạng thái cục bộ (x, z, yaw)."""
        nx, nz = self.ap_diem(x, z)
        return Goc(nx, nz, self.yaw + yaw)

    @property
    def dang_troi(self) -> bool:
        return abs(self.dx) > 1e-4 or abs(self.dz) > 1e-4 or abs(self.yaw) > 1e-4


def hop_cuoi_clip(goc: Goc, clip: dict) -> Goc:
    """Kết sổ một đoạn bằng KHUNG CUỐI của chính clip nó — dùng cho đoạn đã hết TRƯỚC cửa sổ.

    Đoạn kết đúng tại biên hai câu thì KHÔNG xuất hiện khung nào trong cửa sổ nào cả: câu trước
    dừng trước mốc kết (nó còn đang diễn), câu sau bắt đầu TỪ mốc kết (nó hết rồi). Vòng lặp
    theo khung của `ap` vì thế không bao giờ thấy nó thôi diễn. Đo 10/09: 'side_step_right'
    2,48 s đưa nhân vật tới −1,65 m rồi biến mất khỏi sổ — gốc vẫn 0 nên câu sau nhảy về 0.
    """
    hp = clip.get("hips_pos") or []
    hips = (clip.get("bones") or {}).get("Hips") or []
    if not hp or not hips:
        return goc
    yaw = _yaw_cua(_euler_sang_q(hips[-1][1], hips[-1][2], hips[-1][3]))
    return goc.hop(float(hp[-1][1]), float(hp[-1][3]), yaw)


def hop_tai(goc: Goc, clip: dict, t_cb: float) -> Goc:
    """Kết sổ một đoạn tại MỐC `t_cb` của chính clip nó (khung cuối có mốc ≤ t_cb).

    Dùng cho đoạn bị cue mới ĐÈ NGAY KHUNG ĐẦU của một cửa sổ trong khi nó vẫn còn phủ mốc đó —
    nó không hiện ở khung nào của cửa sổ mới, nên vòng lặp theo khung của `ap` không bao giờ
    thấy nó thôi diễn, còn `hop_cuoi_clip` thì chỉ tới lượt khi nó "hết giờ" ở câu SAU (và khi
    đó lấy khung CUỐI clip, không phải khung cuối còn hiển thị). Đo 10/09: 'pivot_heel_think'
    (xoay ~169°) tràn 0,02 s sang câu có 'walk_forward' đặt ngay đầu — cú đi bộ mở màn từ chỗ
    và hướng TRƯỚC khi xoay (đợt trượt 33–45 cm), sang câu sau gốc mới cộng pha xoay và quay
    luôn cả quãng đi bộ: lệch 2,5 m.
    """
    hp = clip.get("hips_pos") or []
    hips = (clip.get("bones") or {}).get("Hips") or []
    if not hp or not hips:
        return goc
    k = max((i for i, f in enumerate(hips) if f[0] <= t_cb + 1e-6), default=0)
    kh = max((i for i, f in enumerate(hp) if f[0] <= t_cb + 1e-6), default=0)
    yaw = _yaw_cua(_euler_sang_q(hips[k][1], hips[k][2], hips[k][3]))
    return goc.hop(float(hp[kh][1]), float(hp[kh][3]), yaw)


def ap(clip: dict, ket_tai: list, goc: Goc, goc_khung: list | None = None) -> Goc:
    """Áp gốc trôi lên `clip` tại chỗ. Trả GỐC MỚI sau khi cộng phần các đoạn ARDY vừa THÔI DIỄN.

    `ket_tai` = CHỈ SỐ KHUNG cuối cùng mà mỗi đoạn ARDY còn được hiển thị. Bên gọi phải tự biết
    — chỉ nó mới thấy đoạn nào bị cue sau đè ngang; xem `OngStream.chay`.

    KHÔNG SUY RA TỪ `nguon_khung`, hai lần sai vì lý do khác nhau (đo 10/09):
      · bản 1 dùng `nguon_khung[i]=="ardy" and (i+1>=n or ...)`, mà `i+1>=n` nổ cả khi HẾT CỬA
        SỔ CÂU — clip tràn sang câu sau bị cộng hai lần: hông nhảy (2,24 · −9,89) → (4,47 ·
        −19,76) trong 0,1 s, ĐÚNG GẤP ĐÔI.
      · bản 2 dùng mốc t1 lý thuyết, nên đoạn bị cue mới ĐÈ NGANG không bao giờ được cộng —
        nhân vật nhảy về gốc của clip mới (−2,50 m → 0) rồi trượt.
    """
    if not BAT:
        return goc
    b = clip.get("bones") or {}
    hp = clip.get("hips_pos") or []
    hips = b.get("Hips")
    if not hips:
        return goc
    n = min(len(hips), len(hp) if hp else len(hips))
    ket = {i for i in (ket_tai or []) if 0 <= i < n}
    gan = None
    for i in range(n):
        # `goc_khung`: gốc ĐANG DÙNG cho khung i (trước lúc kết sổ ở khung này) — bước chân của nền
        # cần nó để đặt thế đứng nghỉ đúng chỗ, vì gốc có thể đổi GIỮA câu
        if goc_khung is not None:
            goc_khung.append((goc.dx, goc.dz, goc.yaw))
        # trạng thái CỤC BỘ của khung này (đọc trước khi ghi đè)
        q = _euler_sang_q(hips[i][1], hips[i][2], hips[i][3])
        yaw_cb = _yaw_cua(q)
        x_cb = float(hp[i][1]) if hp else 0.0
        z_cb = float(hp[i][3]) if hp else 0.0
        if goc.dang_troi:
            nx, nz = goc.ap_diem(x_cb, z_cb)
            if hp:
                hp[i][1], hp[i][3] = nx, nz
            e = _q_sang_euler(_q_nhan(_q_yaw(goc.yaw), q), gan)
            hips[i][1], hips[i][2], hips[i][3] = e
            gan = e
        else:
            gan = hips[i][1:4]
        # đoạn ARDY THẬT SỰ kết ở khung này -> gốc cộng thêm phần nó vừa dời được
        if i in ket:
            goc = goc.hop(x_cb, z_cb, yaw_cb)
            logger.info("gốc trôi: đoạn ARDY kết ở (%.2f, %.2f)m xoay %.0f° — nhân vật ĐỨNG LẠI ĐÓ",
                        goc.dx, goc.dz, math.degrees(goc.yaw))
    if goc.dang_troi:
        clip["_goc_troi"] = [round(goc.dx, 4), round(goc.dz, 4), round(goc.yaw, 4)]
    return goc
