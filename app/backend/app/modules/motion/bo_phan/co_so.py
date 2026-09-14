"""KHUNG BỘ PHẬN — mỗi bộ phận là một ống nhỏ: danh sách xương, điểm đo, và
chuỗi HOOK chạy theo thứ tự. Thêm hành vi mới = viết một hook, gắn vào bộ phận;
không sửa ống chính.

    @dau.hook
    def cui_khi_noi(clip, t, bp, ctx): ...

Hook nhận:
    clip  — clip đang xử lý (sửa TẠI CHỖ)
    t     — tham số của RIÊNG bộ phận này (đã gộp mặc định + profile.ong.<tên>)
    bp    — chính bộ phận (xuong, diem_do, ten)
    ctx   — ngữ cảnh chung: t_chung, nguon, dong_bo_tieng, yeu_cau_gian, loc_rieng, ghi
Hook KHÔNG tự co giãn thời gian — chỉ báo vào ctx["yeu_cau_gian"]; tầng làm mềm
quyết một lần cho cả clip, vì giãn là việc của cả thân, không của một bộ phận.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BoPhan:
    ten: str
    xuong: tuple[str, ...]          # xương thuộc bộ phận
    diem_do: tuple[str, ...]        # ngọn chuỗi — đo vị trí world ở đây
    mac_dinh: dict = field(default_factory=dict)
    hooks: list[Callable] = field(default_factory=list)
    # QUY ĐỔI KHOÁ CŨ -> tham số bộ phận, chạy CUỐI `tham_so` nên MỌI nơi đọc
    # tham số đều thấy cùng một con số. Trước 2026-08-28 việc quy đổi nằm trong
    # hook của tay, nên `lam_mem._ti_le_vot` (đọc thẳng bp.tham_so) dùng trần
    # KHÁC hook tay: elderly hook 13,5 vs lam_mem 24,0 (lỏng 78%), child hook
    # 28,5 vs lam_mem 24,0 (chặt 16% — giảm tốc chính chuyển động mà ống tay
    # cho là đạt). Hai thước cùng tên đo hai thứ khác nhau, CLAUDE.md §3.
    hop_nhat: Callable | None = None

    def hook(self, fn: Callable) -> Callable:
        """Trang trí để đăng ký hook, giữ thứ tự khai báo."""
        self.hooks.append(fn)
        return fn

    def tham_so(self, t_chung: dict) -> dict:
        """mặc định của bộ phận <- profile.ong.<tên> (người dùng ghi đè)."""
        t = dict(self.mac_dinh)
        t.update(nap_nguong(self.ten))           # TRẦN: nguong_bo_phan.json là nguồn duy nhất
        t.update((t_chung or {}).get(self.ten) or {})
        if self.hop_nhat is not None:
            t = self.hop_nhat(t, t_chung or {})
        return t

    def chay(self, clip: dict, ctx: dict) -> None:
        t = self.tham_so(ctx["t_chung"])
        for h in self.hooks:
            h(clip, t, self, ctx)


def quan_tinh_hook(hz_mac_dinh: float):
    """Sinh hook QUÁN TÍNH cho một bộ phận. Dùng chung cho MỌI bộ phận.

    NGUYÊN LÝ: chuyển động qua ống, dù nguồn cứng đến đâu, đầu ra phải MỀM. Mềm
    không phải là "ít giật" — đo ngày 2026-08-22 cho thấy lưng đã có giật THẤP
    HƠN người thật, phổ tần trùng mocap, nhịp đặc trưng 0,94 so với 1,01 mà nhìn
    vẫn như khối gỗ. Cái thiếu là QUÁN TÍNH: bộ phận bám nguồn TỨC THỜI thì mọi
    đổi hướng xảy ra cùng lúc trên khắp cơ thể. Vật có khối lượng thì vào chậm
    hơn, ra chậm hơn, và trôi thêm một chút sau khi nguồn đã dừng.

    Tầng lam_mem KHÔNG làm việc này: nó GIẢM TỐC (kéo giãn trục thời gian quanh
    chỗ vọt). Quỹ đạo cứng phát chậm lại thì vẫn cứng — không trễ, không dư đà.

    Mô hình bậc hai giảm chấn, tích phân theo từng xương:

        y'' = w^2 (x - y) - 2*zeta*w*y'          w = 2*pi*hz

    zeta < 1 cho DƯ ĐÀ nhỏ (follow-through) — thứ làm chuyển động "có thịt".

    CHỌN hz: phải nằm TRÊN dải chuyển động thật của bộ phận, nếu không sẽ dập
    chính chuyển động chứ không phải làm mềm nó. Đo nhịp đặc trưng trên 6 clip
    mocap BEAT: thân 1,01 · đầu 1,05 · hông 0,99 · tay 1,10 · vai 1,13 Hz —
    gần như đồng đều ~1 Hz. Nên hz >= 2x mức đó. Khác biệt giữa các bộ phận đến
    từ KHỐI LƯỢNG: tay nhẹ bám sát (hz cao), hông/thân nặng trôi sau (hz thấp).
    """
    def quan_tinh(clip, t, bp, ctx):
        # vòng sống là clip OFFSET quanh stance — thêm trễ vào đó là lệch pha
        # với nguồn tuyệt đối đang chạy cùng lúc
        if str(ctx.get("nguon") or "").startswith("song"):
            return
        # lượt hai của lam_mem chỉ ĐO: đây là phép BIẾN ĐỔI, áp kép là trễ gấp
        # đôi (CLAUDE.md §10)
        if ctx.get("chi_do"):
            return
        # CLIP DI CHUYỂN (ARDY toàn thân: chạy/bước/nhảy) — nhịp sải chân 2–3 Hz nằm NGAY TRÊN
        # hz của hook (chân 2,4 · hông 1,6), tức đúng cái docstring cảnh báo "hz phải trên dải
        # chuyển động thật, không thì dập chính chuyển động". Đo clip chạy vòng tròn 07/09:
        # đùi 59°→43°, quãng bàn chân so hông 11,2→7,5 m, trong khi hông giữ nguyên 3,80 m →
        # chân bước ngắn mà người vẫn trôi xa = "chạy như bị lướt" (user 14:12).
        if clip.get("toan_than") and str(ctx.get("nguon") or "").startswith("ardy"):
            return
        import numpy as np
        hz = float(t.get("quan_tinh_hz", hz_mac_dinh) or 0.0)
        if hz <= 0:
            return
        zeta = float(t.get("quan_tinh_giam_chan", 0.85))
        zeta = min(max(zeta, 0.05), 0.999)
        b = clip.get("bones") or {}
        w = 2.0 * math.pi * hz
        wd = w * math.sqrt(1.0 - zeta * zeta)
        # LỌC TRÊN QUATERNION, KHÔNG lọc thẳng Euler.
        #
        # Bẫy #1 (CLAUDE.md §4) — sập lại ngày 2026-08-23, lần này ở chính bộ
        # lọc này. LeftUpLeg có trục z chạy [-179,8 ; +179,9]: Euler quấn qua
        # ±180 nên bộ lọc thấy cú nhảy 360 do và nổ. Hậu quả đo được: bàn chân
        # của TalkingFearTwo từ dao động 2,6 cm thành 179,7 cm — chân bay lên
        # ngang đầu. Chỉ lộ ra ở CHÂN vì thân/đầu/hông không nằm gần ±180.
        from app.modules.motion.hinh_hoc import _euler_q_vec, _q_euler_vec
        ten_xuong = [x for x in bp.xuong if b.get(x) and len(b[x]) >= 4]
        if not ten_xuong:
            return
        mang = {x: np.array(b[x], dtype=float) for x in ten_xuong}
        moc = mang[ten_xuong[0]][:, 0]
        if any(len(mang[x]) != len(moc) for x in ten_xuong):
            return                      # lưới lệch nhau -> bỏ, đừng đoán
        gop = []
        for x in ten_xuong:
            a = mang[x]
            q = _q_euler_vec(a[:, 1], a[:, 2], a[:, 3])
            doi = np.sign(np.einsum("ij,ij->i", q[1:], q[:-1]))
            doi[doi == 0] = 1.0
            q[1:] *= np.cumprod(doi)[:, None]      # ép dấu liên tục
            gop.append(q)
        x = np.concatenate(gop, axis=1)            # (T, n*4)
        y = x[0].copy()
        v = np.zeros(x.shape[1])
        ra = np.empty_like(x)
        ra[0] = y
        for i in range(1, len(moc)):
            h = float(moc[i] - moc[i - 1])
            if h <= 1e-9:
                ra[i] = y
                continue
            # NGHIỆM GIẢI TÍCH của e'' + 2*zeta*w*e' + w^2*e = 0 (e = y - x),
            # coi x hằng trong mỗi khung: chính xác tuyệt đối, ổn định vô điều
            # kiện -> MỘT bước mỗi khung, không cần chia nhỏ.
            e0 = y - x[i]
            tat = math.exp(-zeta * w * h)
            c, sn = math.cos(wd * h), math.sin(wd * h)
            e = tat * (e0 * c + (v + zeta * w * e0) * (sn / wd))
            v = tat * (v * c - (w * w * e0 + 2.0 * zeta * w * v) * (sn / wd))
            y = x[i] + e
            ra[i] = y
        for k, x2 in enumerate(ten_xuong):
            q = ra[:, 4 * k:4 * k + 4]
            q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
            e = np.unwrap(_euler_q_vec(q), axis=0)
            for i, f in enumerate(b[x2]):
                f[1], f[2], f[3] = float(e[i, 0]), float(e[i, 1]), float(e[i, 2])
    return quan_tinh


def yeu_cau_gian(ctx: dict, bp: "BoPhan", ly_do: str, he_so: float) -> None:
    """Bộ phận xin giãn thời gian `he_so` lần. Tầng làm mềm lấy MAX."""
    if he_so > 1.0 + 1e-3:
        ctx.setdefault("yeu_cau_gian", []).append((bp.ten, ly_do, float(he_so)))


def yeu_cau_loc(ctx: dict, bp: "BoPhan", ly_do: str, xuong: tuple[str, ...] | None = None) -> None:
    """Bộ phận xin lọc RIÊNG các xương của nó (giật vị trí, rung)."""
    ctx.setdefault("loc_rieng", {})[bp.ten] = (ly_do, tuple(xuong or bp.xuong))


_NGUONG = None


def nap_nguong(ten: str) -> dict:
    """Trần của bộ phận từ bo_phan/nguong_bo_phan.json (hiệu chuẩn trên lưới
    chuẩn; tay_trai/tay_phai dùng khoá "tay"). Khoá bắt đầu "_" là tài liệu."""
    global _NGUONG
    if _NGUONG is None:
        import json
        from pathlib import Path
        try:
            _NGUONG = json.loads((Path(__file__).parent / "nguong_bo_phan.json").read_text())
        except Exception:
            _NGUONG = {}
    k = "tay" if ten.startswith("tay") else ten
    return {a: b for a, b in (_NGUONG.get(k) or {}).items() if not a.startswith("_")}


def trong_so_cua_so(clip: dict, tm: float) -> float:
    """Trọng số CỬA SỔ TOÀN THÂN tại mốc tm — 0 nếu ngoài mọi cửa sổ.

    Cửa sổ do gru/blend.tron ghi vào clip["_cua_so_toan_than"] dưới dạng
    [t0, t1, len_s, xuong_s]: đoạn một cử chỉ TOÀN THÂN (có chân + hips_pos)
    được trộn vào câu nói. Trong cửa sổ, các hook "chuẩn hoá giọng nói"
    (reset chân, kẹp hông, thu dao động lưng, ghim chân về nghỉ) phải NHƯỜNG
    theo trọng số này — không thì cử chỉ toàn thân render xong chỉ còn tay:
    đo 2026-08-23, hips/chân của clip cử chỉ bị vứt ở XUONG_TRON, phần lưng
    bị dao_dong_he thu còn 25%.

    Hình thang vai trơn (smoothstep) — cùng dạng với blend._trong_so.
    """
    cua = clip.get("_cua_so_toan_than") or []
    w = 0.0
    for c in cua:
        t0, t1 = float(c[0]), float(c[1])
        ln = float(c[2]) if len(c) > 2 else 0.45
        xg = float(c[3]) if len(c) > 3 else 0.55
        if tm < t0 or tm > t1 + xg:
            continue
        if tm < t0 + ln:
            u = (tm - t0) / max(ln, 1e-6)
        elif tm <= t1:
            u = 1.0
        else:
            u = 1.0 - (tm - t1) / max(xg, 1e-6)
        u = min(1.0, max(0.0, u))
        w = max(w, u * u * (3.0 - 2.0 * u))
    return w
