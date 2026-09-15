"""BƯỚC CHÂN — đổi chỗ đứng bằng BƯỚC THẬT, không để bàn chân lướt trên sàn.

User 10/09: *"đề xuất thử giải pháp cho chân bước tự nhiên khi chuyển vị trí thay vì trượt"* →
chốt phương án A (bước thủ tục) + *"tạo function xử lý bước chân riêng biệt để khi cần chỉ cần
truyền tham số vào và chân di chuyển"*.

VÌ SAO CẦN (đo 10/09 trên 13 gói thật của 8010): hết một đoạn ARDY, tầng chuyển tiếp đưa khớp
chân từ tư thế cuối của ARDY về tư thế đứng của transformer trong ≤1,8 s mà KHÔNG giữ bàn chân,
nên bàn chân LƯỚT 7–30 cm. Transformer đứng một mình chỉ xê dịch 1,1–1,6 cm/câu.

HAI CỬA DÙNG:

    di_chuyen_chan(clip, [Buoc(...), ...])     tự chỉ định từng bước: chân nào, lúc nào, bao lâu,
                                               tới đâu, nhấc cao bao nhiêu
    buoc_chan(clip, t0, t1, dich=None, ts=…)   tự lên kế hoạch: giữ hai bàn chân tại chỗ lúc t0,
                                               bước tới chỗ NGUỒN muốn lúc t1 (hoặc tới `dich`)

CÁCH LÀM — chỉ ghi đè tư thế CHÂN trong đúng cửa sổ bước, ngoài cửa sổ không đụng:

    · chân TRỤ đứng yên đúng chỗ (vị trí + hướng lúc bắt đầu cửa sổ)
    · chân BƯỚC đi theo quỹ đạo min-jerk theo phương ngang + chuông 16u²(1−u)² theo phương đứng,
      hướng bàn chân slerp cùng nhịp
    · hông dồn nhẹ sang chân trụ trong lúc chân kia nhấc (chuyển trọng tâm)
    · IK hai xương CÓ SẴN (`build_clip.ik_hai_xuong` — trộn trục gập liên tục, không nhảy nhánh)
      giải đùi + cẳng chân; bàn chân đặt đúng hướng kế hoạch
    · hết bước: trọng số tắt dần bậc 5 trong `tat` giây. Với `buoc_chan` lúc đó bàn chân đã ở
      đúng chỗ nguồn muốn nên phần hiệu chỉnh ≈ 0 — trả lại cho nguồn không có cú giật
    · làm mượt TRƯỜNG HIỆU CHỈNH chứ không làm mượt tư thế, như `ghim_chan_cuoi` (CLAUDE.md §2)

KHÔNG PHẢI GHIM CHÂN (user chốt "tắt hết ghim"): chỉ chạy trong cửa sổ bên gọi truyền vào.
Công tắc đối chứng: `TH_BUOC_CHAN=0`.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from loi.dong_thoi_gian import _q_sang_euler

logger = logging.getLogger(__name__)

BAT = os.getenv("TH_BUOC_CHAN", "1") != "0"
# Cửa sổ bước sau một mép ARDY → nền (giây). Mép trong câu dùng T của tầng chuyển tiếp (kẹp dưới
# CUA_SO_MIN); mép đúng ranh câu không có T đó nên dùng CUA_SO.
CUA_SO = float(os.getenv("TH_BUOC_CUA_SO", "1.2"))
CUA_SO_MIN = 0.6

CHAN = ("LeftFoot", "RightFoot")
_CHUOI = {"LeftFoot": ("LeftUpLeg", "LeftLeg", "LeftFoot"),
          "RightFoot": ("RightUpLeg", "RightLeg", "RightFoot")}
_TEN = ("Hips",) + tuple(x for c in _CHUOI.values() for x in c)
_KHAC = {"LeftFoot": "RightFoot", "RightFoot": "LeftFoot"}
# phần đầu/cuối của một bước chỉ đi theo phương ĐỨNG (nhấc lên / đặt xuống), xem `quy_dao`
NHAC_TRUOC = 0.12


@dataclass
class Buoc:
    """Một bước của một chân. Thời gian trên trục của clip (trừ `t_lech` khi áp)."""

    chan: str                       # "LeftFoot" | "RightFoot"
    t_bd: float                     # giây — lúc nhấc chân
    dai: float                      # giây — từ nhấc tới đặt
    toi: tuple                      # (x, y, z) m, world — chỗ đặt CỔ CHÂN
    huong: tuple | None = None      # quaternion world (w,x,y,z) bàn chân lúc đặt; None = giữ hướng
    cao: float = 0.06               # m — độ nhấc ở giữa bước

    @property
    def t_kt(self) -> float:
        return self.t_bd + self.dai


@dataclass
class ThamSo:
    """Tham số lên kế hoạch của `buoc_chan`. Mặc định cho bước CHỈNH TƯ THẾ ĐỨNG (không phải đi)."""

    nguong_cm: float = 3.0          # lệch ngang dưới mức này và xoay dưới `nguong_do` thì chỉ đứng yên
    nguong_do: float = 8.0
    nguong_nhac_cm: float = 2.0     # lúc t0 bàn chân cao hơn sàn mức này thì phải đặt xuống
    dai_goc: float = 0.30           # thời lượng bước = dai_goc + dai_he·quãng(m), kẹp [min, max]
    dai_he: float = 0.90
    dai_min: float = 0.35
    dai_max: float = 0.60
    cao_he: float = 0.25            # độ nhấc = cao_he·quãng(m), kẹp [min, max]
    cao_min: float = 0.04
    cao_max: float = 0.10
    tre: float = 0.08               # từ t0 tới lúc nhấc chân đầu tiên
    nghi: float = 0.04              # chân sau nhấc khi chân trước đã chạm đất được chừng này
    tat: float = 0.15               # tắt dần sau bước cuối
    don_hong: float = 0.015         # m — hông dồn sang chân trụ trong lúc chân kia nhấc
    # bước ĐUỔI THEO trong lúc nguồn còn đang đổi tư thế (xem `ke_hoach`)
    nguong_buoc_cm: float = 10.0    # chân trụ lệch chỗ nguồn muốn quá mức này thì nhấc
    nguong_buoc_do: float = 25.0
    voi_toi: float = 0.97           # chân trụ cách khớp háng quá tỉ lệ này × chiều dài chân → nhấc
    nhin_truoc: float = 0.15        # đặt chân xuống chỗ nguồn muốn SAU lúc chạm đất chừng này
    max_buoc: int = 8


# ---- toán nhỏ ----------------------------------------------------------------------------
def _minjerk(u: float) -> float:
    """0 → 1, vận tốc và gia tốc bằng 0 ở hai đầu."""
    u = min(1.0, max(0.0, u))
    return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


def _chuong(u: float) -> float:
    """0 → 1 → 0, đỉnh ở giữa, đạo hàm 0 ở hai đầu."""
    u = min(1.0, max(0.0, u))
    return 16.0 * u * u * (1.0 - u) * (1.0 - u)


def _slerp(a, b, t: float) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = float(a @ b)
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(min(1.0, d))
    s = math.sin(th)
    return (math.sin((1.0 - t) * th) * a + math.sin(t * th) * b) / s


def _goc_do(a, b) -> float:
    return math.degrees(2.0 * math.acos(min(1.0, abs(float(np.asarray(a, float) @ np.asarray(b, float))))))


def _rig():
    from app.modules.motion.hinh_hoc import _rig as r
    return r()


def _lo():
    from app.modules.motion.bo_phan.chan import (_euler_xyz_sang_q_lo, _muot_hieu_chinh,
                                                 _q_mul_lo, _q_xoay_v_lo)
    return _euler_xyz_sang_q_lo, _q_mul_lo, _q_xoay_v_lo, _muot_hieu_chinh


def _luoi(clip: dict):
    """(bones, mốc) nếu mọi track chuỗi chân cùng một lưới, không thì None — lệch lưới thì thà
    không bước còn hơn bước sai chỗ (cùng luật với `ghim_chan_cuoi`)."""
    b = clip.get("bones") or {}
    if any(x not in b for x in _TEN):
        return None
    T = len(b["Hips"])
    if T < 2 or any(len(b[x]) != T for x in _TEN):
        return None
    moc = np.array([f[0] for f in b["Hips"]], float)
    for x in _TEN:
        if abs(b[x][0][0] - moc[0]) > 1e-6 or abs(b[x][-1][0] - moc[-1]) > 1e-6:
            return None
    return b, moc


def _fk(clip: dict, b: dict, moc, ks) -> dict:
    """FK chuỗi Hips → đùi → gối → cổ chân theo LÔ cho các khung `ks` (như `ghim_chan_cuoi`)."""
    B, n2i, par, rr, rp, _ = _rig()
    eq, qmul, qrot, _m = _lo()
    q = {x: eq(np.asarray([b[x][k][1:4] for k in ks], float)) for x in _TEN}
    hp = np.asarray(clip.get("hips_pos") or [[0.0, 0.0, 0.0, 0.0]], float)
    dx = np.stack([np.interp(moc[ks], hp[:, 0], hp[:, 1 + j]) for j in range(3)], axis=1)
    qw_h = q["Hips"]
    p_h = np.asarray(rp[n2i["Hips"]], float) + dx
    ra = {"q": q, "qw_h": qw_h, "p_h": p_h, "truc": qrot(qw_h, np.array([1.0, 0.0, 0.0]))}
    for c, (hip, goi, co) in _CHUOI.items():
        qw_u = qmul(qw_h, q[hip])
        p_u = p_h + qrot(qw_h, np.asarray(rp[n2i[hip]], float))
        qw_l = qmul(qw_u, q[goi])
        p_l = p_u + qrot(qw_u, np.asarray(rp[n2i[goi]], float))
        ra[c] = {"qw_u": qw_u, "p_u": p_u, "qw_l": qw_l, "p_l": p_l,
                 "p_f": p_l + qrot(qw_l, np.asarray(rp[n2i[co]], float)),
                 "qw_f": qmul(qw_l, q[co])}
    return ra


# ---- cửa công khai ------------------------------------------------------------------------
def vi_tri_chan(clip: dict, t: float) -> dict | None:
    """{chân: (vị trí cổ chân world (3,), hướng bàn chân world (w,x,y,z))} tại mốc `t` của clip."""
    lu = _luoi(clip)
    if lu is None:
        return None
    b, moc = lu
    k = int(np.clip(np.searchsorted(moc, t - 1e-6), 0, len(moc) - 1))
    fk = _fk(clip, b, moc, np.array([k]))
    return {c: (fk[c]["p_f"][0].copy(), fk[c]["qw_f"][0].copy()) for c in CHAN}


def chan_nghi(goc=None, base_pose: dict | None = None, tu_the: dict | None = None) -> dict:
    """{chân: (vị trí, hướng)} world của hai bàn chân ở THẾ ĐỨNG NGHỈ, đặt tại `goc`.

    Tư thế: nghỉ của rig, đè `base_pose`, rồi đè `tu_the` (hông + chân ở khung đầu clip IDLE —
    `_the_dung_idle`). Hông ở gốc (hips_pos 0), rồi đặt vào `goc` = (dx, dz, yaw) bằng ĐÚNG phép
    của gốc trôi: hông dời (dx, dz), cả chuỗi xoay `yaw` quanh trục đứng qua hông.

    Vì sao cần (đo 10/09): hết lượt, client chuyển sang clip idle, mà MỌI clip idle RPM mở màn từ
    cùng một thế đứng — hai bàn chân rộng 25,1 cm, trái lùi 6,8 cm, phải tiến 7,9 cm so với tư
    thế nghỉ rig (30,9 cm). Chân nền đứng ở chỗ khác (thế của transformer ≈ tư thế nghỉ rig) thì
    lúc idle chạy hai bàn chân trượt về đúng khoảng đó, đỉnh 16 cm. Bản đầu dùng tư thế nghỉ rig +
    `base_pose` làm đích: trình duyệt đo ra y hệt trước khi sửa — đích sai chỗ.
    """
    B, n2i, par, rr, rp, _ = _rig()
    rot = dict(rr)
    for nguon in (base_pose or {}, tu_the or {}):
        for ten, e in nguon.items():
            if ten in n2i and e is not None and len(e) >= 3:
                rot[n2i[ten]] = B.q_from_euler_xyz(float(e[0]), float(e[1]), float(e[2]))
    dx, dz, yaw = (0.0, 0.0, 0.0) if goc is None else (float(goc[0]), float(goc[1]), float(goc[2]))
    c, s = math.cos(yaw), math.sin(yaw)
    qy = np.array([math.cos(yaw / 2.0), 0.0, math.sin(yaw / 2.0), 0.0])
    h = np.asarray(rp[n2i["Hips"]], float)
    ra = {}
    for ch in CHAN:
        q, p = B.world_transform(n2i[ch], par, rot, rp, {})
        v = np.asarray(p, float) - h
        ra[ch] = (np.array([h[0] + dx + c * v[0] + s * v[2], h[1] + v[1],
                            h[2] + dz - s * v[0] + c * v[2]]),
                  np.asarray(B.q_mul(qy, np.asarray(q, float)), float))
    return ra


@lru_cache(maxsize=8)
def _the_dung_idle(nhan_vat: str, ca_nguoi: bool = False) -> tuple:
    """((xương, (ex, ey, ez)), …) — tư thế HÔNG + CHÂN ở khung đầu clip idle của nhân vật.

    Mọi clip idle RPM mở màn từ đúng tư thế này (CLAUDE.md §9; đo 10/09: IdleRpm01/02/03 trùng
    tới 0,1 mm, và khung đầu của tệp thô trùng khung đầu clip `/api/clip_ong` gửi client tới
    0,0000 rad) — tức là chỗ nhân vật SẼ đứng khi hết lượt. Rỗng nếu nhân vật không có clip đó.
    """
    # IdleDung* (10/09, `backend/luong_clip/b1_render/idle_dung.py`) = bộ idle client tự phát, thân
    # dưới ĐÓNG BĂNG đúng ở thế này; IdleRpm01 là nguồn của nó (khung đầu trùng tới 0,0°) — dự phòng.
    kho = Path(__file__).resolve().parents[2] / "backend" / "app" / "motion_clips" / nhan_vat
    for ten in os.getenv("TH_THE_DUNG_IDLE", "IdleDung01,IdleRpm01").split(","):
        p = kho / f"{ten.strip()}.json"
        if not p.exists():
            continue
        try:
            b = json.loads(p.read_text(encoding="utf-8")).get("bones") or {}
        except Exception:  # noqa: BLE001
            logger.warning("không đọc được thế đứng idle %s", p, exc_info=True)
            continue
        # `ca_nguoi`: MỌI xương (đích của đuôi về nghỉ — `than._them_duoi`), không chỉ hông + chân
        return tuple((x, tuple(float(v) for v in b[x][0][1:4]))
                     for x in (tuple(b) if ca_nguoi else _TEN) if b.get(x))
    return ()


def ke_hoach(clip: dict, t0: float, t1: float, dich: dict | None = None,
             ts: ThamSo | None = None, giu: dict | None = None, t_het: float | None = None,
             chot: bool = True) -> list[Buoc]:
    """Lên kế hoạch bước — BỘ ĐIỀU KHIỂN BƯỚC chạy mô phỏng theo từng khung trong [t0, t1].

    Hai loại bước:
      · ĐUỔI THEO, trong lúc nguồn còn đang đổi tư thế: chân trụ lệch chỗ nguồn muốn quá
        `nguong_buoc_cm`/`nguong_buoc_do`, hoặc hông đã bỏ nó lại tới mức chân sắp duỗi hết
        (`voi_toi`), hoặc bàn chân đang lơ lửng → nhấc chân đó và đặt xuống chỗ nguồn muốn lúc
        chạm đất. Cú xoay người lớn (đo 10/09: 148° sau 'pivot_heel_think') vì thế thành vài bước
        xoay tại chỗ. Bản đầu chỉ có bước chốt: đúng mép đó còn trượt 33 cm (không làm gì: 28,9).
      · CHỐT sau cùng: chân nào còn lệch chỗ đứng cuối quá `nguong_cm`/`nguong_do` thì bước nốt.
    Chỉ một chân nhấc một lúc; chân sau nhấc khi chân trước đã chạm đất `nghi` giây.

    `dich` = {chân: (vị trí, hướng | None)} để chốt ở chỗ KHÁC chỗ nguồn muốn.
    `giu`  = {chân: (vị trí, hướng)} chỗ đứng lúc t0 khi nó KHÔNG phải tư thế của clip tại t0.
    `t_het`= HẠN CỨNG: không bước nào (kể cả phần tắt dần) được kéo qua mốc này — đoạn kế tiếp
             bắt đầu ở đó và chân là của nó (đo 10/09: bước đè lên đầu clip ARDY toàn thân,
             chân lệch kế hoạch 3,1 cm và hỏng nội dung động tác).
    `chot` = False thì bỏ bước CHỐT — dùng khi cửa sổ tràn qua cuối clip: phần còn lại để clip
             sau lên kế hoạch tiếp, vì nguồn của clip sau thì clip này chưa nhìn thấy.
    """
    ts = ts or ThamSo()
    lu = _luoi(clip)
    if lu is None:
        return []
    han = float("inf") if t_het is None else float(t_het)
    t1 = min(float(t1), han)
    b, moc = lu
    k0 = int(np.clip(np.searchsorted(moc, t0 - 1e-6), 0, len(moc) - 1))
    ks = np.arange(k0, len(moc))
    fk = _fk(clip, b, moc, ks)
    mk = moc[ks]
    fps = 1.0 / float(np.median(np.diff(moc))) if len(moc) > 1 else 60.0
    B, n2i, par, rr, rp, _ = _rig()
    dai_chan = {c: float(np.linalg.norm(rp[n2i[_CHUOI[c][1]]]) + np.linalg.norm(rp[n2i[_CHUOI[c][2]]]))
                for c in CHAN}
    from loi.neo_san import san_rig
    san = san_rig()

    def chi_so(t: float) -> int:
        return int(np.clip(np.searchsorted(mk, t - 1e-6), 0, len(mk) - 1))

    def nguon(c: str, t: float):
        i = chi_so(t)
        return fk[c]["p_f"][i], fk[c]["qw_f"][i], fk[c]["p_u"][i]

    def noi_dat(c: str, t_cham: float):
        """Chỗ nguồn muốn đặt chân khi chạm đất lúc `t_cham`: nhìn tiếp tới lúc bàn chân của nguồn
        LẮNG (≤0,5 s). Đặt vào chỗ nguồn đang đi QUA là phải bước thêm lần nữa — đo: nguồn lướt
        22 cm thành 2 bước thay vì 1."""
        P, Q = fk[c]["p_f"], fk[c]["qw_f"]
        i, j = chi_so(min(t_cham + ts.nhin_truoc, t1)), chi_so(min(t_cham + 0.5, t1))
        for k in range(i, j):
            if math.hypot(P[k + 1][0] - P[k][0], P[k + 1][2] - P[k][2]) * fps < 0.05:
                return P[k], Q[k]
        return P[j], Q[j]

    def lech(p, q, p2, q2):
        return float(math.hypot(p2[0] - p[0], p2[2] - p[2])), _goc_do(q, q2)

    def dai_cao(d: float):
        return (min(ts.dai_max, max(ts.dai_min, ts.dai_goc + ts.dai_he * d)),
                min(ts.cao_max, max(ts.cao_min, ts.cao_he * d)))

    if giu:
        dung = {c: (np.asarray(giu[c][0], float).copy(), np.asarray(giu[c][1], float).copy())
                for c in CHAN}
    else:
        dung = {c: (fk[c]["p_f"][0].copy(), fk[c]["qw_f"][0].copy()) for c in CHAN}
    ra: list[Buoc] = []
    da_buoc: set = set()

    def them(c: str, t: float, toi, huong) -> float:
        d = lech(dung[c][0], dung[c][1], toi, huong)[0]
        dai, cao = dai_cao(d)
        ra.append(Buoc(c, t, dai, tuple(float(v) for v in toi), tuple(float(v) for v in huong), cao))
        da_buoc.add(c)
        dung[c] = (np.asarray(toi, float).copy(), np.asarray(huong, float).copy())
        return t + dai + ts.nghi

    # 1) ĐUỔI THEO
    ranh = t0 + ts.tre
    for t in mk:
        t = float(t)
        if t > t1 or len(ra) >= ts.max_buoc:
            break
        if t < ranh:
            continue
        tot, chon = 0.0, None
        for c in CHAN:
            p, q = dung[c]
            pn, qn, _pu = nguon(c, min(t + ts.nhin_truoc, t1))
            d, g = lech(p, q, pn, qn)
            # VỚI TỚI = chân trụ phải DUỖI XA HƠN chân của nguồn (hông đã bỏ nó lại phía sau), và
            # đã gần hết tầm. Chỉ xét tầm tuyệt đối là SAI: người đứng thẳng vốn duỗi gần hết
            # chân (gối ~4°), nên clip đứng yên cũng bị bắt bước liên tục (đo: 4 bước khi đứng im).
            pu = nguon(c, t)[2]
            xa = float(np.linalg.norm(p - pu))
            voi = (xa > ts.voi_toi * dai_chan[c]
                   and xa - float(np.linalg.norm(nguon(c, t)[0] - pu)) > 0.04)
            # LƠ LỬNG chỉ xét chân CHƯA bước lần nào (ARDY kết lúc chân còn trên không). Chân vừa
            # đặt xuống thì đứng đúng chỗ nguồn muốn — xét lại là bắt bước thừa tại chỗ.
            lo_lung = c not in da_buoc and (float(p[1]) - san) * 100.0 > ts.nguong_nhac_cm
            # chân CAO HƠN đặt xuống TRƯỚC (14/09): cắt giữa cú chạy, hai chân cùng lơ lửng mà điểm
            # bằng nhau thì chân trái (thứ tự CHAN) bước trước — chân phải cao 49 cm treo 0,57 s, gối 26°
            diem = max(d * 100.0 / ts.nguong_buoc_cm, g / ts.nguong_buoc_do,
                       2.0 if voi else 0.0, 3.0 + max(0.0, float(p[1]) - san) if lo_lung else 0.0)
            if diem >= 1.0 and diem > tot:
                tot, chon = diem, c
        if chon is None:
            continue
        dai, _cao = dai_cao(lech(*dung[chon], *nguon(chon, min(t + 0.5, t1))[:2])[0])
        if t + dai + ts.tat > han:
            continue
        ranh = them(chon, t, *noi_dat(chon, t + dai))

    # 2) CHỐT — về đúng chỗ đứng cuối (chỗ nguồn muốn lúc t1, hoặc `dich`)
    if not chot:
        return ra
    cuoi = {}
    for c in CHAN:
        if dich and c in dich:
            p1 = np.asarray(dich[c][0], float)
            q1 = (np.asarray(dich[c][1], float) if len(dich[c]) > 1 and dich[c][1] is not None
                  else dung[c][1])
        else:
            p1, q1, _pu = nguon(c, t1)
        cuoi[c] = (p1, q1)
    can = []
    for c in CHAN:
        d, g = lech(*dung[c], *cuoi[c])
        if d * 100.0 > ts.nguong_cm or g > ts.nguong_do:
            can.append((d, c))
    for d, c in sorted(can, key=lambda x: -x[0]):
        if len(ra) >= ts.max_buoc or ranh + dai_cao(d)[0] + ts.tat > han:
            break
        ranh = them(c, ranh, *cuoi[c])
    return ra


def di_chuyen_chan(clip: dict, cac_buoc: list[Buoc], t_bd: float | None = None,
                   giu: dict | None = None, giu_den: float | None = None, t_lech: float = 0.0,
                   tat: float = 0.15, don_hong: float = 0.015) -> dict:
    """Cho chân bước đúng theo `cac_buoc`. Sửa `clip` tại chỗ; trả báo cáo.

    Cửa sổ điều khiển = [t_bd, giu_den] rồi tắt dần `tat` giây:
      · `t_bd`    từ lúc này hai bàn chân ĐỨNG YÊN tại chỗ (mặc định = lúc nhấc bước đầu)
      · `giu`     {chân: (vị trí, hướng)} chỗ đứng lúc bắt đầu; mặc định đọc từ clip tại `t_bd`.
                  Truyền vào khi cửa sổ bắt đầu ở CLIP TRƯỚC (bước kéo qua ranh câu)
      · `giu_den` giữ chỗ đặt chân tới lúc này rồi mới tắt (mặc định = lúc đặt bước cuối).
                  Bước tới chỗ nguồn KHÔNG muốn thì phải giữ tới hết clip, không thì tắt dần
                  là bàn chân trượt về chỗ nguồn
      · `t_lech`  mốc của clip trên trục thời gian mà `cac_buoc` dùng (0 = cùng trục)
    """
    # danh sách bước RỖNG + `t_bd` + `giu_den` = chỉ GIỮ hai bàn chân đứng yên trong [t_bd, giu_den]
    if not BAT or (not cac_buoc and (t_bd is None or giu_den is None)):
        return {}
    lu = _luoi(clip)
    if lu is None:
        return {"bo_qua": "lưới chân không thẳng hàng"}
    b, moc = lu
    buoc = sorted(cac_buoc, key=lambda s: s.t_bd)
    t_a = (t_bd if t_bd is not None else buoc[0].t_bd) - t_lech
    t_z = (giu_den if giu_den is not None else max(s.t_kt for s in buoc)) - t_lech
    ks = np.nonzero((moc >= t_a - 1e-6) & (moc <= t_z + tat + 1e-6))[0]
    if len(ks) < 2:
        return {"bo_qua": "cửa sổ nằm ngoài clip"}

    if giu is None:
        k0 = int(ks[0])
        fk0 = _fk(clip, b, moc, np.array([k0]))
        giu = {c: (fk0[c]["p_f"][0].copy(), fk0[c]["qw_f"][0].copy()) for c in CHAN}
    giu = {c: (np.asarray(p, float), np.asarray(q, float)) for c, (p, q) in giu.items()}
    theo_chan = {c: [s for s in buoc if s.chan == c] for c in CHAN}

    def quy_dao(c: str, t: float):
        """(vị trí, hướng, đang_nhấc) của chân `c` theo kế hoạch tại mốc clip `t`."""
        p, q = giu[c]
        for s in theo_chan[c]:
            s_bd, s_kt = s.t_bd - t_lech, s.t_kt - t_lech
            toi = np.asarray(s.toi, float)
            hq = np.asarray(s.huong, float) if s.huong is not None else q
            if t < s_bd:
                return p, q, False
            if t <= s_kt:
                u = (t - s_bd) / max(s.dai, 1e-6)
                # NHẤC TRƯỚC, ĐI SAU, ĐẶT THẲNG: người thật nhấc chân lên rồi mới đưa ngang, và hạ
                # xuống gần như thẳng đứng — bàn chân vừa rời/vừa chạm sàn thì KHÔNG lướt.
                m = _minjerk((u - NHAC_TRUOC) / (1.0 - 2.0 * NHAC_TRUOC))
                pos = p + m * (toi - p)
                pos[1] = p[1] + _minjerk(u) * (toi[1] - p[1]) + s.cao * _chuong(u)
                return pos, _slerp(q, hq, m), True
            p, q = toi, hq
        return p, q, False

    def trong_so(t: float) -> float:
        if t <= t_z:
            return 1.0
        s = (t - t_z) / max(tat, 1e-6)
        return 0.0 if s >= 1.0 else 1.0 - _minjerk(s)

    # 1) HÔNG DỒN SANG CHÂN TRỤ — ghi hips_pos TRƯỚC khi giải IK (IK dùng hông thật từng khung).
    #    Bắt đầu dồn sớm 0,1 s: người thật chuyển trọng tâm TRƯỚC khi nhấc chân.
    hp = clip.get("hips_pos") or []
    if don_hong > 0 and hp:
        def lech_hong(t: float, x: float, z: float):
            ox = oz = 0.0
            for s in buoc:
                a0, a1 = s.t_bd - t_lech - 0.10, s.t_kt - t_lech
                if not (a0 <= t <= a1):
                    continue
                tru, _q, _ = quy_dao(_KHAC[s.chan], t)
                vx, vz = float(tru[0]) - x, float(tru[2]) - z
                n = math.hypot(vx, vz)
                if n < 1e-6:
                    continue
                k = don_hong * math.sin(math.pi * (t - a0) / (a1 - a0)) ** 2 / n
                ox += vx * k
                oz += vz * k
            return ox, oz
        p_h0 = np.asarray(_rig()[4][_rig()[1]["Hips"]], float)
        for f in hp:
            if t_a - 1e-6 <= f[0] <= t_z + 1e-6:
                ox, oz = lech_hong(float(f[0]), p_h0[0] + float(f[1]), p_h0[2] + float(f[3]))
                f[1] = float(f[1]) + ox
                f[3] = float(f[3]) + oz

    # 2–4) IK từng khung + tinh chỉnh + tự kiểm — lõi dùng chung với `khoa_truot`
    W = np.array([trong_so(float(moc[k])) for k in ks])
    sai = _giai_ik(clip, b, moc, ks, W, lambda c, t: quy_dao(c, t)[:2])
    bc = {"buoc": [(s.chan, round(s.t_bd, 2), round(s.dai, 2), round(s.cao * 100, 1)) for s in buoc],
          "khung": int(len(ks)), "t": (round(t_a, 2), round(t_z + tat, 2)),
          "sai_cm": round(sai * 100, 2)}
    clip.setdefault("_buoc_chan", []).append(bc)
    logger.info("bước chân: %s · %d khung [%.2f–%.2f]s · lệch kế hoạch ≤ %.2f cm",
                " · ".join(f"{c[:-4]}@{t}s {d}s cao {h}cm" for c, t, d, h in bc["buoc"]),
                bc["khung"], bc["t"][0], bc["t"][1], bc["sai_cm"])
    return bc


def _giai_ik(clip: dict, b: dict, moc, ks, W, quy_dao) -> float:
    """LÕI IK chân dùng chung (`di_chuyen_chan`, `khoa_truot`): kéo cổ chân từng khung `ks` về `quy_dao(chân, t)` =
    (vị trí world, hướng world) theo trọng số `W` (mảng chung hoặc {chân: mảng}). Sửa `clip` tại chỗ; trả sai lệch lớn
    nhất (m) ở những khung trọng số 1.

    Ba lượt, giữ đúng thứ tự đã hiệu chuẩn: (1) IK có LÀM MƯỢT trường hiệu chỉnh ±4 khung (gối khỏi nhảy nhánh khi chân gần
    duỗi — CLAUDE.md §2); (2) ghi Euler nhánh liên tục (§4 bẫy 0/1); (3) TINH CHỈNH một lượt không mượt xuất phát từ nghiệm
    đã mượt — trung bình ±4 khung trễ theo nguồn đổi nhanh làm chân trụ trôi (đo 10/09: 1,04 → 0,11 cm); rồi tự kiểm FK.
    """
    B = _rig()[0]
    _eq, _qm, _qr, muot = _lo()
    Wc = (lambda c: W[c]) if isinstance(W, dict) else (lambda c: W)
    fk = _fk(clip, b, moc, ks)
    moi: dict = {}
    for c, (hip, goi, co) in _CHUOI.items():
        F, Wx = fk[c], Wc(c)
        u_m, l_m, f_m = fk["q"][hip].copy(), fk["q"][goi].copy(), fk["q"][co].copy()
        for i, k in enumerate(ks):
            w = float(Wx[i])
            if w <= 1e-4:
                continue
            pos, qq = quy_dao(c, float(moc[k]))
            dich = F["p_f"][i] + w * (np.asarray(pos, float) - F["p_f"][i])
            huong = _slerp(F["qw_f"][i], qq, w)
            kq = B.ik_hai_xuong(F["p_u"][i], F["p_l"][i], F["p_f"][i], dich, truc_on=fk["truc"][i])
            if kq is None:
                continue
            q1 = np.asarray(kq[0], float)
            q2 = np.asarray(kq[1], float)
            qw_u2 = np.asarray(B.q_mul(q1, F["qw_u"][i]), float)
            qw_l2 = np.asarray(B.q_mul(q2, B.q_mul(qw_u2, fk["q"][goi][i])), float)
            u_m[i] = B.q_mul(B.q_inv(fk["qw_h"][i]), qw_u2)
            l_m[i] = B.q_mul(B.q_inv(qw_u2), qw_l2)
            f_m[i] = B.q_mul(B.q_inv(qw_l2), huong)
        for ten, qn in ((hip, u_m), (goi, l_m), (co, f_m)):
            moi[ten] = muot(fk["q"][ten], qn)
    for ten, qn in moi.items():
        tr = b[ten]
        gan = tuple(tr[int(ks[0]) - 1][1:4]) if ks[0] > 0 else tuple(tr[int(ks[0])][1:4])
        for i, k in enumerate(ks):
            e = _q_sang_euler(tuple(float(v) for v in qn[i]), gan)
            tr[int(k)][1], tr[int(k)][2], tr[int(k)][3] = e
            gan = e
    fk_m = _fk(clip, b, moc, ks)
    for c, (hip, goi, co) in _CHUOI.items():
        F0, F2, Wx = fk[c], fk_m[c], Wc(c)
        for i, k in enumerate(ks):
            w = float(Wx[i])
            if w <= 1e-4:
                continue
            pos, qq = quy_dao(c, float(moc[k]))
            dich = F0["p_f"][i] + w * (np.asarray(pos, float) - F0["p_f"][i])
            huong = _slerp(F0["qw_f"][i], qq, w)
            kq = B.ik_hai_xuong(F2["p_u"][i], F2["p_l"][i], F2["p_f"][i], dich, truc_on=fk_m["truc"][i])
            if kq is None:
                continue
            qw_u2 = np.asarray(B.q_mul(np.asarray(kq[0], float), F2["qw_u"][i]), float)
            qw_l2 = np.asarray(B.q_mul(np.asarray(kq[1], float), B.q_mul(qw_u2, fk_m["q"][goi][i])), float)
            for ten, qn in ((hip, B.q_mul(B.q_inv(fk_m["qw_h"][i]), qw_u2)),
                            (goi, B.q_mul(B.q_inv(qw_u2), qw_l2)),
                            (co, B.q_mul(B.q_inv(qw_l2), huong))):
                tr = b[ten]
                gan = tuple(tr[int(k) - 1][1:4]) if k > 0 else tuple(tr[int(k)][1:4])
                tr[int(k)][1], tr[int(k)][2], tr[int(k)][3] = _q_sang_euler(tuple(float(v) for v in qn), gan)
    fk2 = _fk(clip, b, moc, ks)
    sai = 0.0
    for c in CHAN:
        Wx = Wc(c)
        for i, k in enumerate(ks):
            if Wx[i] >= 1.0 - 1e-6:
                pos, _q = quy_dao(c, float(moc[k]))
                sai = max(sai, float(np.linalg.norm(fk2[c]["p_f"][i] - np.asarray(pos, float))))
    return sai


# KHOÁ TRƯỢT CHÂN TRONG CLIP TOÀN THÂN (15/09, user: *"chân bị trôi lúc chạy clip motion, rõ ràng clip có vấn đề"*).
# Clip toàn thân (xoay gót, dậm, quay người tại chỗ, đi/chạy) để CHÂN LÀ NỘI DUNG (user chốt 12/09 "clip thì chân tự do")
# nên foot-skating vốn có của ARDY hiện nguyên: đo kho, `pivot_heel_think` bàn chân thấp ≤3 cm ở 85–88 % khung mà vẫn trôi
# trung vị 4–5 cm/s (93 cm cả clip), `turns_around_slowly_in_place` 9–11 cm/s; clip ĐI BỘ thì khung chân thấp phần lớn là
# lúc bước nhanh 60–190 cm/s. KHÔNG phải lô làm tròn 15/09 (160 clip trước/sau: trượt TB 129,8 → 127,9 cm). Nghiệm KHÔNG
# theo sổ khoảng contacts (§2: khoảng gộp phủ cả cú vung) mà THEO DỮ LIỆU TỪNG KHUNG: pha chạm = thấp × chậm kéo dài ≥
# `KT_MIN_S` → ghim cổ chân tại ĐIỂM VỪA ĐẶT CHÂN, hướng bàn chân vẫn theo clip (xoay gót còn nguyên), mép vào/ra trộn
# mềm `KT_MEP_S`; pha nhấc chân để nguyên nội dung. Neo không bao giờ xa chân clip hơn `KT_LECH_MAX` (kéo theo, liên tục) —
# hông ARDY dời xa mà ghim cứng là duỗi gãy chân. Tắt: TH_KHOA_TRUOT=0.
KHOA_TRUOT = os.getenv("TH_KHOA_TRUOT", "1") != "0"
KT_CAO = float(os.getenv("TH_KT_CAO", "0.03"))        # m trên đáy bàn chân trong đoạn
KT_V = float(os.getenv("TH_KT_V", "0.25"))            # m/s ngang
KT_MIN_S = 0.10
KT_MEP_S = 0.12
KT_LECH_MAX = 0.10


def khoa_truot(clip: dict, t_a: float, t_b: float) -> dict:
    """Ghim pha chạm sàn của hai bàn chân trong [t_a, t_b] (đoạn chân là nội dung). Sửa `clip` tại chỗ; trả báo cáo."""
    if not (BAT and KHOA_TRUOT):
        return {}
    lu = _luoi(clip)
    if lu is None:
        return {"bo_qua": "lưới chân không thẳng hàng"}
    b, moc = lu
    ks = np.nonzero((moc >= t_a - 1e-6) & (moc <= t_b + 1e-6))[0]
    if len(ks) < 8:
        return {}
    tk = moc[ks]
    fk = _fk(clip, b, moc, ks)
    W: dict = {}
    NEO: dict = {}
    so_pha = 0
    for c in CHAN:
        P = np.asarray(fk[c]["p_f"], float)
        n = len(P)
        v = np.zeros(n)
        v[1:] = np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1) / np.maximum(np.diff(tk), 1e-6)
        v[0] = v[1] if n > 1 else 0.0
        dung = (P[:, 1] - P[:, 1].min() <= KT_CAO) & (v <= KT_V)
        w = np.zeros(n)
        neo = P.copy()
        i = 0
        while i < n:
            if not dung[i]:
                i += 1
                continue
            j = i
            while j < n and dung[j]:
                j += 1
            if tk[j - 1] - tk[i] >= KT_MIN_S:
                so_pha += 1
                a = P[i].copy()
                for k in range(i, j):
                    d = P[k] - a
                    dn = float(np.hypot(d[0], d[2]))
                    if dn > KT_LECH_MAX:                      # chân clip đi quá xa neo: neo KÉO THEO, liên tục
                        a = a + d * (1.0 - KT_LECH_MAX / dn)
                    neo[k] = a
                    w[k] = 1.0
            i = j
        # mép mềm: khung ngoài pha chạm trong KT_MEP_S tính từ khung chạm gần nhất → trọng số smoothstep về neo đó
        chot = np.nonzero(w >= 1.0)[0]
        if len(chot):
            for k in range(n):
                if w[k] >= 1.0:
                    continue
                g = chot[np.argmin(np.abs(tk[chot] - tk[k]))]
                s = 1.0 - abs(tk[k] - tk[g]) / KT_MEP_S
                if s > 0:
                    w[k] = _minjerk(float(s))
                    neo[k] = neo[g]
        W[c], NEO[c] = w, neo
    if not any(float(W[c].max(initial=0.0)) > 0 for c in CHAN):
        return {"pha": 0}
    idx = {round(float(t), 4): i for i, t in enumerate(tk)}
    fk_q = {c: np.asarray(fk[c]["qw_f"], float) for c in CHAN}

    def quy_dao(c: str, t: float):
        i = idx.get(round(t, 4))
        if i is None:
            i = int(np.argmin(np.abs(tk - t)))
        return NEO[c][i], fk_q[c][i]

    sai = _giai_ik(clip, b, moc, ks, W, quy_dao)
    bc = {"pha": so_pha, "khung": int(len(ks)), "t": (round(float(tk[0]), 2), round(float(tk[-1]), 2)),
          "phu": {c[:-4]: round(float((W[c] >= 1.0).mean()), 2) for c in CHAN}, "sai_cm": round(sai * 100, 2)}
    clip.setdefault("_khoa_truot", []).append(bc)
    logger.info("khoá trượt chân: %d pha chạm · %d khung [%.2f–%.2f]s · phủ %s · lệch ≤ %.2f cm",
                so_pha, bc["khung"], bc["t"][0], bc["t"][1], bc["phu"], bc["sai_cm"])
    return bc


def buoc_chan(clip: dict, t0: float, t1: float, dich: dict | None = None,
              ts: ThamSo | None = None, giu: dict | None = None, t_het: float | None = None,
              chot: bool = True, giu_den: float | None = None) -> dict:
    """Tự lên kế hoạch rồi bước: giữ hai bàn chân lúc `t0`, tới chỗ đứng lúc `t1` (hoặc `dich`).

    `giu`     chỗ đứng lúc t0 khi nó KHÔNG phải tư thế của clip tại t0 — mép nằm đúng ranh câu:
              clip này mở bằng nguồn mới, còn bàn chân thật đang ở chỗ câu trước để lại.
    `t_het`   hạn cứng, không bước nào kéo qua (xem `ke_hoach`).
    `chot`    False = bỏ bước chốt (cửa sổ tràn qua cuối clip, clip sau lên kế hoạch tiếp).
    `giu_den` giữ chỗ đặt chân tới mốc này rồi mới tắt dần. Có `giu_den` thì kể cả khi không
              cần bước nào, hai bàn chân vẫn được GIỮ đứng yên trong [t0, giu_den].

    Trả báo cáo kèm `ke_hoach` (list[Buoc]), `giu`, `t_bd`, `tat`: muốn đi nốt bước dở ở clip
    sau thì `di_chuyen_chan(clip_sau, ke_hoach, t_bd=t_bd, giu=giu, t_lech=độ dài clip này)`.
    """
    if not BAT:
        return {}
    ts = ts or ThamSo()
    if giu is None:
        giu = vi_tri_chan(clip, t0)
        if giu is None:
            return {}
    kh = ke_hoach(clip, t0, t1, dich, ts, giu=giu, t_het=t_het, chot=chot)
    if giu_den is None and dich is not None:
        giu_den = float(clip.get("duration_s") or t1)
    ra = {"buoc": [], "ke_hoach": kh, "giu": giu, "t_bd": t0, "tat": ts.tat}
    if not kh and giu_den is None:
        return ra
    bc = di_chuyen_chan(clip, kh, t_bd=t0, giu=giu, tat=ts.tat, don_hong=ts.don_hong,
                        giu_den=giu_den)
    return {**ra, **bc}



# TẮT CHÂN CỦA NỀN (user 10/09: "cần có switch bật tắt chân đúng lúc, nếu là transformer thì tắt
# chân"). Trong khung NỀN, bàn chân ĐỨNG YÊN — chân của transformer không điều khiển nữa, IK giải
# đùi/gối theo hông THẬT của nền (hông, thân, tay vẫn là transformer). ARDY vẫn điều khiển chân của
# nó; chỗ đổi thế đứng thì BƯỚC. Lưới an toàn: nền dời hông xa tới mức chân lệch >10 cm / sắp duỗi
# quá tầm thì bước một bước (bước ĐUỔI THEO của `ke_hoach`), không duỗi gãy chân.
# Đo trước khi có (10/09, trình duyệt): câu chỉ có transformer trượt 2,6–6,6 cm cả câu.
TAT_CHAN_NEN = os.getenv("TH_TAT_CHAN_NEN", "1") != "0"
# TẮT CẢ CHÂN CỦA CỬ CHỈ ARDY KHÔNG TOÀN THÂN (vẫy tay, nghiêng đầu, xoè tay…): đoạn đó cũng vào
# `_nen_doan` (`OngStream.chay`), bàn chân đứng ở thế nghỉ, IK theo hông của ARDY. Chân của các cử
# chỉ này là thế đứng riêng của ARDY (hai bàn chân rộng ~35 cm, thế idle 25 cm): mỗi cử chỉ phải
# bước vào rồi bước ra, và lượt kết khi chưa kịp bước ra thì hết lượt hai bàn chân trượt 13,5 cm về
# idle trong 0,25 s (đo trình duyệt 10/09). Clip TOÀN THÂN giữ chân của nó.
TAT_CHAN_ARDY = os.getenv("TH_TAT_CHAN_ARDY", "1") != "0"
# Việc chân của CẢ CÂU — mọi mối nối + giữ chân các đoạn nền/cử chỉ — ở `loi/moi_noi.py`
# (một cửa, tham số ghi trên đúng khung sẽ gửi, sổ mối nối). File này chỉ là hàm BƯỚC CHÂN.
