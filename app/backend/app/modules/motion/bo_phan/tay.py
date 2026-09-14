"""TAY — trái/phải là hai bộ phận riêng; đo ở bàn tay VÀ khuỷu.

Đo bàn tay thôi thì cú vung cẳng tay mà bàn tay che mất sẽ lọt. Trần theo thể
trạng như trước (toc_tran_tay cũ vẫn được tôn trọng).
"""
from __future__ import annotations

import math

from app.modules.motion.bo_phan.co_so import BoPhan, yeu_cau_gian, yeu_cau_loc
from app.modules.motion.bo_phan.do_bo_phan import toc_world
from app.modules.motion.hinh_hoc import _exp_q, _log_q, _rig

_MAC_DINH = {"toc_tran": 2.6, "gia_toc_tran": 32.0, "toc_san": 0.0,
             "khuyu_toc_tran": 1.6, "khuyu_gia_toc_tran": 16.0, "v_vot": 6.0,
             # BIÊN ĐỘ: co góc vai/cánh/cẳng tay về phía TÂM chuyển động của
             # chính clip (trung bình véc-tơ xoay) theo hệ số. 1.0 = giữ nguyên.
             # GRU vung tay cao hơn người thật nên có hệ số riêng cho nguồn GRU.
             "bien_do": 1.0, "bien_do_gru": 1.0,     # 0,8 → 1,0 (06/09, user "tăng thêm nhịp GRU")
             # QUÁN TÍNH CHUỖI (quan_tinh_chuoi). hz CAO NHẤT trong các bộ phận:
             # tay là đoạn NHẸ NHẤT nên bám nguồn sát nhất. 3,0 chọn bằng đo —
             # nhịp đặc trưng ra 1,12 so với người thật 1,10; hz 2,0 cho 1,02
             # (mềm quá), 4,0 cho 1,18 (cứng lại).
             "quan_tinh_hz": 3.0, "quan_tinh_giam_chan": 0.85}


def _tay(ben: str) -> BoPhan:
    return BoPhan(
        ten=f"tay_{ben}",
        xuong=tuple(f"{ben.capitalize() if ben in ('left','right') else ben}{x}" for x in
                    ("Shoulder", "Arm", "ForeArm", "Hand")),
        diem_do=(f"{ben.capitalize()}Hand",),
        mac_dinh=dict(_MAC_DINH))


tay_trai = _tay("Left")
tay_phai = _tay("Right")
# sửa tên cho đúng khoá cấu hình: profile.ong.tay_trai / tay_phai
tay_trai.ten, tay_phai.ten = "tay_trai", "tay_phai"
tay_trai.xuong = ("LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand")
tay_phai.xuong = ("RightShoulder", "RightArm", "RightForeArm", "RightHand")
tay_trai.diem_do, tay_phai.diem_do = ("LeftHand",), ("RightHand",)


def _hop_nhat_khoa_cu(t, t_chung):
    """toc_tran_tay / gia_toc_toi_da / toc_san_tay cũ -> tham số bộ phận.

    IDEMPOTENT (04/09): `BoPhan.tham_so` đã gọi hàm này; theo_doi/noi/tay_nghi
    gọi LẦN HAI trên dict đã quy đổi → hệ số áp kép = bình phương (elderly
    24·18/32 = 13,5 rồi ×18/32 = 7,59; heavy 16,5 → 11,3) — cổng dataset và
    log realtime chấm trần tay type non-adult ở NỬA trần thật, ống (lam_mem)
    thì đúng 13,5 → "vượt trần" hàng loạt mà ống không thấy gì để sửa. Đúng
    bẫy CLAUDE.md §2 "áp kép là bình phương hệ số". Đánh dấu để lần hai no-op.
    """
    if t.get("_khoa_cu_da_ap"):
        return t
    rieng = (t_chung.get("tay") or {})
    t.update({k: v for k, v in rieng.items()})           # ong.tay chung cho hai tay
    t["_khoa_cu_da_ap"] = True
    # KHOÁ CŨ = HỆ SỐ so với mặc định cũ (2,6 m/s · 32 m/s²), KHÔNG phải giá trị
    # tuyệt đối: trần thật nằm ở nguong_bo_phan.json (hiệu chuẩn theo người thật,
    # 2026-08-21: tay 24). teen 38/32 = 1,19x, elderly 26/32 = 0,81x vẫn giữ ý
    # "thể trạng khác nhau" mà không kéo trần về số đoán cũ.
    if "toc_tran_tay" in t_chung and "toc_tran" not in rieng:
        t["toc_tran"] = round(float(t.get("toc_tran") or 2.6) * float(t_chung["toc_tran_tay"]) / 2.6, 2)
    if "gia_toc_toi_da" in t_chung and "gia_toc_tran" not in rieng and float(t_chung["gia_toc_toi_da"] or 0) > 0:
        t["gia_toc_tran"] = round(float(t.get("gia_toc_tran") or 24.0) * float(t_chung["gia_toc_toi_da"]) / 32.0, 2)
    if "toc_san_tay" in t_chung and "toc_san" not in rieng:
        t["toc_san"] = t_chung["toc_san_tay"]
    return t


tay_trai.hop_nhat = tay_phai.hop_nhat = _hop_nhat_khoa_cu


def _tran(clip, t, bp, ctx):
    ban = toc_world(clip, bp.diem_do)
    khuyu = toc_world(clip, (bp.xuong[2],))          # ForeArm = khuỷu
    k = 1.0
    for gia, tr, can in ((ban["v95"], float(t.get("toc_tran") or 0), False),
                         (ban["a95"], float(t.get("gia_toc_tran") or 0), True),
                         (khuyu["v95"], float(t.get("khuyu_toc_tran") or 0), False),
                         (khuyu["a95"], float(t.get("khuyu_gia_toc_tran") or 0), True)):
        if tr > 0 and gia > tr:
            k = max(k, math.sqrt(gia / tr) if can else gia / tr)
    if k > 1.0:
        yeu_cau_gian(ctx, bp, f"{bp.ten} bàn {ban['v95']:.2f} m/s · khuỷu {khuyu['v95']:.2f}", k)
    # SÀN: gần bất động cả clip cũng là lỗi — nhưng chỉ xét khi CẢ HAI tay cùng
    # đơ, nên bộ phận chỉ ghi nhận, lam_mem quyết.
    ctx.setdefault("tay_v95", {})[bp.ten] = ban["v95"]
    ctx["tay_san"] = float(t.get("toc_san") or 0)
    vot = float(t.get("v_vot") or 0)
    if vot > 0 and ban["v_max"] > vot:
        yeu_cau_loc(ctx, bp, f"{bp.ten} vọt {ban['v_max']:.1f} m/s một khung")


def _bien_do(clip, t, bp, ctx):
    """Co BIÊN ĐỘ tay quanh tâm chuyển động của clip — trước khi đo trần.

    Tâm = trung bình véc-tơ xoay từng xương (vai, cánh, cẳng — KHÔNG bàn tay,
    giữ dáng bàn tay). Mỗi khung: v' = tâm + k·(v − tâm). Không dùng tư thế
    nghỉ của rig (T-pose) làm tâm: co về đó là tay dang ra chứ không hạ xuống.
    """
    if ctx.get("chi_do"):
        return
    t = _hop_nhat_khoa_cu(t, ctx["t_chung"])
    gru = str(ctx.get("nguon") or "").startswith("gru")
    k = float((t.get("bien_do_gru") if gru else t.get("bien_do")) or 1.0)
    if abs(k - 1.0) < 1e-3 or k <= 0:
        return
    B, *_ = _rig()
    b = clip.get("bones") or {}
    import numpy as np
    so = 0
    for ten in bp.xuong[:3]:
        tr = b.get(ten)
        if not tr or len(tr) < 2:
            continue
        R, prev = [], None
        for f in tr:
            q = B.q_from_euler_xyz(f[1], f[2], f[3])
            w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            if prev is not None and w * prev[0] + x * prev[1] + y * prev[2] + z * prev[3] < 0:
                w, x, y, z = -w, -x, -y, -z
            prev = (w, x, y, z)
            R.append(_log_q((w, x, y, z)))
        R = np.array(R)
        tam = R.mean(axis=0)
        for i, f in enumerate(tr):
            e = B.q_to_euler_xyz(np.array(_exp_q(tam + k * (R[i] - tam))))
            f[1], f[2], f[3] = float(e[0]), float(e[1]), float(e[2])
        so += 1
    if so:
        ctx.setdefault("ghi", []).append(f"biên độ {bp.ten} x{k:.2f}")


def _hinh_the(clip, t, bp, ctx):
    """Góc hình thể từ client (vai mở, khuỷu mở, tay ra trước) — chạy TRƯỚC biên độ/trần."""
    if ctx.get("chi_do"):
        return
    from app.modules.motion.bo_phan.hinh_the import ap_hinh_the
    ben = ('LeftArm', 'LeftForeArm') if bp.ten == 'tay_trai' else ('RightArm', 'RightForeArm')
    ap_hinh_the(clip, ctx, ben)


tay_trai.hook(_hinh_the)
tay_phai.hook(_hinh_the)
tay_trai.hook(_bien_do)
tay_phai.hook(_bien_do)
tay_trai.hook(_tran)
tay_phai.hook(_tran)

def _nhan_q(p, q):
    import numpy as np
    w1, x1, y1, z1 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)


def _lien_hop(q):
    import numpy as np
    return np.concatenate([q[:, :1], -q[:, 1:]], axis=1)


def _loc_bac_hai(x, moc, hz, zeta):
    """Lọc quán tính bậc hai, nghiệm GIẢI TÍCH, trên mảng (T, k) bất kỳ.

    e'' + 2*zeta*w*e' + w^2*e = 0 với e = y - x, coi x hằng trong mỗi khung:
    chính xác tuyệt đối, ổn định vô điều kiện, MỘT bước mỗi khung.
    """
    import numpy as np
    w = 2.0 * math.pi * hz
    zeta = min(max(zeta, 0.05), 0.999)
    wd = w * math.sqrt(1.0 - zeta * zeta)
    y = x[0].copy()
    v = np.zeros(x.shape[1])
    ra = np.empty_like(x)
    ra[0] = y
    for i in range(1, len(moc)):
        h = float(moc[i] - moc[i - 1])
        if h <= 1e-9:
            ra[i] = y
            continue
        e0 = y - x[i]
        tat = math.exp(-zeta * w * h)
        c, sn = math.cos(wd * h), math.sin(wd * h)
        e = tat * (e0 * c + (v + zeta * w * e0) * (sn / wd))
        v = tat * (v * c - (w * w * e0 + 2.0 * zeta * w * v) * (sn / wd))
        y = x[i] + e
        ra[i] = y
    return ra


def quan_tinh_chuoi(clip, t, bp, ctx):
    """QUÁN TÍNH CHO CHUỖI TAY — lọc góc TÍCH LUỸ, không lọc góc cục bộ.

    VÌ SAO PHẢI RIÊNG: quán tính per-xương (co_so.quan_tinh_hook) dùng tốt cho
    đoạn NGẮN (thân, đầu, hông, chân) nhưng PHÁ chuỗi tay. Lý do đo được:

        biên độ góc TÍCH LUỸ dọc chuỗi (TalkingOne, tay trái)
            đến LeftShoulder  19,4 do
            đến LeftArm       28,7 do
            đến LeftForeArm   86,6 do
            đến LeftHand      57,7 do

    Lọc từng góc CỤC BỘ thì xương thứ k trễ bằng TỔNG trễ của k xương trên nó —
    bàn tay trễ gấp 4 lần vai. Góc TƯƠNG ĐỐI khuỷu/cổ tay vì thế bị méo, dù từng
    xương vẫn mượt: nghiệm thu 10/20 -> 5/20, trượt d_omega_max và giat_max.

    CÁCH ĐÚNG: dựng góc TÍCH LUỸ dọc chuỗi (acc_k = acc_{k-1} . local_k), lọc
    MỖI acc bằng CÙNG một hằng số thời gian, rồi phân rã ngược:

        local_0' = acc_0'
        local_k' = acc_{k-1}'^-1 . acc_k'

    Khi ấy mọi xương trễ BẰNG NHAU nên góc tương đối giữ nguyên — cả cánh tay
    trôi sau như MỘT khối có khối lượng, đúng thứ mắt đọc ra là "mềm".

    Lọc trên 4 thành phần quaternion (đã ép dấu liên tục) rồi chuẩn hoá: với độ
    trễ nhỏ đây là xấp xỉ chuẩn và không có điểm kỳ dị như log map ở góc lớn.
    """
    if str(ctx.get("nguon") or "").startswith("song"):
        return
    if ctx.get("chi_do"):
        return
    import numpy as np
    from app.modules.motion.hinh_hoc import _euler_q_vec, _q_euler_vec
    hz = float(t.get("quan_tinh_hz", 0.0) or 0.0)
    if hz <= 0:
        return
    zeta = float(t.get("quan_tinh_giam_chan", 0.85))
    b = clip.get("bones") or {}
    chuoi = [x for x in bp.xuong if b.get(x) and len(b[x]) >= 4]
    if len(chuoi) < 2:
        return
    mang = {x: np.array(b[x], dtype=float) for x in chuoi}
    moc = mang[chuoi[0]][:, 0]
    if any(len(mang[x]) != len(moc) for x in chuoi):
        return                      # lưới lệch -> bỏ, đừng đoán

    cuc_bo = [_q_euler_vec(mang[x][:, 1], mang[x][:, 2], mang[x][:, 3]) for x in chuoi]
    acc, q = [], None
    for qi in cuc_bo:
        q = qi if q is None else _nhan_q(q, qi)
        acc.append(q)

    # GỘP cả chuỗi thành MỘT mảng (T, 4k) rồi lọc một lần: vòng lặp Python theo
    # thời gian là chỗ tốn, không phải phép toán. Bốn phép lọc rời tốn 28 ms/clip.
    gop = []
    for q in acc:
        q = q.copy()
        doi = np.sign(np.einsum("ij,ij->i", q[1:], q[:-1]))
        doi[doi == 0] = 1.0
        q[1:] *= np.cumprod(doi)[:, None]      # ép dấu liên tục trước khi lọc
        gop.append(q)
    r = _loc_bac_hai(np.concatenate(gop, axis=1), moc, hz, zeta)
    loc = []
    for k in range(len(acc)):
        qk = r[:, 4 * k:4 * k + 4]
        n = np.linalg.norm(qk, axis=1, keepdims=True)
        loc.append(qk / np.maximum(n, 1e-12))

    moi = [loc[0]] + [_nhan_q(_lien_hop(loc[k - 1]), loc[k]) for k in range(1, len(loc))]
    for x, q in zip(chuoi, moi):
        e = np.unwrap(_euler_q_vec(q), axis=0)
        for i, f in enumerate(b[x]):
            f[1], f[2], f[3] = float(e[i, 0]), float(e[i, 1]), float(e[i, 2])


NGON = ("Thumb", "Index", "Middle", "Ring", "Pinky")


# ĐỐT NGÓN KHÔNG ĐƯỢC DUỖI NGƯỢC. Tầm duỗi của khớp ngón người: khớp bàn-ngón
# (đốt 1) duỗi được ~30-45°, còn hai khớp liên đốt (đốt 2, 3) gần như KHÔNG duỗi
# quá thẳng. Đo Kimodo qua ống: đốt 2 duỗi ngược tới 23,9° (Thumb2), 18,8
# (Ring2), 16,4 (Middle2) — bất khả về giải phẫu. User bắt bằng mắt 27/08:
# "mặc định các đốt cong tự nhiên, khi có motion thì đốt gốc bị bẻ ngược trong
# khi các đốt phía trên cong xuôi nên không tự nhiên".
# Trần đặt CHẶT HƠN tầm giải phẫu vì phép kẹp làm trên véc-tơ xoay quanh MỘT
# trục, còn góc FK thật pha cả ba trục — kẹp ở 5° đo lại vẫn còn 8,3°. Hạ về 2°
# cho hai khớp liên đốt thì góc FK về đúng tầm người.
# Khớp bàn-ngón (đốt 1) người duỗi được 30-45°, nhưng ở đây để 8°: nguồn
# Kimodo KHÔNG có dữ liệu ngón nên mọi độ duỗi đều là nhiễu, mà đúng cái duỗi
# ở đốt gốc là thứ user nhìn thấy ("đốt gốc bẻ ngược trong khi đốt trên cong
# xuôi"). Giữ 8° để bàn tay vẫn mở ra được, không thành nắm cứng.
NGON_DUOI_TRAN = {1: 8.0, 2: 3.0, 3: 3.0}   # độ, so với tư thế nghỉ của rig
_TRUC_GAP: dict = {}


def _truc_gap_ngon(ten: str):
    """Trục GẬP của một đốt ngón, trong hệ CỤC BỘ của chính đốt đó.

    Gập là phép xoay giữ đốt con TRONG mặt phẳng chứa xương cha và xương con,
    nên trục gập chính là PHÁP TUYẾN mặt phẳng đó. Tính ở tư thế nghỉ rồi đưa
    về hệ cục bộ của đốt.

    DẤU PHẢI TỰ KIỂM (§2): pháp tuyến định nghĩa mặt phẳng chứ không định nghĩa
    chiều. Xoay thử +ε quanh trục vừa tính, góc cha-con CO LẠI thì đó là chiều
    gập, giãn ra thì lấy dấu ngược.
    BẢN ĐẦU dò theo BA TRỤC CƠ SỞ và lấy trục nào làm góc co nhiều nhất — quá
    thô: cùng một ngón mà đốt 1 ra -z còn đốt 2 ra +x, và kẹp xong đo lại vẫn
    còn duỗi ngược 8,3°.
    """
    if ten in _TRUC_GAP:
        return _TRUC_GAP[ten]
    import numpy as np
    B, n2i, par, rr, rp, _ = _rig()
    if ten not in n2i or not ten[-1].isdigit():
        _TRUC_GAP[ten] = None
        return None
    con = ten[:-1] + str(int(ten[-1]) + 1)
    idx = n2i[ten]
    cha = next((x for x, i2 in n2i.items() if i2 == par.get(idx)), None)
    if con not in n2i or cha is None:
        _TRUC_GAP[ten] = None
        return None

    def _the(rot):
        cc: dict = {}
        w = lambda n: np.asarray(B.world_transform(n2i[n], par, rot, rp, cc)[1], float)
        q = np.asarray(B.world_transform(idx, par, rot, rp, cc)[0], float)
        return w(cha), w(ten), w(con), q

    def _goc(rot):
        a, b_, c, _ = _the(rot)
        u, v = a - b_, c - b_
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-9 or nv < 1e-9:
            return None
        return float(np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1.0, 1.0))))

    a, b_, c, qw = _the(dict(rr))
    n_w = np.cross(a - b_, c - b_)               # pháp tuyến mặt phẳng, hệ WORLD
    if float(np.linalg.norm(n_w)) < 1e-9:
        _TRUC_GAP[ten] = None
        return None
    n_w = n_w / float(np.linalg.norm(n_w))
    qi = np.array([qw[0], -qw[1], -qw[2], -qw[3]], float)   # world -> cục bộ
    def _q_rot(q, v):
        w_, x, y, z = q
        t = 2.0 * np.cross([x, y, z], v)
        return v + w_ * t + np.cross([x, y, z], t)
    ax = np.asarray(_q_rot(qi, n_w), float)
    ax = ax / max(float(np.linalg.norm(ax)), 1e-9)
    g0 = _goc(dict(rr))
    if g0 is None:
        _TRUC_GAP[ten] = None
        return None
    d = np.radians(8.0)
    q = np.array([np.cos(d / 2), *(ax * np.sin(d / 2))], float)
    rot = dict(rr)
    rot[idx] = B.q_mul(np.asarray(rr[idx], float).reshape(4), q)
    g = _goc(rot)
    if g is not None and g > g0:                 # giãn ra -> lấy dấu ngược
        ax = -ax
    _TRUC_GAP[ten] = ax
    return ax


# BAO NGÓN TAY người thật, đo qua chính ống trên 97 clip mocap: dao động
# (đỉnh-đáy của lệch so với trung bình) p50 33,9 · p90 49,8 · MAX 53,9°.
NGON_DAO_TRAN = 50.0
NGON_MUOT = 3


def ngon_ve_nghi(clip, t, bp, ctx):
    """KÉO THẾ TĨNH CỦA NGÓN VỀ TƯ THẾ NGHỈ — chỉ cho clip KIMODO.

    Kimodo KHÔNG sinh chuyển động ngón: mô hình đoán 30 khớp rồi bung ra 77
    bằng cách chèn `relaxed_hands_rest_pose` HẰNG. Nhưng khâu retarget giải
    hướng xương TỪ VỊ TRÍ, mà xương ngón rất ngắn nên bài toán gần suy biến —
    nghiệm ra là NHIỄU. Đo 26/08 trên 368 clip: ngón CÁI lệch tư thế nghỉ
    trung vị 31°, tới 73,8°, và 330/368 clip lệch trên 40°. User nhìn thấy
    đúng cái đó: "ngón tay cái của tay trái dị dạng, co gập bất thường".

    CHỈ BỎ PHẦN THIÊN LỆCH, GIỮ PHẦN DAO ĐỘNG. Bản trước từng đóng băng ngón
    (bằng trung vị, rồi bằng thư viện dáng nắm/xoè) và user bác cả hai: ngón
    ghim cứng nhìn chết. Nên ở đây tách như cách ống lưng tách trung bình khỏi
    dao động (§2): kéo TRUNG BÌNH về tư thế nghỉ của rig — đó mới là bàn tay
    tự nhiên của nhân vật — còn phần rung nhỏ giữ nguyên cho ngón vẫn sống.

    KHÔNG áp cho nguồn khác: clip RPM/HY có chuyển động ngón THẬT do hoạ sĩ
    dựng (đo được lệch trung vị 25,9°, dao động tới 121°), bỏ thiên lệch của
    chúng là xoá nội dung.
    """
    if ctx.get("chi_do"):
        return
    # HAI MỨC, THEO NGUỒN THẬT CỦA NỘI DUNG NGÓN — KHÔNG theo nhãn clip.
    #   kimodo  : nguồn KHÔNG có kênh ngón -> kéo TRUNG BÌNH về nghỉ + lọc + thu
    #   còn lại : chỉ LỌC RUNG + THU BIÊN về bao người (giữ nguyên dáng tay)
    # VÌ SAO PHẢI TÁCH: bản đầu chỉ chạy khi `clip["nguon"] == "kimodo"`. Đúng
    # cho trang admin (phát thẳng clip kho) nhưng SAI cho lúc chạy thật: cử chỉ
    # Kimodo được TRỘN vào dòng GRU rồi mới qua ống, nên nhãn là "gru" và hook
    # không chạy — nhiễu ngón của Kimodo lọt thẳng xuống client. Đo cùng một
    # clip: qua đường admin ngón dao động 24,7° / rung 0,288; qua đường realtime
    # 59,9° / 0,625, trong khi người thật 33,9 / 0,243. User bắt bằng mắt 27/08:
    # "trang admin thấy chuẩn mà realtime một số motion ngón tay rất kỳ".
    _ng = str(clip.get("nguon") or "")
    ve_nghi = _ng == "kimodo"
    # THU BIÊN chỉ cho nội dung do MÔ HÌNH sinh (kimodo / dòng GRU đã trộn cử
    # chỉ kimodo). Clip do hoạ sĩ dựng (RPM idle) có dao động ngón THẬT — xén
    # là xoá nội dung: đo IdleRpm05 bị hạ 53,1 -> 25,6°. Chúng chỉ cần LỌC RUNG.
    thu_bien = ve_nghi or _ng.startswith("gru")
    import numpy as np

    B, n2i, _, _, _, nghi = _rig()
    b = clip.get("bones") or {}
    ben = "Left" if bp.ten.endswith("trai") else "Right"
    so = 0
    for f in NGON:
        # ĐỒNG BỘ CURL TRONG NGÓN (02/09): phần dao động giữ lại là NHIỄU ĐỘC
        # LẬP từng đốt — đo tương quan gập giữa các đốt cùng ngón: kho p50
        # -0,43 (đốt 1 gập trong khi đốt 2 DUỖI — "uốn éo như sâu", user:
        # "gập mở chưa tự nhiên") trong khi người thật DƯƠNG (BEAT 0,30 ·
        # IdleRpm 0,23, p90 tới 0,95). Nghiệm: chiếu dao động mỗi đốt lên
        # trục gập, lấy MỘT tín hiệu curl chung (trung bình các đốt), dựng
        # lại gập = 65% curl chung x tỉ lệ đốt (1:1:0,7) + 35% tín hiệu
        # riêng — các đốt cong THEO NHAU mà vẫn giữ độ tản người thật;
        # thành phần lệch trục thu về ≤1x RMS gập (tỉ lệ BEAT p50 1,0).
        _cua_ngon: dict = {}
        for k in (1, 2, 3):
            ten = f"{ben}Hand{f}{k}"
            tr = b.get(ten)
            g = nghi.get(ten)
            if not tr or len(tr) < 2 or g is None:
                continue
            q_n = np.asarray(B.q_from_euler_xyz(*g), float)
            q_ni = np.array([q_n[0], -q_n[1], -q_n[2], -q_n[3]], float)
            rv = []
            for fr in tr:
                q = np.asarray(B.q_from_euler_xyz(fr[1], fr[2], fr[3]), float)
                d = np.asarray(B.q_mul(q_ni, q), float)
                if d[0] < 0:
                    d = -d
                w = float(np.clip(d[0], -1.0, 1.0))
                sn = float(np.sqrt(max(1.0 - w * w, 1e-16)))
                rv.append(d[1:] / sn * (2.0 * np.arccos(w)))
            rv = np.asarray(rv, float)
            tb = rv.mean(0)
            rv = rv - tb                            # tách trung bình khỏi dao động
            # DAO ĐỘNG CŨNG LÀ ĐỒ BỊA. Kimodo không sinh ngón, nên phần rung
            # còn lại cũng do khâu giải hướng gần suy biến đẻ ra chứ không phải
            # nội dung. Đo qua chính ống: Kimodo dao động p50 55,1° và nhiễu
            # tần cao (sai phân bậc hai, p95) 0,562° — người thật 33,9° và
            # 0,243°. User bắt bằng mắt: `toe_press_stall` "các ngón tay bị
            # cong bất thường không tự nhiên".
            # KHÔNG ĐÓNG BĂNG (user đã bác: "ngón ghim cứng nhìn chết") — chỉ
            # LỌC cho hết rung rồi THU biên độ về bao người thật.
            if len(rv) >= 5:
                # TÊN RIÊNG cho cỡ cửa sổ: đặt tên `k` ở đây thì nó GHI ĐÈ chỉ
                # số đốt của vòng lặp ngoài, và mọi đốt sau đó tra trần duỗi
                # bằng khoá 7 -> rơi về mặc định 5°. Triệu chứng: hạ trần từ 5
                # xuống 0,5° mà đo lại vẫn đúng -5,0° ở MỌI đốt.
                _w = 2 * NGON_MUOT + 1
                pad = np.pad(rv, ((NGON_MUOT, NGON_MUOT), (0, 0)), mode="edge")
                rv = np.stack([np.convolve(pad[:, j], np.ones(_w) / _w, mode="valid")
                               for j in range(3)], 1)
                rv = rv - rv.mean(0)
            _cua_ngon[k] = {"ten": ten, "tr": tr, "q_n": q_n, "rv": rv, "tb": tb,
                            "tg": _truc_gap_ngon(ten)}
        if not _cua_ngon:
            continue
        # --- pha 2: đồng bộ curl trong ngón (chỉ nội dung mô hình sinh) ---
        if thu_bien:
            co_tg = {k: d for k, d in _cua_ngon.items() if d["tg"] is not None}
            if len(co_tg) >= 2:
                n_min = min(len(d["rv"]) for d in co_tg.values())
                curl = {k: d["rv"][:n_min] @ d["tg"] for k, d in co_tg.items()}
                s = np.mean(np.stack(list(curl.values())), 0)
                W_DOT = {1: 1.0, 2: 1.0, 3: 0.7}
                for k, d in co_tg.items():
                    rv, tg = d["rv"], d["tg"]
                    c = rv[:n_min] @ tg
                    off = rv[:n_min] - c[:, None] * tg[None, :]
                    c_moi = 0.65 * s * W_DOT.get(k, 1.0) + 0.35 * c
                    # đồng bộ PHA, không cong SÂU hơn biên gốc của chính đốt:
                    # curl chung từng đẩy đầu ngón lún thêm vào thân/đùi đúng
                    # 2 clip sát trần (10,4>10,0 · 8,1>8,0)
                    c_moi = np.clip(c_moi, float(c.min()), float(c.max()))
                    rms_c = float(np.sqrt((c_moi ** 2).mean()))
                    rms_o = float(np.sqrt((off ** 2).sum(1).mean()))
                    beta = min(1.0, rms_c / max(rms_o, 1e-9))
                    rv2 = np.array(rv)
                    rv2[:n_min] = c_moi[:, None] * tg[None, :] + off * beta
                    d["rv"] = rv2
        for k, d in _cua_ngon.items():
            ten, tr, q_n, rv, tb, _tg = (d["ten"], d["tr"], d["q_n"],
                                         d["rv"], d["tb"], d["tg"])
            if thu_bien:
                bien = float(np.degrees(np.linalg.norm(rv, axis=1)).max()) * 2.0
                if bien > NGON_DAO_TRAN:
                    rv = rv * (NGON_DAO_TRAN / bien)
            # KẸP DUỖI NGƯỢC — áp cho MỌI nguồn: khớp ngón người không duỗi
            # quá mức này dù clip đến từ đâu. Với nguồn có NGÓN THẬT (RPM, BEAT,
            # ARDY…) góc đo TỪ TƯ THẾ NGHỈ (q_n), tức SAU khi trả lại dáng tay.
            # Bản cũ kẹp TRƯỚC bước đó, lúc `rv` còn là dao động quanh TRUNG BÌNH
            # CỦA CLIP: clip gập ngón nhiều (IdleRpm05, biên 69°) thì mọi khung
            # duỗi hơn trung bình quá 5° bị kẹp — hai mép clip (tư thế chung của
            # bộ idle) dời 10–15°, nối clip nào sau IdleDung05 cũng giật ngón
            # (đo 10/09). Kimodo (ve_nghi) không đổi: trung bình bỏ hẳn nên `rv`
            # vốn đã tính từ nghỉ. GRU GIỮ THỨ TỰ CŨ (kẹp quanh trung bình rồi
            # mới cộng trung bình đã sửa) — đường realtime/dataset của GRU không
            # đổi khi user chưa duyệt.
            la_gru = _ng.startswith("gru")
            if not ve_nghi and not la_gru:
                rv = rv + tb                        # trả lại dáng tay của nguồn
            if _tg is not None:
                lim = np.radians(NGON_DUOI_TRAN.get(k, 5.0))
                c = rv @ _tg
                du = np.minimum(c + lim, 0.0)      # c < -lim là duỗi quá
                if float(np.abs(du).max()) > 1e-6:
                    rv = rv - du[:, None] * _tg[None, :]
            if la_gru:
                # NGUỒN GRU (02/09, user realtime: "ngón cái bẻ ngược, không
                # thẳng cùng 4 ngón"): đo Thumb1 trung bình (−5, −16, 47)° so
                # nghỉ rig (19, 8, 43) — duỗi ngược 24° ở CẢ hai trục ngoài trục
                # gập nên kẹp duỗi theo trục gập không bắt được; ngón út xoè z
                # +5 so nghỉ −22 (tách 27°). Ngón của GRU không phải nội dung
                # đáng tin (thầy DSG không có kênh ngón thật): ngón CÁI kéo
                # TRUNG BÌNH về nghỉ hẳn (giữ dao động), bốn ngón kia chỉ giữ
                # thành phần gập của trung bình, bỏ xoè/xoắn. Clip hoạ sĩ (RPM)
                # vẫn giữ nguyên dáng.
                if f == "Thumb":
                    tb = tb * 0.0
                elif _tg is not None:
                    tb = float(tb @ _tg) * _tg
                rv = rv + tb                        # trả lại dáng tay của nguồn
            for i, fr in enumerate(tr):
                g2 = float(np.linalg.norm(rv[i]))
                if g2 < 1e-9:
                    q = q_n
                else:
                    ax = rv[i] / g2
                    dq = np.array([np.cos(g2 / 2), *(ax * np.sin(g2 / 2))], float)
                    q = np.asarray(B.q_mul(q_n, dq), float)
                e = B.q_to_euler_xyz(q)
                fr[1], fr[2], fr[3] = (float(x) for x in e)
            so += 1
    if so:
        ctx.setdefault("ghi", []).append(
            f"ngón {ben}: {'về nghỉ + lọc' if ve_nghi else 'lọc rung + thu biên'} ({so} xương)")


# ĐĂNG KÝ Ở CUỐI CHUỖI, SAU quán tính — xem chỗ đăng ký thật ở cuối tệp.
# Đăng ký TRƯỚC `quan_tinh_chuoi` thì tầng lọc chuỗi chạy sau sẽ hoàn lại phần
# vừa kẹp: đo được đốt ngón vẫn duỗi ngược đúng 5,0° dù hạ trần kẹp từ 5 xuống
# 0,5° — dấu hiệu kinh điển của "ràng buộc hình học không phải tiếng nói cuối"
# (§2, cùng lý do chan.ghim_contacts phải đứng sau quán tính).


# Khuỷu KHÔNG được vào gần trục ngực hơn mức này. Đo trên 89 clip NGƯỜI THẬT
# (BEAT + Talking RPM): khuỷu gần nhất 0,126 m bên phải, 0,133 m bên trái —
# giải phẫu người không cho khuỷu chui vào trong thân. Kimodo xuống tới
# 0,015 m: 31/368 clip có khuỷu dưới 0,10 m.
KHUYU_TOI_THIEU = 0.12
KHUYU_MUOT = 4              # nửa cửa sổ làm mượt lượng đẩy (khung)


def khuyu_khong_chim(clip, t, bp, ctx):
    """ĐẨY KHUỶU RA KHỎI THÂN khi nó chui vào trong.

    User báo `alert_side_step_pivot_gesture`: "tay phải bị chìm vào body khi
    xoay người". Đo đúng: bàn tay vẫn ở 0,16 m nhưng KHUỶU tụt xuống 0,062 m
    so với trục ngực, đúng đoạn hông xoay -37°..-41°. Chìm là ở CẲNG TAY, và
    chỉ lộ khi thân xoay — nên phải đo trong HỆ TOẠ ĐỘ CỦA NGỰC, đo theo trục
    đứng của thế giới thì tay xoay theo thân sẽ không thấy gì.

    Vì sao xảy ra: tỉ lệ thân/tay của bộ xương Kimodo khác rig RPM, cùng một
    bộ góc khớp cho ra khuỷu nằm ở chỗ khác. Đây là RÀNG BUỘC HÌNH HỌC, không
    phải nội dung — người thật không làm được động tác đó.

    Phép sửa nhỏ nhất: xoay xương CÁNH TAY sao cho khuỷu ra đúng bán kính tối
    thiểu, giữ nguyên độ cao và phương vị. Cẳng tay + bàn tay là con nên đi
    theo nguyên vẹn — dáng cử chỉ không đổi, chỉ tay rời khỏi thân. Lượng đẩy
    làm mượt ±4 khung để không sinh điểm gãy (§2).
    """
    if ctx.get("chi_do"):
        return
    import numpy as np

    B, n2i, par, rr, rp, _ = _rig()
    b = clip.get("bones") or {}
    ben = "Left" if bp.ten.endswith("trai") else "Right"
    vai, khuyu = f"{ben}Arm", f"{ben}ForeArm"
    nguc = "Spine2" if "Spine2" in n2i else ("Spine1" if "Spine1" in n2i else "Spine")
    if any(x not in n2i for x in (vai, khuyu, nguc)) or vai not in b or "Hips" not in b:
        return
    moc = [f[0] for f in b["Hips"]]
    if len(moc) < 2:
        return
    hp = clip.get("hips_pos") or []
    hy = {round(f[0], 4): (f[1], f[2], f[3]) for f in hp}
    off_h = np.asarray(rp[n2i["Hips"]], float)
    tra = {n: {round(f[0], 4): f for f in trk} for n, trk in b.items()}

    # 1) đo lượng thiếu từng khung
    thieu = _do_thieu(B, n2i, par, rp, rr, b, tra, moc, hy, off_h, nguc, khuyu)
    if float(thieu.max()) < 1e-4:
        return
    k = 2 * KHUYU_MUOT + 1
    pad = np.pad(thieu, (KHUYU_MUOT, KHUYU_MUOT), mode="edge")
    thieu = np.maximum(np.convolve(pad, np.ones(k) / k, mode="valid"), thieu)

    # 2) xoay CÁNH TAY cho khuỷu ra đủ bán kính.
    # LẶP: phép xoay chỉ sửa HƯỚNG cánh tay, mà đích "giữ độ cao, nới bán kính"
    # nằm ngoài mặt cầu bán kính xương — nên một lượt chỉ đi được một phần
    # đường (đo: sâu nhất 0,025 -> 0,072 m). Lặp lại vài lượt thì tới nơi.
    n_sua = 0
    for _lap in range(4):
        n_sua += _mot_luot(B, n2i, par, rp, rr, b, tra, moc, hy, off_h,
                           nguc, vai, khuyu, thieu)
        if _lap:
            thieu = _do_thieu(B, n2i, par, rp, rr, b, tra, moc, hy, off_h,
                              nguc, khuyu)
            if float(thieu.max()) < 1e-4:
                break
    if n_sua:
        ctx.setdefault("ghi", []).append(f"đẩy khuỷu {ben} khỏi thân ({n_sua} lượt sửa)")
    return


def _tu_the_chung(B, n2i, rr, rp, b, tra, hy, off_h, tm):
    import numpy as np
    rot = dict(rr)
    for n, _trk in b.items():
        if n in n2i:
            f = tra[n].get(round(tm, 4))
            if f is not None:
                rot[n2i[n]] = B.q_from_euler_xyz(f[1], f[2], f[3])
    d = hy.get(round(tm, 4), (0.0, 0.0, 0.0))
    rp2 = dict(rp)
    rp2[n2i["Hips"]] = list(off_h + np.asarray(d, float))
    return rot, rp2


def _do_thieu(B, n2i, par, rp, rr, b, tra, moc, hy, off_h, nguc, khuyu):
    import numpy as np
    out = []
    for tm in moc:
        rot, rp2 = _tu_the_chung(B, n2i, rr, rp, b, tra, hy, off_h, tm)
        cc: dict = {}
        qn, pn = B.world_transform(n2i[nguc], par, rot, rp2, cc)
        qn = np.asarray(qn, float)
        qi = np.array([qn[0], -qn[1], -qn[2], -qn[3]], float)
        E = np.asarray(B.world_transform(n2i[khuyu], par, rot, rp2, cc)[1], float)
        v = B.q_rot(qi, E - np.asarray(pn, float))
        out.append(max(0.0, KHUYU_TOI_THIEU - float(np.hypot(v[0], v[2]))))
    return np.asarray(out, float)


def _mot_luot(B, n2i, par, rp, rr, b, tra, moc, hy, off_h, nguc, vai, khuyu, thieu):
    import numpy as np
    n_sua = 0
    for i, tm in enumerate(moc):
        if thieu[i] < 1e-4:
            continue
        rot, rp2 = _tu_the_chung(B, n2i, rr, rp, b, tra, hy, off_h, tm)
        cc: dict = {}
        qn, pn = B.world_transform(n2i[nguc], par, rot, rp2, cc)
        qn = np.asarray(qn, float); pn = np.asarray(pn, float)
        qi = np.array([qn[0], -qn[1], -qn[2], -qn[3]], float)
        S = np.asarray(B.world_transform(n2i[vai], par, rot, rp2, cc)[1], float)
        E = np.asarray(B.world_transform(n2i[khuyu], par, rot, rp2, cc)[1], float)
        v = B.q_rot(qi, E - pn)
        r = float(np.hypot(v[0], v[2]))
        if r < 1e-6:
            continue
        # ĐÍCH: cùng độ cao, cùng phương vị, bán kính = tối thiểu
        k_ = (r + thieu[i]) / r
        v2 = np.array([v[0] * k_, v[1], v[2] * k_], float)
        P = pn + B.q_rot(qn, v2)
        d0, d1 = E - S, P - S
        n0, n1 = float(np.linalg.norm(d0)), float(np.linalg.norm(d1))
        if n0 < 1e-6 or n1 < 1e-6:
            continue
        q_them = np.asarray(B.q_between(d0 / n0, d1 / n1), float)
        idx = n2i[vai]; p_idx = par.get(idx)
        p_q = (np.asarray(B.world_transform(p_idx, par, rot, rp2, {})[0], float)
               if p_idx is not None else np.array([1.0, 0, 0, 0]))
        cu_w = B.q_mul(p_q, rot[idx])
        moi_w = B.q_mul(q_them, cu_w)
        e = B.q_to_euler_xyz(np.asarray(B.q_mul(B.q_inv(p_q), moi_w), float))
        f = tra[vai].get(round(tm, 4))
        if f is not None:
            f[1], f[2], f[3] = (float(x) for x in e)
            n_sua += 1
    return n_sua


def huong_gap_khuyu(clip, ben: str, moc=None):
    """(θ gập, φ hướng gập, mốc) của khuỷu `ben` trong HỆ CỤC BỘ CÁNH TAY.

    Khớp bản lề chỉ gập về MỘT nửa mặt phẳng. Đo 28/08 trên người thật: BEAT
    và IdleRpm 100% khung nằm trong φ 225-315°. Kimodo nói chung 75% đúng;
    ZKHalfTurnPointAway 57% khung ở nửa ĐỐI DIỆN (0-90°) = khuỷu bẻ ngược khi
    chỉ tay ra sau — user thấy "tay dị dạng". Thước `khuyu_nguoc` cũ bị loại
    vì trục PCA trúng trục sấp-ngửa; thước này lấy trục xương từ rig (rp) và
    hướng gập từ FK nên không phụ thuộc pronation.
    """
    import numpy as np
    from app.modules.motion.hinh_hoc import _mau, _rig
    B, n2i, par, rr, rp, _ = _rig()
    b = clip.get("bones") or {}
    iA, iF, iH = n2i[ben + "Arm"], n2i[ben + "ForeArm"], n2i[ben + "Hand"]
    # PHẢI SAO CHÉP (np.array, không asarray): rp[iF] đã là ndarray float nên
    # `asarray` trả về CHÍNH object — `/=` bên dưới chuẩn hoá TẠI CHỖ vị trí
    # nghỉ của cẳng tay trong rig cache dùng chung -> cẳng tay dài 1 m cho mọi
    # FK sau đó trong tiến trình. 29/08: thước xuyên đầu đọc 6,26 cm ở lần đo
    # ĐẦU rồi 0,00 mãi mãi; cổng (Pool) mù với 317/318 clip, user bắt bằng mắt
    # 2 clip "tay dính vào mặt/thân" mà cổng báo đạt.
    truc = np.array(rp[iF], float); truc /= max(np.linalg.norm(truc), 1e-9)
    e1 = np.cross(truc, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-3:
        e1 = np.cross(truc, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(truc, e1)
    if moc is None:
        moc = [f[0] for f in b.get(ben + "Arm") or []]
    th, ph = [], []
    for t in moc:
        rot = dict(rr)
        for tn, tr in b.items():
            if tn in n2i:
                rot[n2i[tn]] = _mau(B, tr, t)
        c = {}
        qA, _ = B.world_transform(iA, par, rot, rp, c)
        _, pF = B.world_transform(iF, par, rot, rp, c)
        _, pH = B.world_transform(iH, par, rot, rp, c)
        v = np.asarray(pH, float) - np.asarray(pF, float)
        vl = np.asarray(B.q_rot(B.q_inv(np.asarray(qA, float)), v), float)
        vl /= max(np.linalg.norm(vl), 1e-9)
        doc = float(vl @ truc); vg = vl - doc * truc
        th.append(np.degrees(np.arccos(np.clip(doc, -1, 1))))
        ph.append(np.degrees(np.arctan2(vg @ e2, vg @ e1)) % 360.0)
    return np.array(th), np.array(ph), np.asarray(moc, float), truc


# NỬA MẶT PHẲNG GẬP HỢP LỆ THEO BÊN (φ, độ). Hệ e1/e2 dựng từ trục xương nên
# hai bên KHÔNG đối xứng gương đơn giản — phải đo riêng: người thật (BEAT +
# IdleRpm, 28/08) tay TRÁI 90-135° (100%), tay PHẢI 225-315° (100%). Bản đầu
# dùng một dải chung 210-330 cho cả hai bên -> tool báo tay trái "0% hợp lệ"
# trên 317/318 clip và định xoắn 115° mỗi clip — may là mới chạy chế độ xem.
KHUYU_PHI = {"Left": (60.0, 165.0), "Right": (195.0, 330.0)}   # chừa biên ~30°
KHUYU_PHI_TU, KHUYU_PHI_TOI = KHUYU_PHI["Right"]               # tương thích cũ
KHUYU_THETA_MIN = 15.0      # gập dưới mức này thì hướng gập vô nghĩa
KHUYU_BIEN_DO = 8.0         # kẹp φ lùi vào trong dải 8° (98 % bao — tầng kẹp nhắm đúng trần thì lấy mẫu lại luôn vượt)


def khuyu_dung_mat_phang(clip, t, bp, ctx):
    """KHUỶU GẬP ĐÚNG MẶT PHẲNG BẢN LỀ — xoắn cánh tay, bù ngược cẳng tay.

    Khung nào φ ngoài [210, 330] thì XOẮN cánh tay quanh trục xương của chính
    nó một góc Δ để φ về nửa hợp lệ, và quay cẳng tay −Δ quanh cùng trục (trong
    hệ cha) để world của cẳng tay/bàn tay KHÔNG ĐỔI: chỉ da cánh tay và nếp
    khuỷu quay về đúng chỗ. Không đụng vị trí — nên không sinh xuyên. Δ làm
    mượt ±4 khung để không nhảy tại biên (§2: ngưỡng chọn nhánh = điểm gãy).
    """
    if ctx.get("chi_do") or str(ctx.get("nguon") or "").startswith("song"):
        return
    if not int(t.get("khuyu_mat_phang", 1) or 0):        # công tắc lùi: ong.tay.khuyu_mat_phang = 0
        return
    import numpy as np
    from app.modules.motion.hinh_hoc import _rig
    B, n2i, par, rr, rp, _ = _rig()
    ben = "Left" if bp.ten == "tay_trai" else "Right"
    b = clip.get("bones") or {}
    trA, trF = b.get(ben + "Arm"), b.get(ben + "ForeArm")
    if not trA or not trF or len(trA) != len(trF):
        return
    th, ph, moc, truc = huong_gap_khuyu(clip, ben)
    # DẢI THEO BÊN + BIÊN TRONG (05/09): bản cũ dùng dải tay PHẢI cho cả hai bên
    # (KHUYU_PHI_TU/TOI) và không được đăng ký hook nào — GRU đi qua ống không
    # được nắn; cổng lô đủ: khuỷu TRÁI lệch phẳng 15–22 % ở 6/289 mẫu, toàn
    # quãng GIỮ 1,3–3,6 s với φ 55–60° sát mép dưới dải trái [60, 165]. Đích
    # kẹp lùi vào trong KHUYU_BIEN_DO để nối/hồi phía sau đẩy nhẹ vẫn trong dải.
    lo, hi = KHUYU_PHI[ben]
    lo_d, hi_d = lo + KHUYU_BIEN_DO, hi - KHUYU_BIEN_DO
    # Δ = góc xoắn đưa φ về biên gần nhất của nửa hợp lệ (0 nếu đã trong)
    d = np.zeros(len(ph))
    ngoai = (th > KHUYU_THETA_MIN) & ((ph < lo_d) | (ph > hi_d))
    if not ngoai.any():
        return
    for i in np.where(ngoai)[0]:
        a = (lo_d - ph[i]) % 360.0; c_ = (hi_d - ph[i]) % 360.0
        a = a - 360.0 if a > 180.0 else a; c_ = c_ - 360.0 if c_ > 180.0 else c_
        d[i] = a if abs(a) < abs(c_) else c_
    # làm mượt Δ (trung bình trượt ±4 khung) — chỉ trên chuỗi đã có Δ
    k = 4
    pad = np.pad(d, (k, k), mode="edge")
    d = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    n_sua = 0
    for i, (fA, fF) in enumerate(zip(trA, trF)):
        if abs(d[i]) < 0.5:
            continue
        g = np.radians(d[i]); s = np.sin(g / 2)
        qd = np.array([np.cos(g / 2), truc[0] * s, truc[1] * s, truc[2] * s])
        qA = np.asarray(B.q_from_euler_xyz(fA[1], fA[2], fA[3]), float)
        qF = np.asarray(B.q_from_euler_xyz(fF[1], fF[2], fF[3]), float)
        qA2 = B.q_mul(qA, qd)                     # xoắn cánh tay quanh trục xương (local)
        qF2 = B.q_mul(B.q_inv(qd), qF)            # bù ngược trong hệ cha -> world cẳng tay giữ nguyên
        eA = B.q_to_euler_xyz(np.asarray(qA2, float)); eF = B.q_to_euler_xyz(np.asarray(qF2, float))
        fA[1], fA[2], fA[3] = float(eA[0]), float(eA[1]), float(eA[2])
        fF[1], fF[2], fF[3] = float(eF[0]), float(eF[1]), float(eF[2])
        n_sua += 1
    if n_sua:
        ctx.setdefault("ghi", []).append(
            f"khuỷu {ben} về mặt phẳng bản lề ({n_sua} khung, xoắn tối đa {np.abs(d).max():.0f}°)")


# KHÔNG đăng ký khuyu_dung_mat_phang vào ống (user chốt 28/08: "không được
# sửa ống, phải sửa clip — mọi clip được render và hiệu chỉnh chuẩn qua tool").
# Lỗi khuỷu bẻ ngược là NỘI DUNG clip -> tools/chuan_clip_kimodo.py sửa thẳng
# vào tệp clip (bản gốc giữ ở *.goc.json); ống chỉ còn lo việc runtime.

# QUÁN TÍNH CHUỖI — sau mọi hook sửa tư thế/biên độ.
def tay_nghi_hook(clip, t, bp, ctx):
    """TAY BÙNG NỔ – GIỮ cho nguồn GRU (motion/tay_nghi.py) — chạy TRONG ống,
    TRƯỚC quán tính chuỗi + lam_mem. Bản đầu (02/09) gọi ở hau_ky_cau_gru, tức
    SAU quán tính: mép cửa sổ stroke không được quán tính hoá, hộp đen đo tay
    a_max p50 9,5–10,5 -> 12,3 m/s² trong khi a95 8 -> 4,2; user thấy "ống
    gia tốc/quán tính không hoạt động". Một lần cho cả hai tay mỗi lượt."""
    if ctx.get("chi_do") or not str(ctx.get("nguon") or "").startswith("gru"):
        return
    if ctx.get("_tay_nghi_da_chay"):
        return
    ctx["_tay_nghi_da_chay"] = True
    from app.modules.motion import tay_nghi as _tn
    r = _tn.ap(clip, ctx.get("t_chung") or {})
    if r:
        ctx.setdefault("ghi", []).append(
            f"tay nghỉ: bận {r.get('ban_truoc_pct')}% -> {r.get('ban_sau_pct')}% · v {r.get('v_truoc')} -> {r.get('v_sau')}")


tay_trai.hook(tay_nghi_hook)
tay_phai.hook(tay_nghi_hook)
# ĐÃ THỬ VÀ GỠ (05/09): đăng ký `khuyu_dung_mat_phang` cho GRU sau tay_nghi —
# Δ xoắn đổi theo khung (dù mượt ±4) sinh giật: mẫu tham chiếu adult+cheerful
# từ 0 lỗi thành a150 tay 4/37 + trần 1,01, khuỷu lệch phẳng TĂNG 15,5 → 20,0 %
# và 21,9 → 32,7 %, bẻ hướng 24–33°. Lệch phẳng của GRU nằm ở TƯ THẾ GIỮ
# (φ 55–60° sát mép dải trái) — phải sửa tư thế giữ, không xoắn theo khung.
tay_trai.hook(quan_tinh_chuoi)
tay_phai.hook(quan_tinh_chuoi)


# FOLLOW-THROUGH sau cú dừng (motion/tay_nghi.hoi_sau_dung): KHÔNG đăng ký làm
# hook bộ phận — hook trong chuỗi tay trái chạy trước quán tính tay phải và
# trước hông/chân chuẩn hoá (đổi Hips), nên cú dừng đo lúc đó không phải cú
# dừng cuối và bật tay phải bị quán tính lọc (đo 04/09). Nay là bước CHUNG
# trong lam_mem.chay sau vòng bộ phận.

# RÀNG BUỘC HÌNH HỌC ĐĂNG KÝ SAU CÙNG — sau cả quán tính. Cùng lý do chân đặt
# `dung_song`/`ghim_contacts` sau quán tính: ràng buộc phải là TIẾNG NÓI CUỐI,
# nếu không lớp làm mềm chạy sau lọc lại và kéo khuỷu chui vào thân lần nữa.
# Đo 26/08: đăng ký TRƯỚC quán tính chỉ đưa được 31 -> 8/368 clip khuỷu <0,10 m.
# Cổ tay KHÔNG gập quá mức này (độ, giữa hướng cẳng tay và hướng bàn tay).
# Đo 89 clip NGƯỜI THẬT: tv 48,4 · p90 75,2 · MAX 79,3°. Cổ tay người ngửa
# được ~70°, gập ~80° — vượt hẳn mức đó là bàn tay bẻ ngược.
CO_TAY_TOI_DA = 80.0
CO_TAY_MUOT = 4


# Kẹp về 98% bao, chừa chỗ cho sai số lấy mẫu lại — xem chú thích trong hàm.
BIEN_BAO = 0.98


def _bao_co_tay():
    """Bao giải phẫu cổ tay (độ), đọc từ trần đã hiệu chuẩn trên người thật."""
    import json as _json
    import os as _os
    mac_dinh = {"gap": 65.5, "ngua": 69.5, "lech_cai": 73.2, "lech_ut": 78.4}
    try:
        f = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                          "nguong_di_dang.json")
        t = (_json.load(open(f, encoding="utf-8")) or {}).get("tran") or {}
        ra = {}
        for chieu in mac_dinh:
            v = [t[f"co_tay_{b}_{chieu}"] for b in ("trai", "phai")
                 if f"co_tay_{b}_{chieu}" in t]
            ra[chieu] = float(min(v)) if v else mac_dinh[chieu]
        # CHỪA BIÊN. Kẹp nhắm ĐÚNG vào trần thì mọi lần LẤY MẪU LẠI sau đó
        # (luoi_chuan sau khi chuẩn nhịp, nối câu) đều có thể đẩy qua một hạt:
        # nội suy cầu giữa hai khung đều ≤35,0° vẫn cho ra 35,0000x° ở giữa,
        # vì góc giải phẫu không phải đại lượng tuyến tính theo quaternion.
        # Cổng dị dạng bắt đúng kiểu đó: hai clip "35,0 > 35,0".
        return {k: v * BIEN_BAO for k, v in ra.items()}
    except Exception:
        return {k: v * BIEN_BAO for k, v in mac_dinh.items()}


def co_tay_khong_be_nguoc(clip, t, bp, ctx):
    """KẸP CỔ TAY VỀ BAO GIẢI PHẪU — bốn chiều, không phải một trần chung.

    User: `big_tree_spread`, `both_hands_chest_greeting` (26/08), rồi
    `funnel_cause_narrow`, `half_turn_point_away`, `rub_back_of_neck`,
    `shy_retreat_half_step_hug_self`, `slow_neck_rub`, `thread_fingers_weave`
    (27/08) — "bàn tay bẻ ngược / gập quá không tự nhiên".

    VÌ SAO SINH RA: build_clip giải hướng xương BÀN TAY từ vị trí đốt ngón
    giữa. Nhưng Kimodo KHÔNG sinh ngón — cả kho dùng chung một tư thế bàn tay
    HẰNG — nên hướng đó không mang thông tin về cổ tay, và nghiệm giải ra là
    thứ khớp với đốt ngón đông cứng chứ không phải cổ tay thật.

    VÌ SAO BẢN CŨ CÒN LỌT: nó kẹp ĐỘ LỚN góc cẳng-bàn ở MỘT trần 80°. Cổ tay
    người không đẳng hướng — đo 97 clip người thật qua chính ống này: gập 65,5
    · NGỬA 69,5 · lệch ngón cái 73,2 · lệch ngón út 78,4°. Trần chung 80° cho
    phép NGỬA 80°, vượt tầm người mà thước vẫn báo đạt (funnel_cause_narrow
    ngửa 81,9° cả hai tay, thread_fingers_weave gập 81,5°).

    Nay phân rã CÓ DẤU theo hai trục giải phẫu (di_dang.co_tay_phan_ra), kẹp
    về BAO ELIP có bán trục theo đúng chiều đang lệch, rồi xoay bàn tay bằng
    phép quay NHỎ NHẤT đưa hướng bàn tay về biên (q_between). Quay nhỏ nhất
    nên không vặn lòng bàn tay — bản trước quay quanh pháp tuyến làm hướng lòng
    bàn tay lệch so với nguồn từ 11° lên 39°.

    Lượng kẹp làm mượt ±4 khung để không sinh điểm gãy (§2).
    """
    if ctx.get("chi_do"):
        return
    import numpy as np

    from app.modules.motion.di_dang import co_tay_phan_ra

    B, n2i, par, rr, rp, _ = _rig()
    b = clip.get("bones") or {}
    ben = "Left" if bp.ten.endswith("trai") else "Right"
    tay, ban, ngon = f"{ben}ForeArm", f"{ben}Hand", f"{ben}HandMiddle1"
    tro, ut = f"{ben}HandIndex1", f"{ben}HandPinky1"
    if any(x not in n2i for x in (tay, ban, ngon, tro, ut)) or ban not in b or "Hips" not in b:
        return
    moc = [f[0] for f in b["Hips"]]
    if len(moc) < 2:
        return
    hy = {round(f[0], 4): (f[1], f[2], f[3]) for f in (clip.get("hips_pos") or [])}
    off_h = np.asarray(rp[n2i["Hips"]], float)
    tra = {n: {round(f[0], 4): f for f in trk} for n, trk in b.items()}
    BAO = _bao_co_tay()

    def _tu_the(tm):
        rot = dict(rr)
        for n, _t in b.items():
            if n in n2i:
                f = tra[n].get(round(tm, 4))
                if f is not None:
                    rot[n2i[n]] = B.q_from_euler_xyz(f[1], f[2], f[3])
        d = hy.get(round(tm, 4), (0.0, 0.0, 0.0))
        rp2 = dict(rp)
        rp2[n2i["Hips"]] = list(off_h + np.asarray(d, float))
        return rot, rp2

    CAN = (tay, ban, ngon, tro, ut)

    def _du():
        """Hệ số vượt bao theo từng khung (e>1 là ngoài bao)."""
        P = {x: np.empty((len(moc), 3), float) for x in CAN}
        for i, tm in enumerate(moc):
            rot, rp2 = _tu_the(tm)
            cc: dict = {}
            for x in CAN:
                P[x][i] = np.asarray(
                    B.world_transform(n2i[x], par, rot, rp2, cc)[1], float)
        r = co_tay_phan_ra(P, ben)
        if r is None:
            return None
        gap, lech = r
        G = np.where(gap < 0, BAO["ngua"], BAO["gap"])
        L = np.where(lech < 0, BAO["lech_ut"], BAO["lech_cai"])
        return np.hypot(gap / np.maximum(G, 1e-6), lech / np.maximum(L, 1e-6))

    e = _du()
    if e is None or float(e.max()) <= 1.0:
        return
    e_dau = float(e.max())
    n_sua = 0
    # LẶP: xoay bàn tay làm đốt ngón dời theo nên một lượt chỉ khép được một
    # phần (đo bản cũ: 169,7 -> 130,8°). Lặp vài lượt thì về trong bao.
    for _lap in range(6):
        if float(e.max()) <= 1.002:
            break
        vuot = np.maximum(e - 1.0, 0.0)
        k = 2 * CO_TAY_MUOT + 1
        pad = np.pad(vuot, (CO_TAY_MUOT, CO_TAY_MUOT), mode="edge")
        vuot = np.minimum(np.convolve(pad, np.ones(k) / k, mode="valid"), vuot)
        n_sua += _kep_bao(B, n2i, par, rp, rr, b, tra, moc, _tu_the,
                          tay, ban, ngon, tro, ut, vuot, ben, BAO)
        e2 = _du()
        if e2 is None:
            break
        e = e2
    if n_sua:
        ctx.setdefault("ghi", []).append(
            f"kẹp cổ tay {ben} ({n_sua} khung, vượt bao {e_dau:.2f}x -> {float(e.max()):.2f}x)")
    return


def _q_giua(a, b):
    """Phép quay NHỎ NHẤT đưa véc-tơ a về b (quaternion w,x,y,z)."""
    import numpy as np
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    a, b = a / na, b / nb
    c = float(np.clip(a @ b, -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-9:
        tr = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(tr)) < 1e-6:
            tr = np.cross(a, np.array([0.0, 1.0, 0.0]))
        tr = tr / float(np.linalg.norm(tr))
        return np.array([0.0, *tr])
    v = np.cross(a, b)
    q = np.array([1.0 + c, *v], float)
    return q / float(np.linalg.norm(q))


def _kep_bao(B, n2i, par, rp, rr, b, tra, moc, _tu_the,
             tay, ban, ngon, tro, ut, vuot, ben, BAO):
    """Xoay bàn tay từng khung về BIÊN bao, bằng phép quay nhỏ nhất."""
    import numpy as np
    n_sua = 0
    for i, tm in enumerate(moc):
        if vuot[i] <= 1e-3:
            continue
        rot, rp2 = _tu_the(tm)
        cc: dict = {}
        w = lambda nm: np.asarray(B.world_transform(n2i[nm], par, rot, rp2, cc)[1], float)
        u = w(ban) - w(tay)
        v = w(ngon) - w(ban)
        nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        if nu < 1e-9 or nv < 1e-9:
            continue
        u, v = u / nu, v / nv
        lat = w(tro) - w(ut)
        if ben == "Right":
            lat = -lat
        lat = lat - float(lat @ u) * u
        nl = float(np.linalg.norm(lat))
        if nl < 1e-9:
            continue
        lat = lat / nl
        phap = np.cross(u, lat)
        doc = float(v @ u)
        dv = v - doc * u
        gap = np.degrees(np.arctan2(float(dv @ phap), doc))
        lech = np.degrees(np.arctan2(float(dv @ lat), doc))
        # THU TỈ LỆ CẢ HAI THÀNH PHẦN: giữ nguyên HƯỚNG lệch, chỉ rút ngắn —
        # bàn tay vẫn nghiêng đúng phía nguồn muốn, chỉ bớt quá đà.
        # ĐÍCH NHÍCH VÀO TRONG BAO 6%: nhắm đúng biên thì sau vài lượt lặp
        # (đốt ngón dời theo) còn đọng lại sát mép và trượt vì làm tròn —
        # đo lượt trước: 4 clip đọng ở 71,2-71,8° so với trần 70,86.
        he = 0.94 / (1.0 + vuot[i])
        g2 = np.radians(float(np.clip(gap * he, -85.0, 85.0)))
        l2 = np.radians(float(np.clip(lech * he, -85.0, 85.0)))
        v_dich = u + np.tan(g2) * phap + np.tan(l2) * lat
        q_them = _q_giua(v, v_dich)
        idx = n2i[ban]
        p_idx = par.get(idx)
        p_q = (np.asarray(B.world_transform(p_idx, par, rot, rp2, {})[0], float)
               if p_idx is not None else np.array([1.0, 0, 0, 0]))
        moi_w = B.q_mul(q_them, B.q_mul(p_q, rot[idx]))
        e_ = B.q_to_euler_xyz(np.asarray(B.q_mul(B.q_inv(p_q), moi_w), float))
        f = tra[ban].get(round(tm, 4))
        if f is not None:
            f[1], f[2], f[3] = (float(x) for x in e_)
            n_sua += 1
    return n_sua


# KHỐI CƠ THỂ để chống tay chìm. Kích thước ĐO TỪ RIG, không đoán:
#   nửa rộng thân = khoảng cách trục cột sống -> khớp vai (0,127 m trên Mira)
#   nửa dày thân  = 0,70 x nửa rộng
#   bán kính đùi  = 0,45 x khoảng cách hai khớp hông
# Bản đầu lấy 0,55 x nửa rộng vai = 6,9 cm — khối thân thành cái QUE nên tay
# không bao giờ "xuyên" được và thước đọc 0,0 trên mọi clip, kể cả clip user
# nhìn thấy tay chìm hẳn (bow_head_respect). Hiệu chuẩn sai vì tiêu chí chọn
# lúc đó không nhạy với kích thước.
def _khoi_co_the(B, n2i, par, rr, rp):
    import numpy as np
    cc = {}
    w = lambda n: np.asarray(B.world_transform(n2i[n], par, dict(rr), rp, cc)[1], float)
    u = w("Neck") - w("Hips")
    u /= max(float(np.linalg.norm(u)), 1e-9)
    v = w("LeftArm") - w("Spine2")
    v -= float(v @ u) * u
    a = float(np.linalg.norm(v))
    return a, a * 0.70, float(np.linalg.norm(w("LeftUpLeg") - w("RightUpLeg"))) * 0.45


DIEM_TAY = ("ForeArm", "Hand", "HandMiddle1", "HandMiddle3", "HandThumb3", "HandPinky3")


tay_trai.hook(khuyu_khong_chim)
tay_phai.hook(khuyu_khong_chim)
# ĐÃ THỬ VÀ GỠ: hook `tay_khong_chim_khoi` đẩy tay ra khỏi khối thân/đùi bằng
# cách xoay vai. Đo ra LÀM XẤU ĐI: đẩy khỏi ĐÙI thì tông vào THÂN
# (bow_head_respect xuyên thân 0,0 -> 5,2 cm) vì chốt "chỉ nhận nếu tốt hơn"
# so GIÁ TRỊ SÂU NHẤT, mà sâu nhất là đùi — giảm đùi một chút được nhận trong
# khi xuyên thân tăng âm thầm. Đẩy đúng cần bộ giải NHIỀU RÀNG BUỘC cùng lúc,
# không phải xoay một xương theo một hướng. Chưa làm; hiện dùng
# `tools/kiem_tu_dong.py --go` để CÁCH LY clip vi phạm.
tay_trai.hook(co_tay_khong_be_nguoc)
tay_phai.hook(co_tay_khong_be_nguoc)


# RÀNG BUỘC NGÓN LÀ TIẾNG NÓI CUỐI của ống tay.
tay_trai.hook(ngon_ve_nghi)
tay_phai.hook(ngon_ve_nghi)
