"""NEO CLIP ARDY XUỐNG MẶT SÀN — một phép dời hằng số theo phương đứng.

VÌ SAO CẦN: `ardy_retarget` đặt `hips_pos = root − root_khung0`, tức cao độ của CẢ CLIP neo
vào KHUNG ĐẦU của ARDY. Khung đầu không phải lúc nào cũng là thế đứng — khuếch tán có thể mở
màn ở tư thế đã khuỵu hoặc đã nhấc chân — nên cả clip lệch theo. Đo 09/09 trên 8 cue:

    "A person crouches down and stands up."  bàn chân KHÔNG BAO GIỜ xuống dưới 41,4 cm
    gói thật gửi client (hộp đen backend)    chan_lun_san_cm 37,9 · chan_lo_lung_cm 21,0
                                             goi_trai_nguoc_max 14,2 (gối bẻ NGƯỢC)

Đó chính là "biến dạng cơ thể" user thấy: nhân vật lún qua sàn rồi lại treo lơ lửng.

PHÉP SỬA NHỎ NHẤT CÓ THỂ: dời cả clip một hằng số cho bàn chân THẤP NHẤT chạm sàn. Không đụng
thời gian, không đụng một góc khớp nào, không trộn — LUẬT SỐ 1 của bản dựng này còn nguyên.
(Cùng ý với `ardy_goc.san_rig()` của `backend/`, nhưng gọn còn một phép trừ.)
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SAN: list = []


def san_rig() -> float:
    """Cao độ bàn chân của rig lúc nghỉ = mặt sàn."""
    if not _SAN:
        from app.modules.motion.hinh_hoc import _rig
        B, n2i, par, rr, rp, _ = _rig()
        cc: dict = {}
        _SAN.append(min(B.world_transform(n2i[n], par, dict(rr), rp, cc)[1][1]
                        for n in ("LeftFoot", "RightFoot")))
    return _SAN[0]


def chan_thap_nhat(clip: dict) -> float | None:
    """Cao độ bàn chân thấp nhất của clip (FK world). None nếu clip thiếu dữ liệu.

    QUÉT TỪNG KHUNG, không lấy mẫu thưa: cực tiểu của bàn chân là một điểm NHỌN (đúng khoảnh
    khắc chạm đất), lấy mẫu 4 khung/lần trượt mất nó — đo 09/09 còn sót lún 3,0 cm. Chuỗi FK
    chỉ dài 4 xương nên quét cả clip vẫn rẻ, và nó chạy ở luồng sinh ARDY, không ở event loop.
    """
    hp = clip.get("hips_pos") or []
    tr0 = (clip.get("bones") or {}).get("Hips") or []
    if not hp or not tr0:
        return None
    import numpy as np
    from app.modules.motion.hinh_hoc import _rig
    B, n2i, par, rr, rp, _ = _rig()
    hpa = np.asarray(hp, float)
    thap = None
    for k in range(len(tr0)):
        rot = dict(rr)
        for n, tr in clip["bones"].items():
            if n in n2i and k < len(tr):
                rot[n2i[n]] = B.q_from_euler_xyz(tr[k][1], tr[k][2], tr[k][3])
        rp2 = dict(rp)
        t = tr0[k][0]
        rp2[n2i["Hips"]] = [rp[n2i["Hips"]][j] + float(np.interp(t, hpa[:, 0], hpa[:, 1 + j]))
                            for j in range(3)]
        cc: dict = {}
        y = min(B.world_transform(n2i[n], par, rot, rp2, cc)[1][1]
                for n in ("LeftFoot", "RightFoot"))
        thap = y if thap is None else min(thap, y)
    return thap


def nang_khoi_san(clip: dict, k0: int = 0, n: int | None = None) -> float:
    """Nâng hông đúng phần bàn chân thủng sàn, bằng TRƯỜNG HIỆU CHỈNH ĐÃ LÀM MƯỢT. Trả mức nâng.

    Khác `neo`: `neo` dời cả clip một HẰNG SỐ (chỗ đứng của clip), còn hàm này sửa TỪNG KHUNG và
    CHỈ MỘT PHÍA — chỉ nâng chỗ thủng, không bao giờ hạ. Dùng cho những chỗ mà một tầng khác vừa
    ghi đè lên chân: mép đoạn ARDY (`chuyen_tiep`) và cầu nối giữa hai câu của orchestrator.

    ĐÃ THỬ KẸP TRỌNG SỐ BÙ THEO TỪNG KHUNG — SAI, đúng kiểu §2 đã cảnh báo ("mọi ngưỡng chọn
    nhánh trong chuỗi thời gian đều thành điểm gãy"): khung này kẹp còn 0,3, khung kế 0,9, thế
    là đẻ ra cú giật MỚI ngay chỗ vừa chữa — A/B tất định: lún 1,2 → 1,0 cm nhưng bước quaternion
    10,9 → 61,3 °/khung. Trường hiệu chỉnh làm mượt thì không đụng góc khớp nên không sinh điểm
    gãy nào: lún 1,7 → 0,1 cm mà bước quaternion giữ nguyên 5,0 °/khung.
    """
    import numpy as np
    from app.modules.motion.hinh_hoc import _rig
    B, n2i, par, rr, rp, _ = _rig()
    CAP = (("LeftUpLeg", "LeftLeg", "LeftFoot"), ("RightUpLeg", "RightLeg", "RightFoot"))
    if any(x not in n2i for c in CAP for x in c):
        return 0.0
    b = clip.get("bones") or {}
    hp = clip.get("hips_pos") or []
    tr0 = b.get("Hips")
    if not hp or not tr0:
        return 0.0
    hpa = np.asarray(hp, float)
    san = san_rig()
    k1 = min(len(tr0), k0 + n) if n else len(tr0)
    thieu = []
    for k in range(k0, k1):
        rot = dict(rr)
        for x, tr in b.items():
            if x in n2i and k < len(tr):
                rot[n2i[x]] = B.q_from_euler_xyz(tr[k][1], tr[k][2], tr[k][3])
        t = tr0[k][0]
        rp2 = dict(rp)
        rp2[n2i["Hips"]] = [rp[n2i["Hips"]][j] + float(np.interp(t, hpa[:, 0], hpa[:, 1 + j]))
                            for j in range(3)]
        cc: dict = {}
        y = min(B.world_transform(n2i[c[2]], par, rot, rp2, cc)[1][1] for c in CAP)
        thieu.append(max(0.0, san - y))
    if not thieu or max(thieu) < 0.002:            # dưới 2 mm: mắt không thấy
        return 0.0
    for _ in range(2):                             # trung bình động hai lượt, ±4 khung
        m = list(thieu)
        for i in range(len(thieu)):
            a, z = max(0, i - 4), min(len(thieu), i + 5)
            thieu[i] = sum(m[a:z]) / (z - a)
    # hips_pos có thể ở lưới khác track xương -> cộng theo THỜI GIAN, không theo chỉ số
    moc = [tr0[k][0] for k in range(k0, k1)]
    for f in hp:
        if moc[0] - 1e-6 <= f[0] <= moc[-1] + 1e-6:
            f[2] = float(f[2]) + float(np.interp(f[0], moc, thieu))
    return max(thieu)


def neo(clip: dict, giu: list) -> float:
    """Dời `hips_pos` theo phương đứng cho bàn chân thấp nhất chạm sàn. Trả độ dời (m).

    `giu` là ô nhớ MỘT phần tử giữ độ dời đã chốt. Bộ đệm lớn dần theo luồng ARDY, mà miếng
    sau có thể xuống thấp hơn miếng đầu (ARDY lặp động tác, lần lặp sau hạ sâu hơn) — đo
    10/09: neo theo miếng đầu rồi giữ nguyên thì câu dùng phần đuôi clip lún 5,4 cm.

    LUẬT: CHỈ NÂNG THÊM, KHÔNG BAO GIỜ HẠ. Nâng thì chân rời khỏi sàn — mắt không bắt được vì
    nó xảy ra đúng lúc bàn chân đang ở điểm thấp nhất. Hạ thì chân cắm xuống sàn, thấy ngay.
    Và vì đơn điệu nên mốc hội tụ, không dao động qua lại giữa các câu.
    """
    hp = clip.get("hips_pos") or []
    if not hp:
        return 0.0
    try:
        thap = chan_thap_nhat(clip)
        can = 0.0 if thap is None else float(san_rig() - thap)
    except Exception:  # noqa: BLE001
        logger.warning("neo sàn bỏ qua", exc_info=True)
        can = 0.0
    giu[0] = can if giu[0] is None else max(float(giu[0]), can)
    dy = float(giu[0])
    if abs(dy) < 0.002:                     # dưới 2 mm: mắt không thấy, đừng đụng vào dữ liệu
        return 0.0
    for f in hp:
        f[2] = float(f[2]) + dy
    clip["_neo_san_cm"] = round(dy * 100, 1)
    return dy
