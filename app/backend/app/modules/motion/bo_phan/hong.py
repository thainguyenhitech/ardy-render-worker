"""HÔNG — kẹp góc xoay + trần vận tốc góc.

KHÔNG phải IK chân: xương chân trong clip đã khoá 0,0°, hips_pos = 0; 'trượt
chân' thật ra là Hips xoay 7,6° quét cả thân (chân dài 0,8 m -> ~10 cm).
"""
from __future__ import annotations

import math

from app.modules.motion.bo_phan.co_so import (BoPhan, la_ardy_toan_than,
                                              quan_tinh_hook, yeu_cau_gian)
from app.modules.motion.bo_phan.do_bo_phan import toc_goc

hong = BoPhan(ten="hong", xuong=("Hips",), diem_do=("Hips",),
              mac_dinh={"xoay_toi_da": 0.0, "nguon_kep": ("gru",), "toc_goc_tran": 2.1, "gia_toc_goc_tran": 12.0})



@hong.hook
def tu_the_chuan(clip, t, bp, ctx):
    """Nguồn giọng nói (GRU): Hips = tư thế nghỉ rig, HẰNG cả clip.

    GRU "để yên" Hips nhưng ở giá trị khung đầu của TỪNG câu -> mỗi câu một
    góc hông, cầu nối câu sau xoay cả thân dưới -> chân chuẩn (cứng) bị quét,
    bàn chân trượt ("thân dưới dao động qua lại kéo theo chân"). Gốc đứng yên
    thì chân đứng yên; lắc lư thuộc về cột sống (song.py, thân). Tắt bằng
    profile.ong.hong.chuan = 0. Clip kho giữ nguyên (quay người, nhìn lại...).
    """
    if ctx.get("chi_do"):
        return
    if not t.get("chuan", 1):
        return
    ng = str(ctx.get("nguon") or "")
    if not ng.startswith("gru"):
        return
    from app.modules.motion.hinh_hoc import _rig
    *_, nghi = _rig()
    e = nghi.get("Hips")
    tr = (clip.get("bones") or {}).get("Hips")
    if e is None or not tr:
        return
    ex, ey, ez = (round(float(v), 5) for v in e)
    for f in tr:
        f[1], f[2], f[3] = ex, ey, ez
    ctx.setdefault("ghi", []).append("hông tư thế chuẩn (hằng)")


# Dưới mốc này, tịnh tiến y của hông là ARTEFACT retarget, không phải nội
# dung: bộ Idle RPM dao động <= 1,25 cm; khuỵu/nhún/ngồi đều trên 2 cm.
DAO_DONG_Y_TOI_THIEU = 0.02


@hong.hook
def hips_pos_kho(clip, t, bp, ctx):
    """Bỏ ARTEFACT nhấc hông của khâu retarget — KHÔNG bỏ nội dung.

    Ý đồ gốc: retarget để lại hông nhấc 5–6 cm trong clip Idle (hands clasp)
    -> client ease cả người lên = bước gia tốc ở đầu/tay, bàn chân rời sàn rồi
    IK kéo về. x/z (dồn trọng tâm) giữ.

    TIÊU CHÍ LÀ DAO ĐỘNG, KHÔNG PHẢI max|y|. Bản cũ xoá y khi max|y| < 8 cm.
    Artefact retarget là một DỊCH CHUNG gần hằng (dao động ~0), còn khuỵu gối /
    nhún / hạ người là DAO ĐỘNG — và nó nằm gọn dưới 8 cm, nên bản cũ xoá luôn
    cả nội dung. Kimodo diễn động tác hạ thấp bằng HẠ GỐC + CO CHÂN; bỏ vế đầu
    thì hông đứng nguyên độ cao mà chân vẫn co = NGƯỜI BAY. Đo 26/08 trên
    ZKEmphasizeNumberContrast (hạ gốc 7,7 cm): bàn chân sau ống 0,350 -> 0,787 m;
    cả kho 3 -> 31 clip chân rời sàn. Cùng bài học mà `kep_goc` ngay dưới đã học
    cho TRỤC XOAY ("clip dựng sẵn thì hông CHÍNH LÀ nội dung"), chỉ chưa học cho
    trục TỊNH TIẾN. Mốc: Idle RPM dao động y <= 1,25 cm, mọi clip nội dung đều
    trên 2 cm.
    """
    if ctx.get("chi_do"):
        return
    ng = str(ctx.get("nguon") or "")
    if not ng.startswith("kho"):
        return
    hp = clip.get("hips_pos") or []
    if len(hp) < 2:
        return
    ys = [float(f[2]) for f in hp]
    dao_dong = max(ys) - min(ys)
    if dao_dong < DAO_DONG_Y_TOI_THIEU:
        ymax = max(abs(v) for v in ys)
        for f in hp:
            f[2] = 0.0
        if ymax > 0.005:
            ctx.setdefault("ghi", []).append(
                f"bỏ nhấc hông {ymax*100:.1f}cm (kho, dao động {dao_dong*100:.1f}cm)")


@hong.hook
def kep_goc(clip, t, bp, ctx):
    # CHỈ nguồn giọng nói. Ý đồ "gốc phải đứng yên" (CLAUDE.md §8) là dành cho
    # GRU: ở đó Hips không mang thông tin, lắc Hips là quét cả hai chân. Clip
    # DỰNG SẴN thì độ xoay hông CHÍNH LÀ nội dung (cúi, chạm mũi chân, ngả
    # người) và chân đã được dựng khớp với nó.
    #
    # Kẹp hông mà KHÔNG chỉnh chân theo = phá thế cân bằng: đo được bàn chân của
    # ActionTouchToes nhấc từ 5,0 lên 48,3 cm, ActionBowDeep 2,8 -> 25,7,
    # GLeanForwardTorso 3,2 -> 38,2 (19 clip toàn kho, 2026-08-23).
    ng = str(ctx.get("nguon") or "")
    if ng.startswith("song") or not any(
            ng.startswith(p) for p in (t.get("nguon_kep") or ("gru",))):
        return
    gh = float(t.get("xoay_toi_da") or ctx["t_chung"].get("hong_xoay_toi_da") or 0.0)
    gh = math.radians(gh)
    tr = (clip.get("bones") or {}).get("Hips")
    if gh <= 0 or not tr:
        return
    from app.modules.motion.bo_phan.co_so import trong_so_cua_so
    RONG = math.radians(30.0)     # trong cửa sổ toàn thân: hông xoay là NỘI DUNG
    goc = tr[0][1:4]
    for f in tr:
        w = trong_so_cua_so(clip, f[0])
        gh_f = gh * (1.0 - w) + RONG * w
        for k in (1, 2, 3):
            f[k] = max(goc[k - 1] - gh_f, min(goc[k - 1] + gh_f, f[k]))


@hong.hook
def tran_toc_do(clip, t, bp, ctx):
    # CLIP ARDY TOÀN THÂN: hông quay nhanh là NỘI DUNG (xoay người, chạy vòng), không phải "vượt trần
    # người đang nói" — trần lấy từ mocap BEAT/IdleRpm vốn không chứa hành vi đó (CLAUDE.md §3). Đo
    # 16/09 trên clip kho: "turns to the right" xin giãn ×1,85 vì hông 3,12 rad/s → clip 7,43 s phát
    # thành 14,05 s, bước chân chậm mà quãng giữ nguyên = LẾT CHÂN trên sàn (user báo "trôi chân").
    if la_ardy_toan_than(clip, ctx):
        return
    g = toc_goc(clip, bp.xuong)
    k = 1.0
    tr = float(t.get("toc_goc_tran") or 0)
    if tr > 0 and g["v95"] > tr:
        k = max(k, g["v95"] / tr)
    tr = float(t.get("gia_toc_goc_tran") or 0)
    if tr > 0 and g["a95"] > tr:
        k = max(k, math.sqrt(g["a95"] / tr))
    if k > 1.0:
        yeu_cau_gian(ctx, bp, f"hông {g['v95']:.2f} rad/s", k)


# QUÁN TÍNH đăng ký CUỐI: phải chạy SAU các hook sửa tư thế, để nó làm mềm
# quỹ đạo CUỐI CÙNG chứ không phải quỹ đạo giữa chừng.
hong.hook(quan_tinh_hook(2.0))   # hông nặng nhất, dẫn cả người — trôi sau nhiều nhất
