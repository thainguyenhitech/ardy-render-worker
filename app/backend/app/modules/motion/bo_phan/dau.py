"""ĐẦU — tư thế đầu theo ngữ cảnh + trần vận tốc/gia tốc góc.

Trước đây đầu KHÔNG qua cửa nào: tầng tốc độ chỉ nhìn bàn tay, mà đầu không
nằm trên chuỗi tới tay. Gật/quay đầu quá nhanh là dấu hiệu robot điển hình —
đo kho Talking*: 1,67 rad/s, gấp đôi mocap 0,78.
"""
from __future__ import annotations

import math

from app.modules.motion.bo_phan.co_so import BoPhan, quan_tinh_hook, yeu_cau_gian, yeu_cau_loc
from app.modules.motion.bo_phan.do_bo_phan import toc_goc, toc_world
from app.modules.motion.hinh_hoc import _rig

dau = BoPhan(
    ten="dau", xuong=("Neck", "Head"), diem_do=("Head",),
    mac_dinh={
        "cui_do": 0.0,            # + cúi xuống, − ngẩng (độ) — cộng đều lên mọi khung
        "nghieng_do": 0.0,        # nghiêng sang bên (độ)
        "toc_goc_tran": 1.7, "gia_toc_goc_tran": 18.0,
        "toc_world_tran": 0.7, "v_vot": 2.0,
    })



@dau.hook
def du_track(clip, t, bp, ctx):
    """NGUỒN GIỌNG NÓI PHẢI PHÁT ĐỦ TRACK Neck + Head — client không phải đoán.

    GRU không sinh track Head (chỉ Neck). Xương không có track thì client BỎ
    NGUYÊN hiện trạng: idle quay đầu 33-50° (IdleRpm02/04/06/10), user gõ tin
    đúng lúc đầu đang quay -> tư thế bị chụp giữ, và SUỐT lượt nói không có
    dữ liệu nào kéo đầu về chính diện (user bắt 2026-08-23: "idle đầu đang
    quay bên nào thì talking vẫn giữ nguyên"). Cùng bài với tu_the_chuan của
    chân: thiếu track thì thêm track TƯ THẾ NGHỈ — client ease về chính diện,
    các lớp gật/gaze phía trên cộng tiếp như thường.
    """
    if ctx.get("chi_do"):
        return
    ng = str(ctx.get("nguon") or "")
    if ng.startswith("song") or not ng.startswith("gru"):
        return
    b = clip.setdefault("bones", {})
    moc = sorted({f[0] for tr in b.values() for f in tr})
    if not moc:
        return
    _, _, _, _, _, nghi = _rig()
    so = 0
    for ten in bp.xuong:                      # Neck, Head
        if ten in b and b[ten]:
            continue
        e = nghi.get(ten)
        if e is None:
            continue
        ex, ey, ez = (round(float(v), 5) for v in e)
        b[ten] = [[tm, ex, ey, ez] for tm in moc]
        so += 1
    if so:
        ctx.setdefault("ghi", []).append(f"đầu: thêm {so} track nghỉ (thiếu từ nguồn)")


@dau.hook
def nhin_thang(clip, t, bp, ctx):
    """NGUỒN GIỌNG NÓI (GRU): đầu LUÔN về chính diện — khử TRÔI CHẬM cả ba trục.

    GRU để đầu ngước lên, nghiêng xéo, lệch yaw từng quãng vài giây (người xem:
    "không nhìn chính diện"). Cách làm: trên véc-tơ xoay của Neck và Head, tách
    phần CHẬM (trung bình trượt cửa sổ `nhin_thang_cua_so_s`, mặc định 1,2 s) và
    thay bằng tư thế NGHỈ của rig (= nhìn thẳng); phần NHANH (gật, lắc, liếc,
    nhấn nhịp) giữ nguyên. Chạy TRƯỚC hook tư thế tính cách (cúi/nghiêng theo
    tính cách cộng sau). Clip cử chỉ kho KHÔNG qua hook này. Tắt:
    profile.ong.dau.nhin_thang = 0.
    """
    if ctx.get("chi_do"):
        return
    if not t.get("nhin_thang", 1):
        return
    if not str(ctx.get("nguon") or "").startswith("gru"):
        return
    import numpy as np
    from app.modules.motion.hinh_hoc import (_euler_q_vec, _exp_q_vec, _lien_tuc,
                                              _log_q_vec, _q_euler_vec, _rig)
    *_, nghi = _rig()
    b = clip.get("bones") or {}
    cua_so = float(t.get("nhin_thang_cua_so_s", 1.2) or 1.2)
    ghi = []
    for ten in ("Neck", "Head"):
        tr = b.get(ten)
        if not tr or len(tr) < 4 or ten not in nghi:
            continue
        a = np.asarray(tr, dtype=float)
        rv = _log_q_vec(_lien_tuc(_q_euler_vec(a[:, 1], a[:, 2], a[:, 3])))
        e0 = nghi[ten]
        rv0 = _log_q_vec(_q_euler_vec(np.array([e0[0]]), np.array([e0[1]]), np.array([e0[2]])))[0]
        dt = float(np.median(np.diff(a[:, 0]))) if len(a) > 1 else 1 / 60
        w = max(3, int(round(cua_so / max(dt, 1e-3))))
        if w % 2 == 0:
            w += 1
        h = w // 2
        # trung bình trượt có đệm biên (edge pad) -> không kéo hai đầu về 0
        pad = np.concatenate([np.repeat(rv[:1], h, axis=0), rv, np.repeat(rv[-1:], h, axis=0)], axis=0)
        cs = np.cumsum(pad, axis=0)
        ma = (cs[w - 1:] - np.concatenate([np.zeros((1, 3)), cs[:-w]], axis=0)) / w
        ma = ma[: len(rv)]
        troi = rv.mean(axis=0) - rv0
        rv_moi = rv - ma + rv0
        # NGƯỚC BÙ: tư thế nghỉ của rig đã gập đầu ~8° ra trước (đo trục đầu
        # Head->HeadTop_End); "chính diện" = trục đầu thẳng đứng -> xoay Head thêm
        # −nguoc_do quanh x local (x+ = cúi, đo FK). Chỉ áp cho Head, không Neck.
        if ten == "Head":
            # 8,0 = trục đầu thẳng đứng tuyệt đối — đo 02/09 thì người thật
            # đứng nói (idle RPM, F_Talking) giữ đỉnh đầu ngả TRƯỚC +5,6°, rig
            # nghỉ +8,0°; ép 0° là "hếch" so với người. Mặc định 2,5 -> +5,5°.
            nguoc = float(t.get("nhin_thang_nguoc_do", 2.5) or 0.0)
            if abs(nguoc) > 1e-6:
                from app.personality.than_hinh import xoay_local
                from app.modules.motion.hinh_hoc import _rig as _rig2
                B2, *_ = _rig2()
                e_all = _euler_q_vec(_exp_q_vec(rv_moi))
                e_all = np.array([xoay_local(B2, list(e_all[i3]), 0, -np.radians(nguoc)) for i3 in range(len(e_all))])
                rv_moi = _log_q_vec(_lien_tuc(_q_euler_vec(e_all[:, 0], e_all[:, 1], e_all[:, 2])))
        e = _euler_q_vec(_exp_q_vec(rv_moi))
        for i2, f in enumerate(tr):
            f[1], f[2], f[3] = round(float(e[i2, 0]), 5), round(float(e[i2, 1]), 5), round(float(e[i2, 2]), 5)
        ghi.append(f"{ten} trôi tb ({np.degrees(troi[0]):+.0f},{np.degrees(troi[1]):+.0f},{np.degrees(troi[2]):+.0f})°")
    # GIỮ HƯỚNG ĐẦU TRONG HỆ THẾ GIỚI (02/09). Reset Neck về nghỉ rig ở trên
    # xoá luôn phần `than._bu_co` đã bù cho lưng: lưng khom bao nhiêu, đầu cúi
    # theo bấy nhiêu (đo 5 mẫu dataset: pitch trục đầu elderly +26°, heavy
    # +18–21°, người thật đứng nói +5,6°). Nay đo FK trục Head->HeadTop_End
    # từng khung, lấy thành phần CHẬM (cùng cửa sổ), xoay Neck bù về đích
    # `nhin_thang_pitch_dich_do` (mặc định 5,5 = idle RPM). Gật/nhấn nhịp
    # (thành phần nhanh) giữ nguyên.
    try:
        from app.modules.motion.hinh_hoc import _rig as _rig3
        from app.personality.than_hinh import xoay_local as _xl
        dich = float(t.get("nhin_thang_pitch_dich_do", 5.5))
        B3, n2i3, par3, rr3, rp3, _ = _rig3()
        chuoi = [x for x in ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head") if x in b and x in n2i3]
        tr_neck = b.get("Neck")
        if tr_neck and "Head" in chuoi and "HeadTop_End" in n2i3 and len(tr_neck) >= 4:
            n_k = len(tr_neck)
            cung_luoi = all(len(b[x]) == n_k for x in chuoi)
            tra3 = None if cung_luoi else {x: {round(f[0], 4): f for f in b[x]} for x in chuoi}
            p = np.zeros(n_k)
            for i3, f0 in enumerate(tr_neck):
                rot = dict(rr3)
                for x in chuoi:
                    f = b[x][i3] if cung_luoi else tra3[x].get(round(f0[0], 4))
                    if f is not None:
                        rot[n2i3[x]] = B3.q_from_euler_xyz(f[1], f[2], f[3])
                cc: dict = {}
                a3 = np.asarray(B3.world_transform(n2i3["Head"], par3, rot, rp3, cc)[1], float)
                b3 = np.asarray(B3.world_transform(n2i3["HeadTop_End"], par3, rot, rp3, cc)[1], float)
                d3 = b3 - a3
                p[i3] = np.degrees(np.arctan2(d3[2], d3[1]))          # +z = đỉnh đầu ngả trước
            dt3 = float(np.median(np.diff([f[0] for f in tr_neck]))) or 1 / 60
            w3 = max(3, int(round(cua_so / max(dt3, 1e-3)))); w3 += (w3 % 2 == 0); h3 = w3 // 2
            pad3 = np.concatenate([np.repeat(p[:1], h3), p, np.repeat(p[-1:], h3)])
            cs3 = np.cumsum(pad3)
            p_cham = (cs3[w3 - 1:] - np.concatenate([[0.0], cs3[:-w3]]))[:n_k] / w3
            sai = p_cham - dich
            for i3, f in enumerate(tr_neck):
                e3 = _xl(B3, [f[1], f[2], f[3]], 0, -np.radians(float(sai[i3])))
                f[1], f[2], f[3] = round(float(e3[0]), 5), round(float(e3[1]), 5), round(float(e3[2]), 5)
            ghi.append(f"pitch đầu thế giới {p_cham.mean():+.1f}° -> {dich:+.1f}°")
    except Exception as _e:  # noqa: BLE001
        ghi.append(f"giữ hướng đầu bỏ qua: {type(_e).__name__}")
    if ghi:
        ctx.setdefault("ghi", []).append("nhìn thẳng (khử trôi " + f"{cua_so:.1f}s): " + " · ".join(ghi))


@dau.hook
def tu_the(clip, t, bp, ctx):
    """Cộng độ cúi/nghiêng cố định vào Head. Ví dụ: tính cách rụt rè cúi 4°,
    nhân vật 'nói chuyện luôn cúi nhẹ' đặt cui_do theo type. Chia đều Neck/Head
    để cổ không gãy khúc."""
    if ctx.get("chi_do"):
        return
    cui = math.radians(float(t.get("cui_do") or 0.0))
    ngh = math.radians(float(t.get("nghieng_do") or 0.0))
    if abs(cui) < 1e-6 and abs(ngh) < 1e-6:
        return
    b = clip.get("bones") or {}
    co = [x for x in bp.xuong if x in b]
    if not co:
        return
    for ten in co:
        for f in b[ten]:
            f[1] += cui / len(co)
            f[3] += ngh / len(co)


@dau.hook
def dau_song_hook(clip, t, bp, ctx):
    """ĐẦU SỐNG KHI NÓI (bob theo năng lượng, gật theo nhấn âm, nghiêng cảm
    xúc, bám liếc) — motion/dau_song.py, chạy TRONG ống: sau nhìn thẳng + tư
    thế tính cách, TRƯỚC trần tốc độ và quán tính đầu. Bản đầu (02/09) cộng ở
    hau_ky_cau_gru tức SAU ong.chay — nằm ngoài trần/quán tính/làm mềm, trái
    "ống là cửa duy nhất" (user: "check lại cấu trúc các tầng"). Dữ liệu vào
    qua ctx["mat"] = {frames, gaze, emotion, level, pha, talk_head_bob,
    gaze_dau_theo, cuong_do} do gru_provider nhận từ orchestrator/dong_goi_mau.
    Chỉ nguồn gru, một lần mỗi lượt (lam_mem chạy chuỗi hai lượt, lượt 2 chi_do)."""
    if ctx.get("chi_do") or not str(ctx.get("nguon") or "").startswith("gru"):
        return
    mat = ctx.get("mat")
    if not mat or ctx.get("_dau_song_da_chay"):
        return
    ctx["_dau_song_da_chay"] = True
    from app.modules.motion import dau_song as _ds
    r = _ds.ap_dau_khi_noi(clip, mat.get("frames"), mat.get("gaze") or [],
                           mat.get("emotion") or "neutral", int(mat.get("level") or 2),
                           mat, float(mat.get("pha") or 0.0))
    clip["_dau_song"] = r
    ctx.setdefault("ghi", []).append(f"đầu sống: gật {r.get('gat_n')} · lệch bám {r.get('lech_max_do')}°")


@dau.hook
def tran_toc_do(clip, t, bp, ctx):
    """Đầu quay quá nhanh -> xin giãn; vọt một khung -> xin lọc riêng."""
    g = toc_goc(clip, bp.xuong)
    w = toc_world(clip, bp.diem_do)
    k = 1.0
    tr = float(t.get("toc_goc_tran") or 0)
    if tr > 0 and g["v95"] > tr:
        k = max(k, g["v95"] / tr)
    tr = float(t.get("gia_toc_goc_tran") or 0)
    if tr > 0 and g["a95"] > tr:
        k = max(k, math.sqrt(g["a95"] / tr))
    tr = float(t.get("toc_world_tran") or 0)
    if tr > 0 and w["v95"] > tr:
        k = max(k, w["v95"] / tr)
    if k > 1.0:
        yeu_cau_gian(ctx, bp, f"đầu {g['v95']:.2f} rad/s · {g['a95']:.1f} rad/s²", k)
    vot = float(t.get("v_vot") or 0)
    if vot > 0 and w["v_max"] > vot:
        yeu_cau_loc(ctx, bp, f"đầu vọt {w['v_max']:.1f} m/s một khung")


# QUÁN TÍNH đăng ký CUỐI: phải chạy SAU các hook sửa tư thế, để nó làm mềm
# quỹ đạo CUỐI CÙNG chứ không phải quỹ đạo giữa chừng.
dau.hook(quan_tinh_hook(2.6))   # đầu nhẹ hơn thân, bám sát hơn
