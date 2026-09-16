"""CHÂN — TƯ THẾ CHUẨN cho nguồn sinh từ giọng nói.

Chuyển động sinh từ giọng nói (GRU) và vòng sống không mang thông tin thân
dưới. Trước đây mỗi nguồn để chân một kiểu (GRU: không có track -> client tự
về stance; kho: --lock-lower ghim về khung đầu của TỪNG clip; vòng sống: không
track) và hips_pos/contacts mỗi clip một khác -> client phải dùng legIK sửa
chân theo contact, rồi "nhả điều chỉnh cũ" ở đầu câu sau: bàn chân nhảy ~6 cm
đúng khung đầu câu (theo dõi: chânP đỉnh 191–231 m/s², giật @0.0s, BE ✓).

Backend-first: nguồn giọng nói xuất chân GIỐNG HỆT NHAU và giống stance của
client (tư thế nghỉ của rig = baseRot phía client), hips_pos rỗng, contacts
rỗng -> legIK không có gì để sửa, không có gì để nhả. Clip kho (hành động có
chân: march, spin, jump...) GIỮ NGUYÊN chân đã dựng — không ghim nữa (hook
`ghim` cũ ghim cả kho là sai: giết chân của clip hành động).
"""
from __future__ import annotations

import math

import numpy as _np_ct
from app.modules.motion.bo_phan.co_so import (BoPhan, la_ardy_toan_than,
                                              quan_tinh_hook, trong_so_cua_so)
from app.modules.motion.hinh_hoc import _rig


def khong_ghim() -> bool:
    """MỘT CÔNG TẮC TẮT MỌI TẦNG GHIM CHÂN — `ONG_GHIM_CHAN=0`.

    User chốt 09/09 cho bản `talkinghead_backend` (cổng 8010): *"không khoá chân
    nữa, cả transformer và ARDY"* → *"tắt hết ghim"*. Chân do NGUỒN quyết định
    (transformer học từ mocap, ARDY sinh trọn thân), backend không kéo về đâu cả.

    Ba tầng cùng tắt bằng công tắc này vì chúng chồng lên nhau — tắt một tầng mà
    quên tầng kia thì chân vẫn bị giữ (đã dính đúng kiểu đó 27/08 với `dung_song`
    và `ong.ghim_chan_cuoi`):

        `ghim_contacts`   foot-lock theo sổ contacts (clip kho/idle/ARDY)
        `ghim_chan_cuoi`  giải lại IK cổ chân trên lưới gửi đi, MỌI cửa ra
        `tu_the_chuan`    (công tắc riêng `ONG_CHAN_CHUAN`) ghi đè chân về tư
                          thế nghỉ cho nguồn giọng nói — cái làm chân ĐỨNG IM
        `dung_song`       (công tắc riêng `ONG_DUNG_SONG`) IK ghim cổ chân tuyệt đối

    `backend/` 8002 không đặt biến này nên chạy y như cũ.
    """
    import os
    return os.getenv("ONG_GHIM_CHAN", "1") == "0"


XUONG_CHAN = ("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
              "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase")

chan = BoPhan(ten="chan",
              xuong=("LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"),
              diem_do=("LeftFoot", "RightFoot"),
              # chuan: nguồn giọng nói/vòng sống -> chân = tư thế nghỉ rig, bỏ hips_pos/contacts
              # CHỈ nguồn tuyệt đối (GRU). KHÔNG BAO GIỜ "song": vòng sống là
              # clip OFFSET (client cộng quanh stance) — ghi tư thế tuyệt đối vào
              # đó thì stance(−176°) + offset(−176°) ≈ +8°: chân lật ngược lên
              # khi nghỉ (hộp đen: bàn chân y=1,9 m, đỉnh 2800 m/s² mỗi lần đổi).
              mac_dinh={"chuan": 1, "nguon_chuan": ("gru",),
                       # ĐỨNG SỐNG (xem dung_song). 0 = tắt.
                       "dung_song_bien": 0.007, "dung_song_chu_ky": 9.0,
                       "dung_song_xoay": 6.6, "dung_song_vat": 0.45,
                       # vi chuyển động bàn chân (độ / giây chu kỳ) — xem dung_song
                       # 0 = bàn chân GHIM TUYỆT ĐỐI lúc nói. Từng để 2,2° rồi 1,0° cho khớp
                       # ω bàn chân người thật (4,2°/s) — user hai lần bắt đúng cái quét
                       # tuần hoàn đó bằng mắt ("chân trượt nhẹ qua lại"). Mắt thắng chỉ số:
                       # W bàn chân chịu "lệch nhẹ · đờ" (0,15) đổi lấy chân đứng chết chặt.
                       "chan_vi_bien": 0.0, "chan_vi_chu_ky": 3.4})



def _nguon_giong_noi(ctx: dict, t: dict) -> bool:
    ng = str(ctx.get("nguon") or "")
    if ng.startswith("song"):
        return False                       # vòng sống = OFFSET, cấm tuyệt đối
    return any(ng.startswith(p) for p in (t.get("nguon_chuan") or ("gru",)))


@chan.hook
def tu_the_chuan(clip, t, bp, ctx):
    if ctx.get("chi_do"):
        return
    import os
    if os.getenv("ONG_CHAN_CHUAN", "1") == "0" or khong_ghim():
        # công tắc A/B khi soi lỗi; `ONG_GHIM_CHAN=0` tắt luôn (xem khong_ghim).
        # ĐÂY MỚI LÀ CÁI LÀM CHÂN ĐỨNG IM: nó ghi đè MỌI khung của 6 xương chân
        # bằng MỘT bộ Euler hằng, nên đo trên trình duyệt ra bàn chân đúng
        # 0,127 m và gối đúng −4,3° suốt 17 giây liền sau khi lượt nói kết thúc.
        return
    if not t.get("chuan") or not _nguon_giong_noi(ctx, t):
        return
    # THÂN DƯỚI THUỘC VỀ CLIP: clip nào đã lái chân thì GRU không kéo về nghỉ
    # nữa — kể cả ngoài cửa sổ. Tư thế chân cuối cùng của clip được GIỮ.
    if clip.get("_cua_so_chan_clip"):
        ctx.setdefault("ghi", []).append("chân: thuộc về clip, GRU không đụng")
        return
    # KHÔNG có cử chỉ trong câu này: chân GIỮ tư thế bàn giao từ câu trước
    # (ctx["chan_giu"]) thay vì kéo về nghỉ rig. Người vừa làm cử chỉ xong thì
    # đứng nguyên thế đó, không "đứng thẳng lại" mỗi khi hết cử chỉ.
    giu = (ctx.get("chan_giu") or {})   # tư thế bàn giao, dung_song sẽ neo tiếp
    *_, nghi = _rig()
    b = clip.setdefault("bones", {})
    moc = sorted({f[0] for tr in b.values() for f in tr})
    if not moc:
        return
    # CỬA SỔ CHÂN-TỪ-CLIP: giữ nguyên chân của cử chỉ, không kéo về nghỉ
    cs_chan = clip.get("_cua_so_chan_clip") or []
    def _trong_cs(tm):
        return any(a - c <= tm <= b_ + d for a, b_, c, d in cs_chan)
    so = 0
    for ten in XUONG_CHAN:
        e = giu.get(ten) or nghi.get(ten)
        if e is None:
            continue
        ex, ey, ez = (round(float(v), 5) for v in e)
        tr = b.get(ten)
        if tr:
            for f in tr:
                if cs_chan and _trong_cs(f[0]):
                    continue
                f[1], f[2], f[3] = ex, ey, ez
        else:
            b[ten] = [[tm, ex, ey, ez] for tm in moc]     # thêm track: client không phải đoán
        so += 1
    if clip.get("hips_pos"):
        if not clip.get("_cua_so_toan_than"):
            clip["hips_pos"] = []          # có cửa sổ: hips_pos là NỘI DUNG cử chỉ
    if clip.get("contacts"):
        clip["contacts"] = {}              # chân luôn ghim -> dung_song khai lại
    ctx.setdefault("ghi", []).append(f"chân tư thế chuẩn ({so} xương), bỏ hips_pos/contacts")


def dung_song(clip, t, bp, ctx):
    """ĐỨNG SỐNG — dồn trọng tâm qua lại, BÀN CHÂN GHIM XUỐNG SÀN.

    Trước hook này, nguồn giọng nói có chân GHIM CỨNG về tư thế nghỉ
    (`tu_the_chuan`) — thân dưới đứng chết suốt buổi nói. Người thật không đứng
    như thế: trọng tâm dồn qua lại rất chậm, gối đổi bên trụ, mà BÀN CHÂN KHÔNG
    hề trượt trên sàn.

    Vì sao KHÔNG làm bằng cách xoay khớp chân trực tiếp: hông đứng yên (§8) +
    xoay khớp chân = bàn chân quét trên sàn, đúng thứ phải tránh. Cách đúng là
    làm NGƯỢC LẠI, như cơ thể thật:

        1. dời TRỌNG TÂM (hips_pos) ngang + hạ nhẹ, rất chậm
        2. GIỮ NGUYÊN đích cổ chân = vị trí world lúc nghỉ (bàn chân ghim)
        3. giải IK hai xương cho mỗi chân để cổ chân bám đúng đích

    Bàn chân vì thế đứng yên TUYỆT ĐỐI theo dựng hình, còn hông/gối chuyển động
    thật. Hông vẫn KHÔNG XOAY (chỉ tịnh tiến) nên §8 vẫn được tôn trọng.

    Chỉ chạy cho nguồn giọng nói; clip dựng sẵn đã có thân dưới của riêng nó.
    """
    _cs_chan_clip = clip.get("_cua_so_chan_clip") or []
    # THÂN DƯỚI THUỘC VỀ CLIP (user chốt 25/08): bỏ HẲN dồn trọng tâm thủ tục.
    # §8 dựng dung_song vì "người thật không đứng chết khi nói" — nay chuyển
    # động thân dưới đến từ clip Kimodo (thật, từ 700h mocap) nên tầng tự sinh
    # vừa thừa vừa cãi nhau với clip. Đánh đổi: giữa hai cử chỉ, thân dưới
    # ĐỨNG YÊN ở tư thế cử chỉ vừa xong (đúng ý "clip dừng đâu, giữ đó").
    import os as _os2
    if _os2.getenv("ONG_DUNG_SONG", "1") == "0" or khong_ghim():
        return
    if ctx.get("chi_do"):
        return
    if not _nguon_giong_noi(ctx, t):
        return
    bien = float(t.get("dung_song_bien") or 0.0)          # m, lệch ngang tối đa
    xoay = math.radians(float(t.get("dung_song_xoay") or 0.0))   # rad, hông xoay
    if bien <= 0 and xoay <= 0:
        return
    ck = float(t.get("dung_song_chu_ky") or 9.0)          # s, một chu kỳ dồn
    import numpy as np
    from app.modules.motion.hinh_hoc import _rig
    B, n2i, par, rr, rp, nghi = _rig()
    b = clip.get("bones") or {}
    moc = sorted({f[0] for tr in b.values() for f in tr})
    if len(moc) < 4:
        return
    CAP = (("LeftUpLeg", "LeftLeg", "LeftFoot"),
           ("RightUpLeg", "RightLeg", "RightFoot"))
    if any(x not in n2i for c in CAP for x in c):
        return

    def _the(hips_xyz, rot):
        """FK với gốc dời `hips_xyz`; trả hàm lấy vị trí world của một xương."""
        rp2 = dict(rp)
        rp2[n2i["Hips"]] = [rp[n2i["Hips"]][0] + hips_xyz[0],
                            rp[n2i["Hips"]][1] + hips_xyz[1],
                            rp[n2i["Hips"]][2] + hips_xyz[2]]
        cc = {}
        return lambda ten: np.array(B.world_transform(n2i[ten], par, rot, rp2, cc)[1]), rp2, cc

    # tư thế nghỉ của chân -> đích cổ chân (ghim)
    rot0 = dict(rr)
    lay0, rp0, cc0 = _the((0.0, 0.0, 0.0), rot0)
    dich = {c[2]: lay0(c[2]) for c in CAP}
    _dich_nghi = {k: np.asarray(v, float).copy() for k, v in dich.items()}
    # NEO VÀO VỊ TRÍ CHÂN CUỐI CLIP TRƯỚC (user chốt 25/08): dồn trọng tâm và
    # lắc hông của GRU vẫn chạy cho sinh động, nhưng bàn chân đứng ở CHỖ CỬ
    # CHỈ VỪA KẾT THÚC, không bật về stance nghỉ. Nhờ vậy chuỗi cử chỉ -> nói
    # -> cử chỉ liền mạch: mỗi đoạn bắt đầu từ đúng chỗ đoạn trước dừng.
    #
    # ĐÃ TẮT 27/08 — CƠ CHẾ NÀY CHÍNH LÀ CÚ TRƯỢT CHÂN LÚC NÓI.
    # User: "trang admin các clip thấy chuẩn mà realtime chân vẫn trượt vị trí".
    # Đích neo lấy từ khung CUỐI của clip đã TRỘN (GRU + cử chỉ), nên nó nhảy
    # loạn giữa các câu — đo bằng log mới: 3,6 -> 5,1 -> 6,2 -> 17,4 -> 0,0 cm.
    # Mỗi lần nhảy là bàn chân phải đi từ đích cũ sang đích mới trong lúc vẫn
    # chạm sàn: đo trên trình duyệt đúng các mốc đó ra 95 · 108 · 133 cm/s
    # (mốc chân ghim của người thật: 21,5 cm/s). Đích quá 12 cm bị chốt an toàn
    # BỎ QUA, nhưng bỏ qua cũng là một cú nhảy — về stance nghỉ.
    # Hộp đen backend không thấy gì vì nó đo TRONG LÒNG từng câu (chân 0,1-0,4);
    # cú trượt nằm ở CHỖ NỐI giữa hai câu.
    # Lý do 25/08 nay không còn: hồi đó clip kết thúc ở đâu cũng được; nay mọi
    # clip đã chuẩn hoá về stance (chan_troi <= 1,5 cm) nên "chỗ cử chỉ vừa kết
    # thúc" CHÍNH LÀ stance — neo thêm chỉ đẻ nhiễu.
    # TẮT MẶC ĐỊNH TỪ 27/08 — xem ghi chú dưới. Bật lại: profile
    # ong.chan.neo_chan_cuoi_clip = 1
    _bat_neo = int(((ctx.get("t_chung") or {}).get("chan", {})
                    or {}).get("neo_chan_cuoi_clip", 0) or 0)
    _neo = (ctx.get("chan_dich") or {}) if _bat_neo else {}
    for _c in CAP:
        _v = _neo.get(_c[2])
        if _v is None:
            continue
        _v = np.asarray(_v, float)
        # CHỐT AN TOÀN: đích quá xa stance nghỉ thì bỏ, dùng nghỉ. Đích lệch
        # nhiều (clip trước kết ở tư thế bất thường) làm IK không với tới ->
        # trả None -> chân giữ nguyên góc và bị hông kéo đi: đo trên trình
        # duyệt thấy p95 nhảy 0,6 -> 33 cm/s giữa các lượt. Ngưỡng 12 cm là
        # tầm dồn trọng tâm hợp lý; xa hơn là dữ liệu bất thường.
        if float(np.linalg.norm((_v - dich[_c[2]])[[0, 2]])) > 0.12:
            continue
        dich[_c[2]] = _v
    if _neo:
        # ĐO ĐỘ DỜI CỦA MỐC NEO: bàn chân trượt trong lúc nói là do đích ghim
        # NHẢY giữa hai câu, không phải do clip. Không đo được chỗ này thì hộp
        # đen chỉ thấy "chân 0.1" trong khi trên trình duyệt bàn chân đi
        # 128 cm/s (mốc chân ghim của người thật: 21,5).
        # so với TƯ THẾ NGHỈ (đã chụp trước khi vòng trên ghi đè `dich`)
        _do = {c[2]: float(np.linalg.norm((np.asarray(_neo[c[2]], float)
                                           - _dich_nghi[c[2]])[[0, 2]]) * 100)
               for c in CAP if c[2] in _neo}
        ctx.setdefault("ghi", []).append(
            "chân neo vào vị trí cuối clip trước ("
            + " · ".join(f"{k} {v:.1f}cm" for k, v in _do.items()) + ")")
    # HƯỚNG world của bàn chân lúc nghỉ. Ghim VỊ TRÍ cổ chân là chưa đủ: bàn
    # chân vẫn XOAY quanh cổ chân theo chuỗi chân, nên đế quét sàn. Đo ngày
    # 2026-08-23 trên 5 clip 30s: cổ chân đi 1,84-2,15cm mà MŨI chân đi
    # 2,87-3,36cm — gấp 1,7 lần. Nhìn ra đúng là "hai bàn chân đung đưa".
    huong_nghi = {c[2]: B.world_transform(n2i[c[2]], par, rot0, rp0, cc0)[0]
                  for c in CAP}

    # PHA TUYỆT ĐỐI xuyên lượt nói: mỗi câu là một clip riêng nên tm luôn chạy
    # lại từ 0 — trước đây trọng tâm khởi động lại pha mỗi câu (tầng nối che
    # được bậc thang, nhưng nhịp dồn trọng tâm không liên tục). Orchestrator
    # truyền tổng thời gian đã nói qua ctx["pha_goc"].
    pha_goc = float(ctx.get("pha_goc") or 0.0)
    # cửa sổ cử chỉ toàn thân: nhường phần trọng tâm + IK cho dữ liệu cử chỉ
    hp_cu = clip.get("hips_pos") or []
    tra_hips = {round(f[0], 4): f for f in (clip.get("bones", {}).get("Hips") or [])}
    import numpy as _np2
    hp_cu_a = _np2.asarray(hp_cu, float) if hp_cu else None
    hp = []
    for tm in moc:
        # HAI THÀNH PHẦN lệch chu kỳ, không phải một hình sin.
        # Một sin đơn ở chu kỳ đủ nhanh để đạt tốc độ khớp chân của người thật
        # sẽ nhìn ra nhịp máy đếm. Người thật vừa dồn trọng tâm CHẬM vừa chỉnh
        # vặt LIÊN TỤC. Tỉ số 1/2,7 là số vô tỉ xấp xỉ nên hai thành phần không
        # trùng pha lại trong thời lượng một câu (cùng cách song.py dùng f_lac
        # và f_lac2).
        pha = 2.0 * np.pi * (tm + pha_goc) / max(ck, 1e-6)
        pha2 = 2.0 * np.pi * (tm + pha_goc) / max(ck / 2.7, 1e-6)
        nhanh = float(t.get("dung_song_vat") or 0.45)      # phần chỉnh vặt
        w_ts = trong_so_cua_so(clip, tm)
        dx = bien * ((1.0 - nhanh) * np.sin(pha) + nhanh * np.sin(pha2 + 1.1))
        dy = -abs(bien) * 0.22 * (1.0 - np.cos(2.0 * pha)) * 0.5   # hạ nhẹ khi dồn
        # trong cửa sổ: trọng tâm = phần của CỬ CHỈ (đã merge vào hips_pos)
        # + phần dồn thủ tục thu theo (1-w)
        cx = cy = cz = 0.0
        if hp_cu_a is not None and len(hp_cu_a):
            cx = float(_np2.interp(tm, hp_cu_a[:, 0], hp_cu_a[:, 1]))
            cy = float(_np2.interp(tm, hp_cu_a[:, 0], hp_cu_a[:, 2]))
            cz = float(_np2.interp(tm, hp_cu_a[:, 0], hp_cu_a[:, 3]))
        dx = dx * (1.0 - w_ts) + cx
        dy = dy * (1.0 - w_ts) + cy
        hp.append([round(float(tm), 4), float(dx), float(dy), float(cz)])
        rot = dict(rr)
        if w_ts > 1e-3:
            # trong cửa sổ: IK phải giải với HIPS CỦA CỬ CHỈ (blend đã ghi vào
            # dữ liệu) — giải với hips nghỉ là chân khớp với một cái hông khác
            f_h = tra_hips.get(round(float(tm), 4)) if tra_hips else None
            if f_h is not None:
                rot[n2i["Hips"]] = B.q_from_euler_xyz(f_h[1], f_h[2], f_h[3])
        if xoay > 0:
            # HÔNG XOAY khi dồn trọng tâm. §8 cấm xoay Hips vì hồi đó chưa có
            # IK giữ chân — xoay là quét cả hai chân. Nay cổ chân đã ghim bằng
            # IK nên chân tự thích nghi, và người thật KHÔNG đứng cứng hông:
            # RPM talking đo được Hips biên độ 6,61° khi nói.
            gy = xoay * ((1.0 - nhanh) * math.sin(pha + 0.6)
                         + nhanh * math.sin(pha2 + 2.3))
            q = B.q_from_euler_xyz(0.0, gy, 0.0)
            rot[n2i["Hips"]] = B.q_mul(rr[n2i["Hips"]], q)
        # CỬA SỔ CHÂN-TỪ-CLIP: chân do cử chỉ lái, IK ghim nghỉ phải NHƯỜNG
        # (nếu không thì vừa cho chân qua ở blend lại bị kéo về ngay tại đây)
        if _cs_chan_clip:
            continue          # thân dưới thuộc về clip: IK/lắc thủ tục nghỉ hẳn
        lay, rp2, cc = _the((dx, dy, 0.0), rot)
        for hip, goi, co in CAP:
            S, E, H = lay(hip), lay(goi), lay(co)
            q_h = B.world_transform(n2i["Hips"], par, rot, rp2, cc)[0]
            kq = B.ik_hai_xuong(S, E, H, dich[co],
                                truc_on=B.q_rot(np.asarray(q_h, float),
                                                np.array([1.0, 0.0, 0.0])))
            if kq is None:
                continue
            for ten, q_them in ((hip, kq[0]), (goi, kq[1])):
                idx = n2i[ten]
                p_idx = par.get(idx)
                p_q = (B.world_transform(p_idx, par, rot, rp2, cc)[0]
                       if p_idx is not None else np.array([1.0, 0, 0, 0]))
                cu = B.q_mul(p_q, rot[idx]) if p_idx is not None else rot[idx]
                rot[idx] = B.q_mul(B.q_inv(p_q), B.q_mul(q_them, cu))
                cc.clear()
            # GHIM HƯỚNG BÀN CHÂN: đặt góc CỤC BỘ của Foot sao cho hướng WORLD
            # của nó bằng đúng hướng lúc nghỉ -> đế phẳng, mũi chân không quét.
            # Phải làm SAU khi IK đã xoay UpLeg/Leg, vì hướng cha vừa đổi.
            i_co = n2i[co]
            p_co = par.get(i_co)
            if p_co is not None:
                q_cha = B.world_transform(p_co, par, rot, rp2, cc)[0]
                # GHIM SỐNG, không ghim CHẾT. Bản đầu ép hướng bàn chân đúng
                # bằng hướng nghỉ mọi khung -> bàn chân ω p50 = 0,0 độ/s trong
                # khi người thật 4,2 (BEAT bvh thô, thước phân bố W tụt xuống
                # dưới "khớp"). Người đứng nói vẫn chỉnh bàn chân LI TI liên
                # tục. Thêm dao động nhỏ hai thành phần lệch chu kỳ (1/2,7 —
                # cùng thủ pháp dung_song) quanh hướng nghỉ; biên độ nhỏ nên
                # mũi chân chỉ dịch ~1cm, dưới hẳn mức "quét sàn" 3cm đã sửa.
                bien_c = math.radians(float(t.get("chan_vi_bien") or 0.0))
                if bien_c > 0:
                    ph_c = 2.0 * np.pi * tm / max(float(t.get("chan_vi_chu_ky") or 2.6), 0.2)
                    ben = 0.0 if co.startswith("Left") else 2.1
                    gx = bien_c * (0.6 * math.sin(ph_c + ben)
                                   + 0.4 * math.sin(ph_c * 2.7 + 1.9 + ben))
                    gz = bien_c * 0.7 * math.sin(ph_c * 1.37 + 0.8 + ben)
                    q_vi = B.q_mul(B.q_from_euler_xyz(gx, 0.0, 0.0),
                                   B.q_from_euler_xyz(0.0, 0.0, gz))
                    dich_q = B.q_mul(huong_nghi[co], q_vi)
                else:
                    dich_q = huong_nghi[co]
                rot[i_co] = B.q_mul(B.q_inv(q_cha), dich_q)
                cc.clear()
        for ten in (x for c in CAP for x in c[:3]) if xoay <= 0 else \
                   list(x for c in CAP for x in c[:3]) + ["Hips"]:
            tr = b.setdefault(ten, [])
            f = next((f for f in tr if abs(f[0] - tm) < 1e-9), None)
            q_moi = np.asarray(rot[n2i[ten]], float)
            if ten == "Hips" and f is not None and w_ts > 1e-3:
                # HIPS trong cửa sổ là NỘI DUNG cử chỉ (blend đã trộn) — sway
                # thủ tục nhường theo w. CHÂN thì luôn là của IK (không trộn):
                # chân clip giật vị trí + thiếu quán tính, user chấm rớt.
                qa = np.asarray(B.q_from_euler_xyz(f[1], f[2], f[3]), float)
                if float(qa @ q_moi) < 0:
                    q_moi = -q_moi
                q_moi = qa * w_ts + q_moi * (1.0 - w_ts)
                q_moi = q_moi / max(float(np.linalg.norm(q_moi)), 1e-9)
            e = B.q_to_euler_xyz(q_moi)
            if f is None:
                tr.append([round(float(tm), 4), *(float(v) for v in e)])
            else:
                f[1], f[2], f[3] = (float(v) for v in e)
    # KHAI BÁO TIẾP XÚC. `tu_the_chuan` xoá `contacts` cho nguồn giọng nói, nên
    # client KHÔNG chạy foot-lock IK (CLAUDE.md §9: contacts rỗng = không khoá
    # chân = trượt). Backend ghim cổ chân rất tốt trong FK của nó — đo được lệch
    # trung vị 0,0135cm suốt một câu — nhưng vẫn còn rung li ti: trượt p95
    # 5,2-5,8 cm/s so với mốc người NÓI 4,90. Không có lớp khoá phía client thì
    # cái rung đó hiện ra hết.
    # `dung_song` CHÍNH LÀ hook bảo đảm hai bàn chân đứng yên tuyệt đối, nên nó
    # phải là chỗ khai báo điều đó: cả hai chân chạm đất SUỐT clip.
    dai_clip = float(moc[-1]) if len(moc) else 0.0
    if dai_clip > 0:
        # chân LUÔN ghim (kiến trúc chân-IK 2026-08-23) -> chạm suốt clip
        clip["contacts"] = {c[2]: [[0.0, round(dai_clip, 3)]] for c in CAP}

    for ten in list(x for c in CAP for x in c[:3]) + (["Hips"] if xoay > 0 else []):
        if b.get(ten):
            b[ten].sort(key=lambda f: f[0])
    clip["hips_pos"] = hp
    ctx.setdefault("ghi", []).append(
        f"đứng sống: dồn trọng tâm ±{bien*100:.0f}cm chu kỳ {ck:.0f}s, cổ chân ghim")


@chan.hook
def hinh_the(clip, t, bp, ctx):
    """Chân dang ra theo hình thể (sau tư thế chuẩn -> chân chuẩn GRU cũng được dang)."""
    if ctx.get("chi_do"):
        return
    from app.modules.motion.bo_phan.hinh_the import ap_hinh_the
    ap_hinh_the(clip, ctx, ('LeftUpLeg', 'RightUpLeg'))




def ghim_contacts(clip, t, bp, ctx):
    """GHIM CHÂN THEO CONTACTS cho clip KHO/IDLE — foot-lock nay ở BACKEND.

    Vì sao: client legIK đã tắt (hợp đồng khung đa nền tảng — mọi nền tảng chỉ
    phát khung, không sửa dữ liệu). Nhưng clip RPM/HY mang phần trượt vốn có
    của dữ liệu retarget: đo 2026-08-23, IdleRpm03 bàn chân đi 18,5 cm/s NGAY
    TRONG pha contacts nói đang chạm đất — trước giờ legIK client che, tắt là
    lộ nguyên ("bàn chân cứ trôi nhẹ qua lại", user bắt bằng mắt).

    Cách làm — cùng cơ chế dung_song nhưng đích lấy TỪ CHÍNH CLIP:
    trong mỗi khoảng contacts [a,b] của một bàn chân, đích = vị trí world của
    cổ chân TẠI LÚC ĐẶT CHÂN (mốc a); giải IK hai xương kéo cổ chân về đích,
    ghim luôn hướng bàn chân như tại a. Trọng số vào/ra bằng smoothstep 0,15s
    ở mép khoảng để pha NHẤC CHÂN của clip còn nguyên. Ngoài khoảng contacts
    không đụng gì — chuyển động thật của clip giữ 100%.

    Chỉ chạy cho nguồn TUYỆT ĐỐI không phải giọng nói (giọng nói đã có
    dung_song; "song" là offset — cấm).
    """
    if ctx.get("chi_do") or khong_ghim():
        return
    ng = str(ctx.get("nguon") or "")
    if ng.startswith("song") or _nguon_giong_noi(ctx, t):
        return
    # ARDY TOÀN THÂN = DI CHUYỂN LÀ NỘI DUNG (user chốt 07/09): clip chạy vòng tròn dời gốc
    # 3,8 m, contacts của ARDY là khoảng GỘP phủ cả cú vung chân — ghim cổ chân về chỗ đặt
    # chân đầu khoảng là kéo chân đang vung về sàn (đúng cảnh báo §2 "foot-lock theo sổ
    # khoảng"). Đo: trượt lúc chống 0,14 m/s ở clip nguồn → 0,48 sau ống, và hệ số dời gốc
    # tối ưu đo lại trên đầu ra ống tụt về 0,5 (chân đã bị IK kéo lệch khỏi hông).
    if la_ardy_toan_than(clip, ctx):
        return
    tx = clip.get("contacts") or {}
    b = clip.get("bones") or {}
    if not tx or not b:
        return
    import numpy as np
    from app.modules.motion.hinh_hoc import _rig
    B, n2i, par, rr, rp, nghi = _rig()
    CAP = {"LeftFoot": ("LeftUpLeg", "LeftLeg", "LeftFoot"),
           "RightFoot": ("RightUpLeg", "RightLeg", "RightFoot")}
    if any(x not in n2i for c in CAP.values() for x in c):
        return
    # lưới chuẩn ở cửa vào ống: mọi track cùng mốc — lấy mốc từ track dài nhất
    mau = max(b.values(), key=len)
    moc = [f[0] for f in mau]
    if len(moc) < 4:
        return
    tra = {n: {round(f[0], 4): f for f in trk} for n, trk in b.items()}
    hp = clip.get("hips_pos") or []
    hp_a = np.asarray(hp, float) if hp else None

    def hips_tai(tm):
        if hp_a is None or not len(hp_a):
            return (0.0, 0.0, 0.0)
        x = np.interp(tm, hp_a[:, 0], hp_a[:, 1])
        y = np.interp(tm, hp_a[:, 0], hp_a[:, 2])
        z = np.interp(tm, hp_a[:, 0], hp_a[:, 3])
        return (float(x), float(y), float(z))

    def rot_tai(tm):
        rot = dict(rr)
        k = round(tm, 4)
        for n, bang in tra.items():
            f = bang.get(k)
            if f is not None and n in n2i:
                rot[n2i[n]] = B.q_from_euler_xyz(f[1], f[2], f[3])
        return rot

    def the_gioi(tm, rot, ten):
        d = hips_tai(tm)
        rp2 = dict(rp)
        rp2[n2i["Hips"]] = [rp[n2i["Hips"]][0] + d[0],
                            rp[n2i["Hips"]][1] + d[1],
                            rp[n2i["Hips"]][2] + d[2]]
        cc = {}
        q, pos = B.world_transform(n2i[ten], par, rot, rp2, cc)
        return np.asarray(q, float), np.asarray(pos, float), rp2

    def tron_q(qa, qb, w):
        """nlerp đủ dùng: góc pha trộn nhỏ (IK sửa vài độ)."""
        qa = np.asarray(qa, float); qb = np.asarray(qb, float)
        if float(np.dot(qa, qb)) < 0:
            qb = -qb
        q = qa * (1.0 - w) + qb * w
        n = float(np.linalg.norm(q))
        return q / n if n > 1e-9 else qa

    # Đo cả 0,15 lẫn 0,25 trên IdleRpm03 (clip trượt nặng nhất 18,5 cm/s):
    # 0,15 dư x1,23 · 0,25 dư x1,36 — nới mép không giúp vì cú kéo nằm GIỮA
    # khoảng chạm, không ở mép. Giữ 0,15; mức dư 1,23 một điểm nằm trong dải
    # dư đã chấp nhận của ống (1,0-1,3x — kẹp cứng hơn là méo dáng).
    MEP = 0.15
    # Chân cao hơn ĐIỂM ĐẶT quá mức này thì KHÔNG ghim: khoảng contacts là sổ
    # gộp, nó phủ cả những khung bàn chân đang rời sàn (CLAUDE.md §2 — foot-lock
    # phải theo DỮ LIỆU TỪNG KHUNG, không theo sổ khoảng).
    CAO_MO, CAO_TAT = 0.03, 0.10
    # Ghim mà làm bàn chân CAO LÊN quá mức này là ghim hỏng -> hoàn tác.
    XAU_DI = 0.01

    def _cao_nhat(ten):
        return max(float(the_gioi(tm, rot_tai(tm), ten)[1][1]) for tm in moc)

    for co, (hip, goi, _c) in CAP.items():
        khoang = tx.get(co) or []
        if not khoang:
            continue
        # BẢN SAO để hoàn tác: ghim là phép sửa CÓ THỂ HỎNG (IK kéo quá tầm ->
        # tư thế dị dạng). Đo trước/sau, xấu đi thì trả lại nguyên trạng.
        luu = {n: [list(f) for f in b[n]] for n in (hip, goi, co) if n in b}
        cao_truoc = _cao_nhat(co)

        dich = {}
        for a, bkt in khoang:
            # ĐÍCH = ĐIỂM VỪA ĐẶT CHÂN, tức khung bàn chân THẤP NHẤT trong
            # khoảng — KHÔNG phải khung mở đầu. Lấy khung mở đầu là ghim chân
            # vào đúng chỗ nó đang lơ lửng: đo 26/08 trên clip Kimodo, bàn chân
            # 0,33 -> 0,77 m (đích ở 0,385 mà IK kéo quá tầm thành dị dạng).
            trong = [tm for tm in moc if a - 1e-9 <= tm <= bkt + 1e-9] or [a]
            t_dat = min(trong, key=lambda tm: float(the_gioi(tm, rot_tai(tm), co)[1][1]))
            r_a = rot_tai(t_dat)
            q_a, p_a, _ = the_gioi(t_dat, r_a, co)
            dich[(a, bkt)] = (q_a, p_a)
        for tm in moc:
            w = 0.0
            cap = None
            for (a, bkt), d in dich.items():
                if a - 1e-9 <= tm <= bkt + 1e-9:
                    vao = min(1.0, max(0.0, (tm - a) / MEP))
                    ra_ = min(1.0, max(0.0, (bkt - tm) / MEP))
                    w = min(vao, ra_)
                    w = w * w * (3 - 2 * w)          # smoothstep
                    cap = d
                    break
            if cap is None or w <= 1e-4:
                continue
            q_dich, p_dich = cap
            rot = rot_tai(tm)
            _, p_now, rp2 = the_gioi(tm, rot, co)
            # CỔNG TỪNG KHUNG: bàn chân đang cao hơn điểm đặt thì nó đang rời
            # sàn — nhường, đừng kéo. Tắt mềm để không sinh điểm gãy (§2: mọi
            # ngưỡng chọn nhánh trong chuỗi thời gian đều thành điểm gãy).
            dz = float(p_now[1] - p_dich[1])
            if dz > CAO_TAT:
                continue
            if dz > CAO_MO:
                u = 1.0 - (dz - CAO_MO) / (CAO_TAT - CAO_MO)
                w *= u * u * (3 - 2 * u)
                if w <= 1e-4:
                    continue
            muon = p_now * (1.0 - w) + p_dich * w
            cc = {}
            S = np.asarray(B.world_transform(n2i[hip], par, rot, rp2, cc)[1], float)
            E = np.asarray(B.world_transform(n2i[goi], par, rot, rp2, cc)[1], float)
            q_h = B.world_transform(n2i["Hips"], par, rot, rp2, {})[0]
            kq = B.ik_hai_xuong(S, E, p_now, muon,
                                truc_on=B.q_rot(np.asarray(q_h, float),
                                                np.array([1.0, 0.0, 0.0])))
            if kq is None:
                continue
            cc = {}
            for ten, q_them in ((hip, kq[0]), (goi, kq[1])):
                idx = n2i[ten]
                p_idx = par.get(idx)
                p_q = (B.world_transform(p_idx, par, rot, rp2, cc)[0]
                       if p_idx is not None else np.array([1.0, 0, 0, 0]))
                cu_w = B.q_mul(p_q, rot[idx])
                rot[idx] = B.q_mul(B.q_inv(p_q), B.q_mul(q_them, cu_w))
                cc.clear()
            # hướng bàn chân: kéo VỀ hướng lúc đặt chân theo w
            i_co = n2i[co]
            p_co = par.get(i_co)
            if p_co is not None:
                q_cha = B.world_transform(p_co, par, rot, rp2, {})[0]
                q_muc = B.q_mul(B.q_inv(np.asarray(q_cha, float)), q_dich)
                rot[i_co] = tron_q(rot[i_co], q_muc, w)
            # ghi ngược ba xương vào clip tại mốc tm
            k = round(tm, 4)
            for ten in (hip, goi, co):
                f = tra.get(ten, {}).get(k)
                e = B.q_to_euler_xyz(np.asarray(rot[n2i[ten]], float))
                if f is not None:
                    f[1], f[2], f[3] = (float(v) for v in e)
                else:
                    trk = b.setdefault(ten, [])
                    trk.append([k, *(float(v) for v in e)])
                    trk.sort(key=lambda x: x[0])
                    tra.setdefault(ten, {})[k] = trk[-1]
        # LÀM MƯỢT TRƯỜNG HIỆU CHỈNH — cùng lý do như ghim_chan_cuoi: IK giải
        # từng khung độc lập nên nghiệm nhảy giữa hai khung kề. Đo 26/08 trên
        # clip NGƯỜI THẬT TalkingHappyOne: gối vào 92,5°/s, ra 549,1°/s.
        try:
            if len(luu) == 3 and all(len(b[n]) == len(v) for n, v in luu.items()):
                for n, khung in luu.items():
                    q_cu = _euler_xyz_sang_q_lo(_np_ct.asarray(khung, float)[:, 1:])
                    q_moi = _euler_xyz_sang_q_lo(_np_ct.asarray(b[n], float)[:, 1:])
                    e = _q_sang_euler_xyz_lo(_muot_hieu_chinh(q_cu, q_moi))
                    for i, f in enumerate(b[n]):
                        f[1], f[2], f[3] = float(e[i, 0]), float(e[i, 1]), float(e[i, 2])
                tra.update({n: {round(f[0], 4): f for f in b[n]} for n in luu})
        except Exception:
            pass

        cao_sau = _cao_nhat(co)
        if cao_sau > cao_truoc + XAU_DI:
            for n, khung in luu.items():
                for f_cu, f in zip(khung, b[n]):
                    f[1], f[2], f[3] = f_cu[1], f_cu[2], f_cu[3]
            tra.update({n: {round(f[0], 4): f for f in b[n]} for n in luu})
            ctx.setdefault("ghi", []).append(
                f"HOÀN TÁC ghim {co}: bàn chân {cao_truoc:.3f}->{cao_sau:.3f} m")

    ctx.setdefault("ghi", []).append("ghim chân theo contacts (foot-lock backend)")




# ---------- quaternion THEO LÔ (T,4), quy ước (w,x,y,z) như build_clip -------
def _q_mul_lo(a, b):
    """Nhân quaternion theo lô: (T,4) x (T,4) -> (T,4)."""
    import numpy as np
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=1)


def _q_inv_lo(q):
    import numpy as np
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _q_xoay_v_lo(q, v):
    """Xoay véc-tơ theo lô: q (T,4) áp lên v (3,) hoặc (T,3) -> (T,3)."""
    import numpy as np
    v = np.broadcast_to(np.asarray(v, float), (len(q), 3))
    u = q[:, 1:]
    w = q[:, :1]
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


def _euler_xyz_sang_q_lo(e):
    """(T,3) Euler XYZ NỘI TẠI (quy ước dự án — xem so_phan_bo §7.8) -> (T,4) wxyz."""
    from scipy.spatial.transform import Rotation as _R
    q = _R.from_euler("XYZ", e).as_quat()          # (x,y,z,w)
    import numpy as np
    return np.concatenate([q[:, 3:4], q[:, :3]], axis=1)


def _q_sang_euler_xyz_lo(q):
    from scipy.spatial.transform import Rotation as _R
    import numpy as np
    return _R.from_quat(np.concatenate([q[:, 1:], q[:, :1]], axis=1)).as_euler("XYZ")


# Nửa cửa sổ trung bình trượt cho trường hiệu chỉnh của IK ghim chân (khung).
# ±4 trên lưới 60 fps = ±67 ms: đủ dập nhiễu chọn-nghiệm, chưa đủ để làm trễ
# thấy được (nhịp chân người thật ~1 Hz).
MUOT_HIEU_CHINH = 4


def _muot_hieu_chinh(q_cu, q_moi):
    """Trung bình trượt phần HIỆU CHỈNH giữa tư thế cũ và tư thế IK vừa giải.

    Trả về mảng quaternion local đã áp hiệu chỉnh ĐÃ LÀM MƯỢT. Làm trên véc-tơ
    xoay (log map) của q_moi · q_cũ⁻¹ — hiệu chỉnh là góc nhỏ nên log map ổn
    định, không vướng bẫy Euler (§4 #1).
    """
    import numpy as np
    q_cu = np.asarray(q_cu, float)
    q_moi = np.asarray(q_moi, float)
    T = len(q_cu)
    if T < 3 or MUOT_HIEU_CHINH < 1:
        return q_moi
    d = _q_mul_lo(q_moi, _q_inv_lo(q_cu))
    d = d * np.sign(d[:, :1] + 1e-12)              # nửa cầu dương: hiệu chỉnh nhỏ
    w = np.clip(d[:, 0], -1.0, 1.0)
    goc = 2.0 * np.arccos(w)
    sn = np.sqrt(np.maximum(1.0 - w * w, 1e-16))
    rv = d[:, 1:] / sn[:, None] * goc[:, None]
    k = 2 * MUOT_HIEU_CHINH + 1
    pad = np.pad(rv, ((MUOT_HIEU_CHINH, MUOT_HIEU_CHINH), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    sm = np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(3)], axis=1)
    g = np.linalg.norm(sm, axis=1)
    nho = g < 1e-9
    ax = np.where(nho[:, None], 0.0, sm / np.where(g[:, None] < 1e-9, 1.0, g[:, None]))
    d2 = np.concatenate([np.cos(g / 2)[:, None], ax * np.sin(g / 2)[:, None]], axis=1)
    return _q_mul_lo(d2, q_cu)


def ghim_chan_cuoi(clip: dict, t: dict, dich_ngoai: dict | None = None) -> None:
    """GHIM LẠI CHÂN TRÊN LƯỚI CUỐI — bước CHỐT sau mọi warp/resample.

    Vì sao cần dù dung_song đã ghim: sau IK còn hai tầng đổi lưới — giảm tốc
    NÉN trục thời gian rồi luoi_chuan nội suy TỪNG XƯƠNG độc lập. Ràng buộc
    "cổ chân đứng yên" là hàm PHI TUYẾN của (Hips xoay, hips_pos, đùi, gối) —
    nội suy từng thành phần rồi ráp lại thì ràng buộc vỡ nhẹ đúng chỗ cong/nén.
    Chân LUÔN ghim về nghỉ (kiến trúc chân-IK 2026-08-23): cử chỉ chỉ điều
    khiển thân + hông, IK dùng Hips + hips_pos THẬT từng khung nên gối tự
    khuỵu theo cử chỉ.

    FK THEO LÔ (kiểm toán V-2): bản đầu gọi B.world_transform từng-khung-
    từng-xương qua dict + cache clear — 113ms/câu ngay trên đường realtime.
    Chuỗi chân chỉ là Hips -> đùi -> gối -> cổ chân: dựng cả chuỗi bằng
    quaternion lô numpy (đối chứng số học với bản cũ: lệch 0,00°), chỉ còn
    vòng lặp Python cho chính phép giải IK (toán vô hướng rẻ).
    """
    b = clip.get("bones") or {}
    hp = clip.get("hips_pos") or []
    if not b or not hp or khong_ghim():
        return
    import numpy as np
    from app.modules.motion.hinh_hoc import _rig
    B, n2i, par, rr, rp, nghi = _rig()
    CAP = (("LeftUpLeg", "LeftLeg", "LeftFoot"),
           ("RightUpLeg", "RightLeg", "RightFoot"))
    if any(x not in n2i or x not in b for c in CAP for x in c) or "Hips" not in b:
        return
    mau = b[CAP[0][0]]
    T = len(mau)
    moc = np.array([f[0] for f in mau], float)
    # LƯỚI PHẢI THẲNG HÀNG (sau luoi_chuan mọi track cùng mốc) — lệch thì thà
    # không ghim còn hơn ghim sai chỗ
    ten_can = ["Hips"] + [x for c in CAP for x in c]
    for ten in ten_can:
        tr = b[ten]
        if len(tr) != T or abs(tr[0][0] - moc[0]) > 1e-6 or abs(tr[-1][0] - moc[-1]) > 1e-6:
            return
    e_lo = {ten: np.asarray(b[ten], float)[:, 1:] for ten in ten_can}
    q_lo = {ten: _euler_xyz_sang_q_lo(e_lo[ten]) for ten in ten_can}
    hp_a = np.asarray(hp, float)
    dx = np.stack([np.interp(moc, hp_a[:, 0], hp_a[:, 1 + k]) for k in range(3)], axis=1)

    # đích nghỉ + hướng nghỉ (hằng). dich_ngoai: ghim về vị trí KHÁC tư thế
    # nghỉ — dùng cho clip ngoài (Kimodo) muốn giữ đúng chỗ bàn chân của CHÍNH
    # clip thay vì kéo về stance (§ retarget làm trượt 0,5 -> 17,6 cm/s).
    rot0 = dict(rr)
    cc0: dict = {}
    dich_nghi, huong_nghi = {}, {}
    for c in CAP:
        q, pos = B.world_transform(n2i[c[2]], par, rot0, rp, cc0)
        dich_nghi[c[2]] = np.asarray((dich_ngoai or {}).get(c[2], pos), float)
        huong_nghi[c[2]] = np.asarray(q, float)

    # FK LÔ: Hips là gốc -> world = local
    off_h = np.asarray(rp[n2i["Hips"]], float)
    qw_h = q_lo["Hips"]
    p_h = off_h + dx
    truc = _q_xoay_v_lo(qw_h, np.array([1.0, 0.0, 0.0]))
    ket: dict = {}
    for hip, goi, co in CAP:
        off_u = np.asarray(rp[n2i[hip]], float)
        off_l = np.asarray(rp[n2i[goi]], float)
        off_f = np.asarray(rp[n2i[co]], float)
        qw_u = _q_mul_lo(qw_h, q_lo[hip])
        p_u = p_h + _q_xoay_v_lo(qw_h, off_u)
        qw_l = _q_mul_lo(qw_u, q_lo[goi])
        p_l = p_u + _q_xoay_v_lo(qw_u, off_l)
        p_f = p_l + _q_xoay_v_lo(qw_l, off_f)
        u_moi = np.empty((T, 4)); l_moi = np.empty((T, 4)); f_moi = np.empty((T, 4))
        dich = dich_nghi[co]; huong = huong_nghi[co]
        for i in range(T):
            kq = B.ik_hai_xuong(p_u[i], p_l[i], p_f[i], dich, truc_on=truc[i])
            if kq is not None:
                q1 = np.asarray(kq[0], float); q2 = np.asarray(kq[1], float)
                qw_u2 = B.q_mul(q1, qw_u[i])
                qw_l2 = B.q_mul(q2, B.q_mul(qw_u2, q_lo[goi][i]))
                u_moi[i] = B.q_mul(B.q_inv(qw_h[i]), qw_u2)
                l_moi[i] = B.q_mul(B.q_inv(qw_u2), qw_l2)
            else:
                qw_l2 = qw_l[i]
                u_moi[i] = q_lo[hip][i]
                l_moi[i] = q_lo[goi][i]
            f_moi[i] = B.q_mul(B.q_inv(np.asarray(qw_l2, float)), huong)
        # LÀM MƯỢT TRƯỜNG HIỆU CHỈNH (§2). IK giải TỪNG KHUNG ĐỘC LẬP; gần
        # tư thế duỗi thẳng nghiệm rất nhạy, nên hai khung kề nhau có thể ra
        # hai nghiệm lệch nhau nhiều dù dữ liệu vào mượt. Đo 26/08: clip NGƯỜI
        # THẬT TalkingLove vào ống với gối 96°/s, ra khỏi ghim_chan_cuoi thành
        # 731°/s — cú giật do chính bước ghim sinh ra, và client làm mượt không
        # theo kịp nên bù trừ IK vỡ, bàn chân tụt xuống dưới sàn.
        # Làm mượt phần HIỆU CHỈNH (q_moi · q_cũ⁻¹) chứ KHÔNG làm mượt tư thế:
        # lọc tư thế thì mất luôn nội dung chuyển động, lọc hiệu chỉnh thì chỉ
        # bỏ cái nhiễu mà bước ghim vừa thêm vào.
        for ten, q_m in ((hip, u_moi), (goi, l_moi), (co, f_moi)):
            ket[ten] = _q_sang_euler_xyz_lo(_muot_hieu_chinh(q_lo[ten], q_m))
    for ten, e_m in ket.items():
        tr = b[ten]
        for i in range(T):
            tr[i][1], tr[i][2], tr[i][3] = (float(e_m[i, 0]), float(e_m[i, 1]),
                                            float(e_m[i, 2]))


# QUÁN TÍNH đăng ký CUỐI: phải chạy SAU các hook sửa tư thế, để nó làm mềm
# quỹ đạo CUỐI CÙNG chứ không phải quỹ đạo giữa chừng.
chan.hook(quan_tinh_hook(2.4))   # chân: chống rung khi retarget, vẫn giữ bám sàn


# ĐỨNG SỐNG đăng ký SAU CÙNG — sau cả quán tính. IK phải là tiếng nói CUỐI thì
# cổ chân mới ghim TUYỆT ĐỐI; để quán tính chạy sau thì nó lọc lại góc chân vừa
# giải và bàn chân trôi (đo được 2,39 cm ngang). Bản thân dung_song đã sinh
# chuyển động mượt sẵn (hình sin + IK) nên không cần làm mềm thêm.
chan.hook(dung_song)
chan.hook(ghim_contacts)
