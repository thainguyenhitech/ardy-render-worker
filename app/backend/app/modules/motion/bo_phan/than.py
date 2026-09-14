"""THÂN (cột sống) — khoá/uốn tư thế + trần vận tốc góc.

Khoá ở CLIP ĐÃ DỰNG chứ không ở đầu ra mô hình: tại khớp nguồn lưng chỉ dao
động 1,9°, sau chuyển hệ sang rig thành 15,1° — khâu chuyển hệ khuếch đại 8x.
"""
from __future__ import annotations

import math

from app.modules.motion.bo_phan.co_so import BoPhan, quan_tinh_hook, yeu_cau_gian
from app.modules.motion.bo_phan.do_bo_phan import toc_goc
from app.modules.motion.cau_hinh import XUONG_LUNG
from app.modules.motion.hinh_hoc import _rig

than = BoPhan(
    ten="than", xuong=tuple(XUONG_LUNG), diem_do=("Spine2",),
    mac_dinh={"khoa": 1, "uon_do": 0.0, "gap_toi_da": 0.0,
              # khoa=1 nay là CHUẨN DÁNG, không phải đóng băng (xem _chuan_dang)
              "khom_bo": 1.0,        # 1 = bỏ hết độ khom hằng, 0 = giữ nguyên
              "dao_dong_he": 0.5,    # xem _chuan_dang — chọn theo trần lưng hiện có
              "toc_goc_tran": 1.9, "gia_toc_goc_tran": 11.0})


def _hop_nhat_khoa_cu(t, t_chung):
    """Khoá cũ phẳng (lung_khoa, lung_uon_do, lung_gap_toi_da) vẫn được tôn
    trọng — profile đã viết theo khoá cũ không phải sửa."""
    if "lung_khoa" in t_chung and "khoa" not in (t_chung.get("than") or {}):
        t["khoa"] = t_chung["lung_khoa"]
    if "lung_uon_do" in t_chung and "uon_do" not in (t_chung.get("than") or {}):
        t["uon_do"] = t_chung["lung_uon_do"]
    if "lung_gap_toi_da" in t_chung and "gap_toi_da" not in (t_chung.get("than") or {}):
        t["gap_toi_da"] = t_chung["lung_gap_toi_da"]
    return t


@than.hook
def khoa_uon(clip, t, bp, ctx):
    # KHÔNG BAO GIỜ cho nguồn "song": vòng sống là clip OFFSET (client cộng
    # quanh stance), còn hook này ghi góc TUYỆT ĐỐI (tư thế nghỉ) vào từng đốt.
    # Ghi tuyệt đối vào offset = XOÁ SẠCH phần lắc: đo được song_vong ra tới
    # client có Spine/Spine1 biên độ 0,00° dù song.sinh_vong có sinh lắc lưng
    # (Spine1 x 1,15° thở · Spine y 1,43° lắc). Cộng với việc clip talking cũng
    # bị khoá về 0,00°, lưng nhân vật CỨNG TUYỆT ĐỐI mọi lúc.
    # Cùng loại bẫy mà chan.py và hinh_the.py đã có chốt chặn (CLAUDE.md §8).
    if str(ctx.get("nguon") or "").startswith("song"):
        return
    # CLIP DỰNG SẴN: TƯ THẾ LƯNG LÀ NỘI DUNG, KHÔNG PHẢI ARTEFACT.
    # `_chuan_dang` kéo TRUNG BÌNH của cột sống về tư thế nghỉ. Đúng cho nguồn
    # GIỌNG NÓI (GRU cho ra lưng khom, cái khom đó là độ lệch hằng vô nghĩa),
    # nhưng SAI cho clip kho: ở đó độ cong lưng chính là cử chỉ, và nó còn có
    # vai trò BÙ LẠI độ nghiêng của hông để thân trên đứng thẳng. Xoá nó đi thì
    # hông nghiêng bao nhiêu, cả thân trên đổ theo bấy nhiêu như một khối cứng.
    # Đo 26/08 trên ZKBeatOpenHandSweep: biên độ nghiêng bên 4,7° -> 11,2°
    # (gấp 2,4 lần) ngay tại hook này — user thấy là "cả thân người nghiêng
    # bất thường". Cùng bài học mà `hong.kep_goc` đã ghi: "clip DỰNG SẴN thì
    # độ xoay hông CHÍNH LÀ nội dung và chân đã được dựng khớp với nó".
    if str(ctx.get("nguon") or "").startswith("kho"):
        return
    # LƯỢT HAI của lam_mem: hook SỬA TƯ THẾ phải bỏ qua (CLAUDE.md §10). Khoá cũ
    # ghi thẳng giá trị tuyệt đối nên áp hai lần vẫn ra một kết quả, không cần
    # chốt này. _chuan_dang thì THU theo tỉ lệ nên áp kép là bình phương hệ số:
    # đo được 0,125^2 -> biên độ 18,27 do còn 0,29 thay vì 2,28.
    if ctx.get("chi_do"):
        return
    t = _hop_nhat_khoa_cu(t, ctx["t_chung"])
    _, _, _, _, _, nghi = _rig()
    b = clip.get("bones") or {}
    # DẤU (02/09, user "đầu hơi ngẩng, cảm giác ngã người"): FK trên rig cho
    # +x của Spine* = GẬP TRƯỚC (đầu dịch +z về phía camera 17 cm khi 3 đốt
    # +10°). Bảng motion_types viết theo NGHĨA "khom (−) / ưỡn-thẳng (+)":
    # elderly −9 · gloomy −6 · shy −4 · heavy −4 · confident +3 — bản cũ cộng
    # thẳng vào trục nên già/buồn/ngại lại ƯỠN RA SAU 4–15° và _bu_co gập cổ
    # +18° bù hướng nhìn -> cằm hếch: trục đầu −9,2° trong khi idle RPM +5,6°.
    # Đổi dấu ở ĐÂY (một chỗ), giữ nguyên bảng: uon_do < 0 = khom về trước.
    uon = -math.radians(float(t.get("uon_do") or 0.0))
    kep = math.radians(float(t.get("gap_toi_da") or 0.0))
    dot = [x for x in bp.xuong if x in b and x in nghi]
    if not dot:
        return
    moi_dot = uon / len(dot)          # chia đều, dồn một đốt thì lưng gãy khúc
    # GIỮ HƯỚNG NHÌN: sửa lưng làm ĐỔI HƯỚNG ĐẦU trong không gian thế giới, vì
    # đầu treo ở cuối chuỗi cột sống. Đo trên 6 clip (2026-08-22): ống nắn hướng
    # nhìn từ -13,3 độ (cúi gằm) tới +22,9 độ (ngước lên trời) so với clip gốc.
    # Sửa LƯNG là chuyện tư thế, KHÔNG được đụng tới ÁNH MẮT: ghi lại xoay cột
    # sống trước/sau rồi bù đúng phần chênh vào Neck.
    truoc = ({ten: [list(f) for f in b[ten]] for ten in dot}
             if t.get("khoa") else None)
    if t.get("khoa"):
        _chuan_dang(b, dot, nghi, moi_dot, t, clip=clip)
    elif kep > 0:
        for ten in dot:
            gx, _, _ = nghi[ten]
            for f in b[ten]:
                f[1] = max(gx - kep, min(gx + kep, f[1] + moi_dot))
    if truoc and b.get("Neck"):
        _bu_co(b, dot, truoc)


def _chuan_dang(b, dot, nghi, moi_dot, t, clip=None):
    # CHUẨN DÁNG, KHÔNG ĐÓNG BĂNG.
    #
    # Bản cũ ghi thẳng tư thế nghỉ vào mọi khung -> biên độ lưng 0,00 độ trên
    # MỌI nguồn (đo: TalkingOne 18,27 -> 0,00; GestureBig 13,25 -> 0,00; cả
    # mocap BeatV1 18,28 -> 0,00). Lưng cứng như tượng suốt lúc nói.
    #
    # Vấn đề THẬT không phải chuyển động mà là TƯ THẾ: GRU cho ra lưng KHOM, và
    # cái khom đó là một ĐỘ LỆCH HẰNG so với tư thế nghỉ. Tách hai phần ra:
    #
    #   - TRUNG BÌNH của track = tư thế (cái khom)  -> kéo về nghỉ, hệ số khom_bo
    #   - DAO ĐỘNG quanh trung bình = chuyển động   -> GIỮ, thu theo dao_dong_he
    #
    # dao_dong_he MẶC ĐỊNH 1,0 — GIỮ NGUYÊN dao động của nguồn.
    #
    # Từng để 1/8 theo docstring đầu tệp ("khớp nguồn 1,9 do, sang rig 15,1 do,
    # khuếch đại 8 lan"). ĐO LẠI thì lập luận đó sai chỗ: mocap NGƯỜI THẬT trên
    # CHÍNH RIG NÀY có biên độ lưng 18,2-20,0 do (BeatV1/V2/Test01, cửa sổ 6s).
    # Tức 18 do CHÍNH LÀ biên độ lưng người thật ở cách tham số hoá của rig,
    # không phải nhiễu do chuyển hệ. Để 1/8 thì lưng chỉ động 2-4 do = 1/5 người
    # thật, nhìn vẫn đơ.
    #
    # Cái xấu là KHOM (độ lệch hằng) — khom_bo đã lo.
    #
    # CHỌN 0,5 BẰNG ĐO, với ĐÚNG thước: quét lần đầu dùng nghiem_thu.cham_diem
    # (giật/vận tốc BÀN TAY) nên mù với trần góc của chính lưng, tưởng 1,0 vô
    # hại. Đo lại bằng theo_doi.bao_cao (trần theo BỘ PHẬN) trên 60 clip:
    #
    #   he     lưng biên độ TV   ti_le TV   câu lưng LỌT trần
    #   0,125       3,08 do        0,07          0/60
    #   0,25        6,15 do        0,13          0/60
    #   0,50       11,85 do        0,33          5/60
    #   0,70       14,13 do        1,09         33/60
    #   1,00       15,68 do        1,76         58/60
    #
    # Mocap người thật trên chính rig này: 18,2-20,0 do. 0,5 cho ~60% mức đó mà
    # chỉ 8% câu lọt trần — ngang tỉ lệ lọt của tay vốn đã chấp nhận (13%).
    #
    # GHI CHÚ TRẦN: than.gia_toc_goc_tran = 11,59 rad/s2, THẤP HƠN cả Spine
    # người thật (p99.9 15,9 · max 17,3) và thấp hơn nhiều so với quy ước
    # "tran = 1,5x max nguoi that" (~26) ở CLAUDE.md §7. Trần này chưa bao giờ
    # lộ ra vì lưng vốn bị đóng băng 0,00 do nên không bao giờ chạm tới nó.
    # CHƯA nới — nới trần để số của mình đi lọt là đúng thứ §7 cấm. Muốn lưng
    # động bằng người thật thì phải hiệu chuẩn lại trần bằng mocap trước.
    #
    # KHÔNG KẸP độ lệch. Bản đầu có kẹp `lech_toi_da` 8 do, và nó chính là thứ
    # làm lưng "vào nhanh ra nhanh": kẹp từng khung ĐỘC LẬP nên đúng lúc quỹ đạo
    # chạm ngưỡng thì đạo hàm GÃY. Đo giật góc lưng của TalkingOne:
    #     kẹp 8 do   -> p95  47,6 · max 1340,6 rad/s3
    #     bỏ kẹp     -> p95  41,4 · max  123,0
    #     người thật -> p95 72-201 · max 307-896
    # Tức cú kẹp tạo ra đỉnh giật gấp 1,5 lần đỉnh người thật, xen giữa một nền
    # vốn đã mượt. Đúng thứ CLAUDE.md §2 cấm: KHÔNG kẹp cứng để chữa vọt.
    # Phần vọt đã có tran_toc_do + lam_mem lo, và hai cái đó chữa bằng GIÃN TRỤC
    # THỜI GIAN chứ không cắt biên độ.
    import numpy as np
    from app.modules.motion.bo_phan.co_so import trong_so_cua_so
    khom_bo = float(t.get("khom_bo", 1.0))
    he = float(t.get("dao_dong_he", 0.85))
    for ten in dot:
        tr = b.get(ten)
        if not tr:
            continue
        a = np.array(tr, dtype=float)
        goc = a[:, 1:]
        g0 = np.array(nghi[ten], dtype=float) + np.array([moi_dot, 0.0, 0.0])
        tb = goc.mean(axis=0)
        # HE THEO KHUNG: trong cửa sổ cử chỉ toàn thân, dao động lưng là NỘI
        # DUNG của cử chỉ (cúi, xoay người) — thu 0,25 là bào mất 75% cử chỉ.
        # he tiến về 1,0 theo trọng số cửa sổ, mượt vì trọng số smoothstep.
        if clip is not None and clip.get("_cua_so_toan_than"):
            w = np.array([trong_so_cua_so(clip, float(x[0])) for x in tr])
            he_k = (he + (1.0 - he) * w)[:, None]
        else:
            he_k = he
        # tư thế: kéo trung bình về nghỉ · chuyển động: giữ dao động quanh nó
        moi = g0 + (tb - g0) * (1.0 - khom_bo) + (goc - tb) * he_k
        for i, f in enumerate(tr):
            f[1], f[2], f[3] = float(moi[i, 0]), float(moi[i, 1]), float(moi[i, 2])


def _nhan_q(a, c):
    import numpy as np
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)


def _bu_co(b, dot, truoc):
    # Bù vào Neck đúng phần xoay mà khoá lưng vừa lấy đi, để ĐẦU giữ nguyên
    # hướng thế giới: Neck' = (sau)^-1 . truoc . Neck. Chỉ bù HƯỚNG — vị trí
    # đầu vẫn dịch theo lưng, không bù được và cũng không cần.
    import numpy as np
    from app.modules.motion.hinh_hoc import _euler_q_vec, _q_euler_vec

    moc = np.array([f[0] for f in b["Neck"]], dtype=float)

    def tich(nguon):
        q = None
        for ten in dot:
            tr = np.array(nguon[ten], dtype=float)
            e = np.stack([np.interp(moc, tr[:, 0], tr[:, k]) for k in (1, 2, 3)],
                         axis=1)
            qi = _q_euler_vec(e[:, 0], e[:, 1], e[:, 2])
            q = qi if q is None else _nhan_q(q, qi)
        return q

    q_truoc = tich(truoc)
    q_sau = tich({ten: b[ten] for ten in dot})
    if q_truoc is None or q_sau is None:
        return
    nghich = np.concatenate([q_sau[:, :1], -q_sau[:, 1:]], axis=1)
    delta = _nhan_q(nghich, q_truoc)
    tr = np.array(b["Neck"], dtype=float)
    q_co = _q_euler_vec(tr[:, 1], tr[:, 2], tr[:, 3])
    e = np.unwrap(_euler_q_vec(_nhan_q(delta, q_co)), axis=0)
    for i, f in enumerate(b["Neck"]):
        f[1], f[2], f[3] = float(e[i, 0]), float(e[i, 1]), float(e[i, 2])


than.hook(quan_tinh_hook(2.2))   # thân nặng: trôi sau nhiều


@than.hook
def tran_toc_do(clip, t, bp, ctx):
    t = _hop_nhat_khoa_cu(t, ctx["t_chung"])
    # KHÔNG return khi khoa nữa. Chú thích cũ ("đã khoá cứng thì không có gì để
    # đo") đúng hồi khoa=1 nghĩa là ĐÓNG BĂNG. Nay khoa=1 là CHUẨN DÁNG, lưng
    # vẫn động theo nguồn — bỏ qua ở đây thì trần vận tốc/gia tốc của lưng KHÔNG
    # BAO GIỜ được áp, lưng vào nhanh ra nhanh, nhìn dứt khoát và cứng.
    g = toc_goc(clip, bp.xuong)
    k = 1.0
    tr = float(t.get("toc_goc_tran") or 0)
    if tr > 0 and g["v95"] > tr:
        k = max(k, g["v95"] / tr)
    tr = float(t.get("gia_toc_goc_tran") or 0)
    if tr > 0 and g["a95"] > tr:
        k = max(k, math.sqrt(g["a95"] / tr))
    if k > 1.0:
        yeu_cau_gian(ctx, bp, f"thân {g['v95']:.2f} rad/s", k)
