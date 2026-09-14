"""CLIP CÓ SỐNG KHÔNG — một định nghĩa duy nhất, dùng chung.

## Đính chính (09/09) — đọc trước

Bản đầu của tệp này tuyên bố "hộp bao chấm 0,73 cho clip đi 0,03 m, nên cổng cũ mù và clip chết
đi thẳng xuống client". **SAI.** Hộp bao là đường chéo bao của quỹ đạo, mà đường chéo bao không
bao giờ lớn hơn ĐỘ DÀI quỹ đạo — đo 8 clip: tỉ lệ hộp bao/quãng đi 0,46–0,86, không ca nào >1.
Hộp bao chỉ có thể ĐÁNH THẤP, không bao giờ thổi phồng.

Con số 0,73 vs 0,03 là lỗi của chính kịch bản đo: nó so hộp bao lấy MAX TRÊN BA XƯƠNG với quãng
đi của RIÊNG bàn tay phải. Hai xương khác nhau. Đúng bẫy §4 bẫy 6 — thước mới phải chạy hai
đường độc lập trước khi tin, mà tôi đã không làm.

Hệ quả: cổng sinh-lại cũ (`hộp bao < 0,08 m`) **có nổ** ở những ca clip chết; các clip chết tôi
đo được nằm ngoài đường sản xuất vì kịch bản gọi thẳng `/motion`, bỏ qua `ardy_client._sinh`.

## Vậy tệp này còn để làm gì

Bốn lý do THẬT, đã đo:

1. **CHIA CHO THỜI LƯỢNG.** Cổng "lát tĩnh" so một LÁT 1,5 s với NỀN cả câu 4 s. Quãng thô của
   đoạn ngắn đương nhiên nhỏ hơn nên cổng thiên vị sẵn; `toc_tb` (m/s) mới là cùng đại lượng.
2. **`pct_dong` mang thông tin hộp bao KHÔNG có.** Một cú quét rồi đứng im và một chuỗi động
   liên tục có thể cùng hộp bao; tỉ lệ khung có tốc độ tách được hai thứ đó.
3. **MỘT định nghĩa thay hai bản gần giống nhau** — `ardy_client._bien_do` và
   `transformer_provider._quang_tay` cùng đo hộp bao, khác nhau ở chỗ trả max hay tuple.
4. **Ngưỡng có gốc.** 0,08 m/s là mốc bàn tay người đứng nói (mocap BEAT, §7 CLAUDE.md), thay
   cho hằng số 0,08 m không ai giải thích được đến từ đâu.

Đo cổng cũ vs mới trên 6 prompt: cũ nổ 1/6, mới nổ 2/6 — ca thêm là "waves arms slow motion",
hộp bao 0,100 (vừa lọt ngưỡng 0,08) nhưng chỉ 0,063 m/s và 22 % khung có tốc độ. Cải thiện có
thật nhưng KHIÊM TỐN, không phải bước ngoặt như bản đầu viết.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

XUONG = ("RightHand", "LeftHand", "Head")
V_DONG = float(os.getenv("ARDY_SONG_V", "0.10"))      # m/s — coi là "đang động" ở một khung
TOC_SAN = float(os.getenv("ARDY_SONG_TOC", "0.08"))   # m/s — mốc người thật BEAT lúc nói
PCT_SAN = float(os.getenv("ARDY_SONG_PCT", "5.0"))    # % khung tối thiểu có tốc độ
# ĐỘNG TÁC CHỈ CÓ ĐẦU (12/09, user "gật đầu / nhìn lên không thấy motion"): thước tay/đầu đo QUÃNG ĐI của
# khớp Head — khớp đó nằm ở chân sọ, gật 37° chỉ dời ~2 cm nên nod/shake/look bị chấm CHẾT, sinh lại tới
# 6 s rồi bỏ (đo 12/09: 'nods the head' 5 lần thử lại 13 s, 'looks up'/'turns head right' bỏ hẳn) trong
# khi thô lẫn rig đều gật 37°. Nay đo thêm GÓC LỆCH đỉnh đầu (Neck→HeadTop_End) so khung đầu; qua sàn này
# là sống dù tay đứng yên. Sàn 12°: gật/lắc thật 25–40°, rung nền của clip đứng yên ≤ 5°.
DAU_SAN = float(os.getenv("ARDY_SONG_DAU_DO", "12.0"))  # độ


def theo_xuong(clip: dict) -> dict:
    """{tên xương: {quang_di, toc_tb, pct_dong, v95}} cho ba xương chỉ báo. Một lượt FK duy nhất.

    `toc_tb` = quãng đi / thời lượng, đơn vị m/s. CHIA CHO THỜI LƯỢNG là điều bắt buộc khi đem
    so hai đoạn khác độ dài: cổng "lát tĩnh" so một LÁT 1,5 s với NỀN cả câu 4 s, mà quãng đi
    thô của đoạn ngắn đương nhiên nhỏ hơn — so bằng tốc độ mới công bằng.
    """
    ra: dict[str, dict] = {}
    try:
        import numpy as np
        from app.modules.motion.bo_phan.do_bo_phan import vi_tri_world
    except Exception:  # noqa: BLE001
        logger.warning("ardy_song: không nạp được thước", exc_info=True)
        return ra
    dai_clip = float(clip.get("duration_s") or 0.0)
    for ten in XUONG:
        try:
            t, p = vi_tri_world(clip, (ten,))
        except Exception:  # noqa: BLE001
            # KÊU LÊN: FK hỏng cho MỌI xương thì `do()` trả 0 → `song()` False → clip bị coi là
            # chết và sinh lại, trong khi lỗi nằm ở THƯỚC chứ không ở clip.
            logger.warning("ardy_song: FK hỏng ở %s — chỉ số của xương này bị bỏ", ten,
                           exc_info=True)
            continue
        p = np.asarray(p).reshape(len(t), -1)[:, :3]
        if len(p) < 3:
            continue
        tt = np.asarray(t, float)
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        # vận tốc theo MỐC THẬT của track, không giả định 60 fps (§4 bẫy 2: đừng so hai lưới)
        dt = np.diff(tt)
        dt[dt <= 1e-9] = 1.0 / 60.0
        v = d / dt
        dai = dai_clip if dai_clip > 1e-6 else float(tt[-1] - tt[0]) or 1.0
        qd = float(d.sum())
        ra[ten] = {"quang_di": qd, "toc_tb": qd / max(dai, 1e-6),
                   "pct_dong": float((v > V_DONG).mean() * 100.0),
                   "v95": float(np.percentile(v, 95)), "dai_s": dai}
    return ra


def dau_goc(clip: dict) -> float:
    """Góc lệch LỚN NHẤT (độ) của đỉnh đầu (Neck→HeadTop_End) so với khung đầu — thước cho động tác chỉ
    có đầu (gật, lắc, ngước, quay). 0 khi FK không có hai điểm đó."""
    try:
        import numpy as np
        from app.modules.motion.bo_phan.do_bo_phan import vi_tri_world
        t, p = vi_tri_world(clip, ("Neck", "HeadTop_End"))
        p = np.asarray(p)
        if p.ndim != 3 or p.shape[1] < 2 or len(t) < 3:
            return 0.0
        v = p[:, 1, :] - p[:, 0, :]
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(v @ v[0], -1.0, 1.0))).max())
    except Exception:  # noqa: BLE001
        logger.warning("ardy_song: không đo được góc đầu", exc_info=True)
        return 0.0


def do(clip: dict) -> dict:
    """Chấm một clip ĐÃ RETARGET (định dạng kho). Trả số đo, KHÔNG kết luận.

    Lấy xương ĐỘNG NHẤT trong ba (hai bàn tay + đầu) rồi báo theo xương đó — báo trung bình cả
    ba thì một tay đứng yên kéo tụt điểm của cử chỉ một tay, mà cử chỉ một tay là đa số.
    Kèm `dau_goc` (độ) cho động tác chỉ có đầu — xem `DAU_SAN`.
    """
    ra = {"xuong": None, "quang_di": 0.0, "toc_tb": 0.0, "pct_dong": 0.0, "v95": 0.0,
          "dai_s": float(clip.get("duration_s") or 0.0)}
    for ten, d in theo_xuong(clip).items():
        if d["quang_di"] > ra["quang_di"]:
            ra = {"xuong": ten, **d}
    ra["dau_goc"] = dau_goc(clip)
    return ra


def toc_tay(clip: dict) -> float:
    """Tốc độ trung bình của bàn tay ĐỘNG HƠN (m/s) — dùng cho cổng so LÁT với NỀN.

    Chỉ hai bàn tay, không tính đầu: cổng đó quyết định lát ARDY có thay được phần TAY của nền
    hay không; đầu vẫn do `dau_song` lo nên đưa vào là so nhầm đại lượng.
    """
    d = theo_xuong(clip)
    return max((d[t]["toc_tb"] for t in ("RightHand", "LeftHand") if t in d), default=0.0)


def song(clip: dict) -> bool:
    """Có động tác thật hay không. Tay/đầu-dời-chỗ: phải đạt CẢ HAI điều kiện tốc độ + % khung động
    (chỉ tốc độ trung bình thì clip rung tại chỗ vẫn qua; chỉ % khung động thì nhúc nhích 1 cm cũng
    qua). HOẶC đầu xoay/gật đủ góc (`DAU_SAN`) — động tác chỉ có đầu không dời khớp Head được mấy."""
    d = do(clip)
    return (d["toc_tb"] >= TOC_SAN and d["pct_dong"] >= PCT_SAN) or d.get("dau_goc", 0.0) >= DAU_SAN


def mo_ta(clip: dict) -> str:
    """Một dòng cho log/báo cáo."""
    d = do(clip)
    s = (d["toc_tb"] >= TOC_SAN and d["pct_dong"] >= PCT_SAN) or d.get("dau_goc", 0.0) >= DAU_SAN
    return ("%s %.2f m / %.1f s = %.3f m/s · %.0f %% khung động · v95 %.2f · đầu %.0f° · %s"
            % (d["xuong"] or "?", d["quang_di"], d["dai_s"], d["toc_tb"], d["pct_dong"],
               d["v95"], d.get("dau_goc", 0.0), "SỐNG" if s else "CHẾT"))
