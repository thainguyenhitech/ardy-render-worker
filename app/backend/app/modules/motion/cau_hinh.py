"""CẤU HÌNH ống xử lý motion — mọi tham số mặc định và kho hệ số per-clip.

Tham số theo type nằm ở app/personality/motion_types.json (profile.ong);
tệp này chỉ giữ MẶC ĐỊNH và cách gộp. Xem docs/tech/ong-xu-ly-motion.md.
"""
from __future__ import annotations

from pathlib import Path



# rig đích — nạp một lần để lấy tư thế nghỉ và làm FK đo tốc độ
# ong.py nằm ở backend/app/modules/motion/ -> lùi 3 cấp mới tới backend/
_BACKEND = Path(__file__).resolve().parents[3]

_GLB = _BACKEND.parent / "characters" / "mira.glb"

_BUILD = _BACKEND / "luong_clip" / "b1_render"     # 29/08: scripts/motion_pipeline -> luong_clip/b1_render

XUONG_LUNG = ("Spine", "Spine1", "Spine2", "Spine3")


# LƯỚI CHUẨN. Mọi clip vào ống được đưa về lưới đều FPS_CHUAN ngay cửa vào
# (Catmull-Rom trên véc-tơ xoay); mọi tầng, mọi phép đo, mọi ngưỡng và HUD đều
# trên đúng lưới này; fps gửi xuống client luôn = FPS_CHUAN. Không còn "GRU 20,
# kho 30, ống 60, HUD 60 nhưng ngưỡng hiệu chuẩn ở 30".
FPS_CHUAN = 60.0

MAC_DINH = {
    # --- tầng 1: hình thể ---
    # HÌNH THỂ MẶC ĐỊNH THEO NGUỒN — cộng vào body_fit của người dùng. Kimodo
    # dựng cánh tay DANG 41,7° (p50, 40 clip, đo góc Arm->ForeArm so trục
    # thân) trong khi người thật 18-25° (BEAT 21,0/18,3 · IdleRpm 24,7/25,1)
    # — lỗi cả kho, không phải hình thể từng nhân vật (user 28/08: "clip
    # motion vẫn có vai mở rộng"). Quét: −10 -> p50 34,1°, dị dạng 0/40;
    # −15 -> 30,7° nhưng 3/40 tay chìm ngực; −20 -> 7/40. Lấy −10.
    # KHÉP VAI PHẢI KÈM MỞ KHUỶU: tay "buông" Kimodo dang 42° với khuỷu chĩa
    # ra + bàn tay chĩa vào, khép vai một mình là bàn tay đâm vào đùi (7 clip
    # 0,3 -> 8,5 cm). Quét khuỷu 0/+6/+10/+14 với vai −10: +10 là 7 -> 0 lỗi,
    # 24 clip ngẫu nhiên 0 lỗi. Hook áp phần nguồn này THEO TRỌNG SỐ góc dang
    # (smoothstep 30->46°, hinh_the.py) nên tay đang buông thật không bị đụng.
    # IdleRpm/BEAT (người thật) KHÔNG có mục ở đây — clip['nguon'] quyết.
    # CHUẨN NGƯỜI HƠI ỐM (user chốt 28/08): kho chuẩn về ốm, nhân vật to hơn
    # chỉ còn MỞ ra — mở không bao giờ sinh xuyên (đo +5/+10: 0/60), khép mới có.
    # Quét 6 cấu hình x 60 clip đều 0 hỏng nhờ ghép vai+khuỷu+trọng số:
    #   −10/+10 dải 30-46 w1,0 : buông p10 32,9 · p50 36,4 · p90 86,1 (cũ)
    #   −15/+15 dải 18-34 w1,5 : p10 19,6 · p50 26,5 · p90 75,5  <- người 18-25°
    #   −12/+12 dải 18-34 w2,2 : p10 23,1 · p50 27,8 · p90 72,1
    # _w_max: trọng số tiếp tục tăng quá dải — tay dang càng xa thân khép càng
    # mạnh (user: "cánh tay dang xa thân thì khuỷu->bàn tay không tự nhiên").
    # 29/08: BỎ khuỷu +15. Sau khi clip đã chuẩn bản lề (tools/chuan_clip_kimodo)
    # thì chỉ khép vai −15 đã 0 xuyên đùi; "bù khuỷu" trước đây chữa hậu quả của
    # khuỷu lệch mặt phẳng, và bản thân nó (quay ForeArm x) lại bẻ khuỷu ra
    # ngoài bản lề (cổng: 137/318 trượt khuyu_lech_phang). Xem docs §7.17.
    # 02/09: dòng GRU (nền lúc nói — thứ user nhìn nhiều nhất) trước đây KHÔNG
    # có nhãn nguồn nên không được khép: đo FK 3 câu thật khuỷu ngoài vai
    # 14,6–19,5 cm (người thật IdleRpm 3,5–4,5), lệch từng câu (tay P 26 vs T
    # 11 cm) — user: "tay trái vẫn không khép sát như tay phải". Cùng dải với
    # kimodo; trọng số theo góc dang TỪNG BÊN nên tự cân hai tay.
    "than_hinh_nguon": {"kimodo": {"vai_mo_do": -15.0,
                                   "_dang_tu": 18.0, "_dang_toi": 34.0, "_w_max": 1.5},
                        "gru": {"vai_mo_do": -15.0,
                                "_dang_tu": 18.0, "_dang_toi": 34.0, "_w_max": 1.5}},
    # w_max 2,0 thử trên TOÀN KHO: 309/318 — 9 clip biên (đùi 8,0-8,5 · thân
    # 10,8-11) ở cử chỉ nhịp tay dang xa; 1,5: xem kết quả cổng bên dưới.
    "lung_khoa": 1,          # ghim cột sống về tư thế nghỉ
    "lung_uon_do": 0.0,      # + ưỡn ngực, − cong lưng (người già)
    "lung_gap_toi_da": 0.0,  # >0: không khoá mà chỉ KẸP góc gập (độ)
    # --- tầng 2: lọc ---
    "loc": 0,                # 1 = bật One-Euro
    "loc_cat_toi_thieu": 1.0,
    "loc_beta": 0.02,
    # --- tầng 3: tốc độ ---
    "toc_san_tay": 0.10,     # m/s — dưới mức này là đứng đơ, nén thời gian lại
    # TRẦN GIA TỐC hiệu chuẩn từ 30 clip kho đang dùng tốt: trung vị 10,3 ·
    # p90 28,2 · tối đa 33,6 m/s². Đặt 12 như tôi bịa lúc đầu thì clip nào cũng
    # chạm trần và HỆ SỐ RIÊNG bị nuốt sạch — đặt 1,4x mà thời lượng chỉ nhích
    # 2,93 -> 2,91s. Lấy p90 cộng biên.
    # CHUẨN NHỊP — nén ĐỘ TẢN giữa các clip trong cùng một type.
    # Trần/sàn chỉ chặn ĐỈNH; cái mắt người đọc ra là "lúc nhanh lúc chậm" lại
    # là PHÂN BỐ NHỊP GIỮA CÁC CLIP, mà băng [0,10 .. 2,60] rộng 26 lần nên
    # không định hình gì (kho Kimodo max 2,17 — chưa bao giờ chạm trần).
    # Đo 2026-08-28, `_toc_tay` p95 mỗi clip, cùng một thước:
    #   người thật BEAT   0,61 · 0,71 · 0,80 m/s  -> TẢN p90/p10 = 1,32x
    #   Kimodo qua ống    0,65 · 1,18 · 2,02      -> TẢN 3,13x, trung vị nhanh 1,66x
    # NÉN MỀM chứ không kẹp: v' = moc x (v/moc)^nen. Kẹp cứng ở trần đẩy cả kho
    # dồn đúng một giá trị — chính là "tầng kẹp thành ràng buộc trùm" (CLAUDE.md
    # §3); nén mềm giữ nguyên thứ tự nhanh/chậm giữa các clip (vẫy tay VẪN nhanh
    # hơn suy tư) mà kéo hai đuôi lại gần nhau.
    # CHỈ cho nguồn được giãn thời gian (kho clip). GRU khớp tiếng thì không.
    # NEO Ở TRUNG VỊ CỦA CHÍNH KHO, KHÔNG ở trung vị người thật. BEAT chỉ chứa
    # cử chỉ NHỊP khi trò chuyện — không có động tác nhấn mạnh (giơ hai tay lên
    # đầu, đẩy ra, vung dứt khoát), nên ép trung vị kho về 0,71 là hiệu chuẩn
    # vào nguồn KHÔNG CHỨA HÀNH VI ĐÓ (CLAUDE.md §3, đã sập ba lần: xoắn cẳng
    # tay, cổ tay, thân dưới). Đuôi nhanh 2,0 m/s vẫn nằm trong tầm người (với
    # tới / vung tay đo lâm sàng 1-3 m/s) — nó KHÔNG sai, chỉ LỆCH so với phần
    # còn lại. Nên tầng này chỉ NÉN ĐỘ TẢN: clip quanh trung vị không đổi
    # (giãn <2% thì bỏ qua), chỉ cái đuôi nhanh bị kéo lại.
    "nhip_moc": 1.15,        # m/s — trung vị v95 của kho Kimodo qua ống (adult)
    "nhip_nen": 0.35,        # 0 = ép hết về mốc; 1 = tắt hẳn
    "nhip_gian_max": 1.60,   # trần hệ số giãn, chặn clip nhanh dị thường
    # SÀN = 1,0: CHỈ nén phía NHANH, không bao giờ đẩy clip chậm nhanh lên.
    # Đặt 0,85 thì clip idle (v95 0,09-0,59 — vốn PHẢI chậm) bị tăng 18%, sai
    # hướng hẳn. Và đuôi chậm của kho (p10 = 0,65) vốn đã trùng người thật
    # (BEAT p10 = 0,61) — không có gì để sửa ở đó.
    "nhip_gian_min": 1.00,
    # --- tầng 4: nối ---
    "noi_t_min": 0.25, "noi_t_max": 1.20,
    "noi_tay_toi_da": 2.6,   # m/s BÀN TAY ở mối ghép — cái mắt thật sự thấy
    "noi_gia_toc_toi_da": 24.0,  # m/s² BÀN TAY xuyên vùng hoà = 1,5x đỉnh người thật (BEAT max 16,4); cũ 30
    "noi_quet_khung": 5,     # số khung quét tìm điểm cắt tối ưu ở đầu B
    # GIAO VẬN TỐC CHO CLIP SAU. Đo profile qua mối nối thấy nhân vật DỪNG HẲN
    # BA LẦN: A phanh về 0, cầu tăng rồi lại phanh về 0, B khởi động lại từ 0.
    # Vì tiếp tuyến cuối của cầu lấy đúng vận tốc khung đầu B, mà build_clip đã
    # bung giãn đầu clip (S_IN=1.7) nên số đó bằng 0.
    # Cách chữa: cắt bỏ đoạn lấy đà của B, cho cầu giao thẳng vận tốc vào chỗ B
    # đã chạy thật, và DỜI B SỚM LÊN đúng bằng đoạn vừa cắt để cử chỉ vẫn rơi
    # trúng từ như cũ.
    "noi_giao_toc": 0.5,     # giao bao nhiêu phần vận tốc hành trình của B
    "noi_cat_da_toi_da": 0.5,  # giây, trần đoạn lấy đà được phép cắt
    # CẦU NGẮN, NẰM SÁT B. Trải cầu trọn khoảng trống 1,15s là sai: để đến B ở
    # 2 m/s thì tay phải đi 2,3 mét, nên đường Hermite buộc phải phanh lại và
    # ta mất đúng cái đang cần. Người thật cũng vậy — giữ tư thế rồi mới lấy đà
    # vào cử chỉ sau. Phần trống trước cầu để client giữ tư thế (lớp sống vẫn
    # thở nên không chết cứng).
    # TRẦN THỜI LƯỢNG CẦU. Hồi dùng Hermite phải để 0,45s vì cầu dài thì đường
    # cong lượn giữa chừng. Cầu định hình vận tốc không lượn — thời lượng suy
    # thẳng từ quãng đường nên dài bao nhiêu là vừa đúng bấy nhiêu. Chặn ở
    # 0,45 khiến cầu phải chạy nhanh hơn vật lý cho phép và đến nơi vọt quá
    # (đo được 2,29 trong khi clip sau chỉ chạy 1,39).
    "noi_cau_toi_da": 0.90,   # giây
    # --- tầng 5: ghim chân ---
    "hong_xoay_toi_da": 0.0,  # độ; 0 = không giới hạn
    # --- tầng 6: dày khung ---
    # Client dựng 60 hình/giây trong khi clip chỉ 16-30. Nội suy TUYẾN TÍNH
    # giữa hai khung thưa tạo một "góc" tại mỗi khung gốc: đo ở 60fps ra gia
    # tốc p95 = 196,6 m/s² trong khi mocap thật ~17. Dày khung bằng nội suy
    # TRƠN ngay ở backend thì client chỉ còn việc đọc.
    "day_khung_fps": 60.0,   # = FPS_CHUAN; 0 = tắt (chỉ để thử)
    "day_khung_toi_da": 8.0, # trần số lần nhân khung, chặn clip thưa bất thường
}

# HIỆU CHUẨN: ghi đè mặc định nguồn bằng JSON trong THAN_HINH_NGUON để chạy
# cổng toàn kho với nhiều ứng viên song song (không sửa mã giữa các lần chạy).
import json as _json_th, os as _os_th
if _os_th.environ.get("THAN_HINH_NGUON"):
    MAC_DINH["than_hinh_nguon"] = _json_th.loads(_os_th.environ["THAN_HINH_NGUON"])

# ---- HỆ SỐ TỐC ĐỘ RIÊNG TỪNG CLIP ----------------------------------------
# Để ở MỘT tệp chung thay vì ghi vào từng clip: admin sửa một chỗ, không phải
# viết lại 600 tệp clip, và clip render mới không mất hệ số đã duyệt.
_HE_SO = _BACKEND / "app" / "motion_clips" / "_toc_do.json"

_he_so_cache: dict | None = None

_he_so_mtime: float = 0.0

def he_so_clip(clip_id: str) -> float:
    """Hệ số tốc độ riêng của clip, mặc định 1.0. Tự nạp lại khi tệp đổi để
    admin sửa xong thấy ngay, khỏi khởi động lại."""
    global _he_so_cache, _he_so_mtime
    try:
        m = _HE_SO.stat().st_mtime
    except OSError:
        return 1.0
    if _he_so_cache is None or m != _he_so_mtime:
        import json as _j
        try:
            _he_so_cache = _j.loads(_HE_SO.read_text(encoding="utf-8"))
            _he_so_mtime = m
        except Exception:
            _he_so_cache = {}
    goc = clip_id[:-2] if len(clip_id) > 2 and clip_id[-2] == "_" \
        and clip_id[-1] in "ABCDE" else clip_id
    v = _he_so_cache.get(clip_id, _he_so_cache.get(goc, 1.0))
    try:
        return max(0.4, min(2.5, float(v)))
    except (TypeError, ValueError):
        return 1.0

def tham_so(profile: dict | None) -> dict:
    """Gộp tham số của type lên mặc định."""
    t = dict(MAC_DINH)
    t.update((profile or {}).get("ong") or {})
    return t
