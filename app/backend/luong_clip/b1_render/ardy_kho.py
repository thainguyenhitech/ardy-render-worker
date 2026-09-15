"""RENDER KHO CLIP ARDY SẴN — mọi clip mở màn từ IDLE và kết ở IDLE (12/09 tối, user chốt).

    .venv/bin/python luong_clip/b1_render/ardy_kho.py --nhom day --bien-the 3        # 47 tag dạy LLM × 3 biến thể
    .venv/bin/python luong_clip/b1_render/ardy_kho.py --nhom bang                    # cả bảng ardy_prompts.json (426 tag)
    .venv/bin/python luong_clip/b1_render/ardy_kho.py --tags wave_hello,bow_greeting --lam-lai

Vì sao (user 12/09): ARDY sinh sống 0,7× thời gian thực trên MỘT CPU, mỗi cue 3,9 s, nhiều phiên là nghẽn; clip
sinh sống cắt ở trần LLM nên kết giữa động tác (36/49 clip cache kết cách tư thế đầu >15 cm) và mối nối clip↔clip
phải hoà 47–157°. Kho render sẵn: ARDY chỉ là công cụ render offline, runtime chỉ nạp clip (`loi/kho_ardy.py`).

LUẬT DỰNG MỘT CLIP (đo trên 12 động tác, xem CLAUDE.md §13 "KHO CLIP ARDY SẴN"):
  1. IDLE ĐÓNG BĂNG: `_idle_tho.json` = khung thô của 'A person stands still.' sinh MỘT LẦN — hai lần sinh cùng câu
     cho hai tư thế lệch nhau 18° ở cánh tay, nên cả kho phải dùng đúng một tệp làm lịch sử (history) lẫn đích.
  2. ĐỘNG TÁC: sinh `--giay` (6/7/8 s theo biến thể) với history = idle → khung 0 trùng idle (lệch 1°).
  3. CẮT ĐỘNG TÁC: quãng NGHỈ đầu tiên SAU ĐỈNH đầu tiên (khớp ≤ NGHI_DO so idle, chậm, không quay, không đi, giữ
     ≥ GIU_NGHI_S) — đúng một lần thực hiện (ARDY lặp lại động tác nếu để dài). Không nghỉ (động tác GIỮ: khoanh
     tay, vỗ tay, vươn vai) thì cắt ở đỉnh + GIU_S. Tư thế giống idle GIỮA động tác (ngắn hơn GIU_NGHI_S, hoặc
     trước đỉnh) bị bác — đếm vào `cua_so_bac` để rà.
  4. LỐI RA: nối chuỗi từ khung cắt cue 'stands still with arms relaxed at the sides' + target_pose = khung idle
     (ràng buộc mềm khung cuối, cfg_constraint 3, 3 s). Cue chữ đơn thuần KHÔNG về idle với động tác chân và nhiều
     động tác giữ; có target thì 12/12 về, lắng sau 0,2–2,2 s, dư 4–9°. Cắt lối ra ở quãng lắng đầu tiên.
  5. TRỘN 0,5 s cuối về ĐÚNG khung idle cho THÂN TRÊN + xoay hông (giữ yaw); CHÂN KHÔNG ĐỤNG (user: chỉ khoá chân ở
     idle và nền transformer; clip thì chân tự do, lệch chân ở mối nối đã có helper bước của ống). Mối nối cuối A →
     đầu B mọi cặp: 0,33° trung vị, 0,46° max.
  6. Cắt phần đứng chờ đầu clip (ARDY chờ 0,4–4,5 s mới bắt đầu), giữ DAU_S trước onset.
  7. Cổng: `ardy_song.song` (có động tác thật) + đỉnh giữ ≥ 60 % bản chưa cắt + dài ≤ DAI_TOI_DA. Trượt → `kho_loi.jsonl`.
Kho nằm NGOÀI git: `ARDY_KHO` (mặc định ~/project/motion_trainer/ardy_kho), chỉ mục `kho.json` (tag → [clip]).
`_kho_toan_than` = clip có DỜI CHỖ/QUAY theo nội dung (ống coi chân là nội dung), không phụ thuộc `:tt` của LLM.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_BE = Path(__file__).resolve().parents[2]
_TH = _BE.parent / "talkinghead_backend"
for p in (str(_BE), str(_TH)):
    if p not in sys.path:
        sys.path.insert(0, p)

KHO = Path(os.getenv("ARDY_KHO", Path.home() / "project/motion_trainer/ardy_kho"))
# FPS GỐC CỦA ARDY (15/09, user: "làm lại chuẩn theo tài liệu chính thức của ARDY, không cần 60fps"): model card + bài báo —
# ARDY-Core-RP huấn luyện 20 fps; `scripts/generate.py` xuất đúng `model.motion_rep.fps`, KHÔNG nâng fps, khử trượt một lần
# trên cả chuỗi. Service nối thẳng 20 → 60 làm quỹ đạo gãy mỗi 3 khung (phải vá bằng spline) và lối ra khởi động từ khung
# nội suy. Kho render ở 20 fps; runtime 8010 nâng lên lưới 60 fps bằng Catmull-Rom lúc NẠP (`loi/kho_ardy`). `ARDY_KHO_FPS=60`
# = đường cũ.
FPS_ARDY = 20        # fps SINH của ARDY-Core-RP — luôn xin service đúng fps gốc (không để service nội suy tuyến tính)
# fps LƯU KHO / retarget / IK. 60 (mặc định): tool nâng 20 → 60 bằng spline C² TRONG KHÔNG GIAN ARDY (`_nang_tho`) rồi
# retarget + IK + nối trên ĐÚNG lưới client phát. Lưu kho 20 fps (ARDY_KHO_FPS=20) đo 15/09: IK chỉ ghim bàn chân tại khung
# gốc, runtime nội suy từng xương độc lập giữa hai khung → trôi pha đứng xoay trái 1,34 · quay người 1,27 · nhảy lò cò
# 2,05 cm (bản 60 fps 0,5–0,8) và gối duỗi quá tay giữa hai khung.
FPS = int(os.getenv("ARDY_KHO_FPS", "60"))
CAU_IDLE = "A person stands still."
CAU_LOI_RA = "A person stands still with arms relaxed at the sides."
LAM_TRON = os.getenv("ARDY_KHO_LAM_TRON", "1") == "1"   # Catmull-Rom giữa các nút 20 fps sau retarget (xem lam_tron_nut)
GIAY_BIEN_THE = (6.0, 7.0, 8.0)      # mỗi biến thể xin một độ dài khác (ARDY KHÔNG tất định — đo 14/09 TB 10,5°; tool ardy_render gieo seed)
GIAY_LOI_RA = 3.0
CFG_LOI_RA = 3.0
NGHI_DO = 15.0                       # khớp (không ngón) cách idle ≤ ngần này = "gần idle"
NGHI_V = 90.0                        # °/s
NGHI_QUAY = 30.0                     # °/s hướng quay
NGHI_DI = 0.10                       # m/s hông
GIU_NGHI_S = 0.4                     # gần idle phải GIỮ ngần này mới là nghỉ (tư thế trùng giữa động tác thì ngắn hơn)
GIU_S = 1.0                          # động tác giữ: cắt ở đỉnh + ngần này
LANG_DO, LANG_V, LANG_S = 12.0, 60.0, 0.3
TRON_S = 0.5
DAU_S = 0.2
NGUONG_DAU = 5.0                     # khung mở màn phải cách idle ≤ ngần này (xem `diem_dau`)
# KIỂM ĐỊNH KỲ (13/09, user: "kiểm tra định kỳ để tránh sai sót nhiều"): mỗi `KIEM_MOI` clip chấm lại lô vừa render
# NGAY TRONG vòng render (đọc meta, không mở lại tệp) và TỰ DỪNG nếu lệch chuẩn — §10b "việc nền phải tự dừng khi
# hỏng liên tiếp", thay vì chạy ba ngày rồi mới biết cả lô sai như lô đầu (đầu clip lệch idle 8° trung vị).
KIEM_MOI = int(os.getenv("ARDY_KHO_KIEM_MOI", "200"))
KIEM_DAU_MAX = 6.0                   # khung đầu xa idle nhất (luật `diem_dau` bảo đảm ≤ NGUONG_DAU)
KIEM_CUOI_MAX = 3.0                  # khung cuối THÂN TRÊN xa idle nhất (trộn 0,5 s ép về 0; chân là nội dung)
KIEM_LOI_MAX = 0.40                  # tỉ lệ trượt cổng tối đa của một lô
HOAT_ONSET_MIN = 10.0                # onset = độ hoạt động vượt max(ngần này, 35 % đỉnh)
DAI_TOI_DA = 12.0
NGON = ("Thumb", "Index", "Middle", "Ring", "Pinky")
CHAN = ("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase")


# ---- hình học --------------------------------------------------------------------------------
def _q(e):
    from loi.goc_troi import _euler_sang_q
    return np.asarray(_euler_sang_q(*e), float)


def _ang(a, b) -> float:
    return math.degrees(2.0 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def _yaw(e) -> float:
    from loi.goc_troi import _euler_sang_q, _yaw_cua
    return _yaw_cua(_euler_sang_q(*e))


def _hips_khong_yaw(e):
    from loi.goc_troi import _euler_sang_q, _q_nhan, _q_yaw, _yaw_cua
    qq = _euler_sang_q(*e)
    return np.asarray(_q_nhan(_q_yaw(-_yaw_cua(qq)), qq), float)


def tu_the(clip: dict, k: int) -> dict:
    """Tư thế khung k: quaternion từng xương (bỏ ngón), Hips bỏ yaw."""
    return {x: (_hips_khong_yaw(tr[k][1:4]) if x == "Hips" else _q(tr[k][1:4]))
            for x, tr in clip["bones"].items() if len(tr) > k and not any(n in x for n in NGON)}


def lech(pa: dict, pb: dict) -> tuple[float, str]:
    return max(((_ang(pa[x], pb[x]), x) for x in pa if x in pb), default=(0.0, ""))


def van_toc(clip: dict, k: int) -> float:
    if k == 0:
        return 0.0
    return max((_ang(_q(tr[k][1:4]), _q(tr[k - 1][1:4])) for x, tr in clip["bones"].items()
                if x != "Hips" and len(tr) > k and not any(n in x for n in NGON)), default=0.0) * FPS


def do_hoat_dong(clip: dict, idle_pose: dict) -> dict:
    """Các dãy theo khung: dd (lệch idle °), vv (°/s), yw (hướng quay so khung 0, °), vy (°/s), vh (m/s), hoat."""
    n = len(clip["bones"]["Hips"])
    H = clip["bones"]["Hips"]
    HP = clip.get("hips_pos") or [[k / FPS, 0.0, 0.0, 0.0] for k in range(n)]
    dd = np.array([lech(tu_the(clip, k), idle_pose)[0] for k in range(n)])
    vv = np.array([van_toc(clip, k) for k in range(n)])
    y0 = _yaw(H[0][1:4])
    yw = np.degrees(np.unwrap(np.array([_yaw(H[k][1:4]) - y0 for k in range(n)])))
    vy = np.abs(np.gradient(yw)) * FPS if n > 1 else np.zeros(n)
    xz = np.array([[float(HP[k][1]), float(HP[k][3])] for k in range(n)])
    vh = np.linalg.norm(np.gradient(xz, axis=0), axis=1) * FPS if n > 1 else np.zeros(n)
    hoat = dd + 0.5 * np.abs(yw) + 100.0 * np.linalg.norm(xz - xz[0], axis=1)
    return {"n": n, "dd": dd, "vv": vv, "yw": yw, "vy": vy, "vh": vh, "hoat": hoat, "xz": xz}


def diem_cat_dong_tac(clip: dict, idle_pose: dict, giu_s: float = GIU_S) -> dict:
    """Luật 3. Trả {k_cat, loai, onset, dinh1, cua_so_bac, lap_sau}."""
    m = do_hoat_dong(clip, idle_pose)
    n, dd, vv, hoat = m["n"], m["dd"], m["vv"], m["hoat"]
    # onset TƯƠNG ĐỐI theo đỉnh: gật đầu 15°, thả vai 20° không bao giờ vượt 30° tuyệt đối (lô 1: nod_yes, shake_head_no,
    # relief_exhale_settle trượt "không có động tác" oan); còn sống hay không đã có cổng `ardy_song` (kể cả góc đầu)
    nguong = max(HOAT_ONSET_MIN, 0.35 * float(hoat.max()))
    onset = next((k for k in range(n) if hoat[k] > nguong), None) if hoat.max() >= HOAT_ONSET_MIN else None
    if onset is None:
        return {"k_cat": None, "loai": "khong_dong", "onset": None, "dinh1": None, "cua_so_bac": {}, "lap_sau": False, "do": m}
    k = onset
    while k + 1 < n and not (hoat[k] >= hoat[k + 1] and hoat[k] > 0.6 * hoat.max() and k - onset > 6):
        k += 1
    dinh1 = k
    nghi = (dd <= NGHI_DO) & (vv <= NGHI_V) & (m["vy"] <= NGHI_QUAY) & (m["vh"] <= NGHI_DI)
    bac = {"truoc_dinh": 0, "ngan": 0}
    cat = None
    i = 0
    while i < n:
        if not nghi[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and nghi[j + 1]:
            j += 1
        # cửa sổ kết trước/đúng đỉnh = tư thế trùng idle TRƯỚC động tác (đứng chờ) → bác; cửa sổ mở trước đỉnh
        # nhưng kéo dài qua đỉnh (cú quay dừng đúng lúc đỉnh) thì chỉ tính phần SAU đỉnh
        if j <= dinh1:
            bac["truoc_dinh"] += 1
        else:
            a = max(i, dinh1 + 1)
            if (j - a + 1) / FPS < GIU_NGHI_S:
                bac["ngan"] += 1
            else:
                cat = a + int(GIU_NGHI_S * FPS * 0.5)
                break
        i = j + 1
    if cat is not None:
        lap_sau = bool(cat + 1 < n and hoat[cat + 1:].max() > 0.5 * hoat.max())
        return {"k_cat": cat, "loai": "tu_nghi", "onset": onset, "dinh1": dinh1, "cua_so_bac": bac, "lap_sau": lap_sau, "do": m}
    return {"k_cat": min(n - 1, dinh1 + int(giu_s * FPS)), "loai": "giu", "onset": onset, "dinh1": dinh1,
            "cua_so_bac": bac, "lap_sau": False, "do": m}


def diem_dau(dd, onset: int) -> int:
    """Khung MỞ MÀN: khung GẦN IDLE cuối cùng trước `onset` (≤ `NGUONG_DAU`), không phải `onset` − 0,2 s cố định.

    12/09 tối, đo 60 clip lô catalog: khung CUỐI so idle 0,0° (lối ra + trộn đúng) nhưng khung ĐẦU trung vị 8,0°,
    p90 25,8°, max 56° — chỉ 35 % clip mở màn trong 5°. Gốc: `onset` là mốc ĐỘ HOẠT ĐỘNG vượt 35 % đỉnh, với động
    tác nhẹ thì mốc đó đã nằm giữa pha vào, lùi 0,2 s không đủ về idle. Mọi clip đều sinh từ history idle nên khung
    0 luôn ≈ idle: lùi tới khi chạm ngưỡng là luôn tìm được, và phần đứng chờ trước đó vẫn bị cắt.
    """
    k = int(onset)
    while k > 0 and float(dd[k]) > NGUONG_DAU:
        k -= 1
    return k


def diem_lang_loi_ra(ra: dict, idle_pose: dict) -> int | None:
    n = len(ra["bones"]["Hips"])
    dd = np.array([lech(tu_the(ra, k), idle_pose)[0] for k in range(n)])
    vv = np.array([van_toc(ra, k) for k in range(n)])
    w = int(LANG_S * FPS)
    for a in range(0, n - w):
        if (dd[a:a + w] <= LANG_DO).all() and (vv[a:a + w] <= LANG_V).all():
            return a + w
    return None


def tron_ve_idle(bones: dict, idle_clip: dict, k_idle: int, fps: float, dau: bool = False, cuoi: bool = True,
                 gom: tuple = (), giu_chan_the_gioi: bool = False) -> None:
    """Trộn TRON_S ở CUỐI (và/hoặc ĐẦU) về khung idle cho thân trên + xoay hông, chân giữ nguyên. Sửa TẠI CHỖ.

    `bones[x]` = danh sách [ex, ey, ez] (không mốc thời gian). Hips giữ HƯỚNG QUAY của chính khung biên (clip toàn
    thân được quay tự do), chỉ lấy phần nghiêng/ngửa của idle. Tách từ lap_ghep (kho ARDY chỉ cần cuối — khung đầu
    đã là idle nhờ history) để clip ngoài kho (Kimodo, lệch idle cả hai đầu) dùng CÙNG một thuật toán; `fps` theo
    clip vì Kimodo 30 còn ARDY 60.
    """
    from loi.dong_thoi_gian import _q_sang_euler, _q_slerp
    from loi.goc_troi import _q_nhan, _q_yaw
    N = len(bones["Hips"])
    nb = int(TRON_S * fps)
    hips_cu = [list(e) for e in bones["Hips"]] if giu_chan_the_gioi else None
    for x, tr in bones.items():
        if (x in CHAN and x not in gom) or x not in idle_clip["bones"] or len(tr) != N or N <= nb + 1:
            continue
        for bien in ([N - 1] if cuoi else []) + ([0] if dau else []):
            qi = _q(idle_clip["bones"][x][k_idle][1:4])
            if x == "Hips":
                qi = np.asarray(_q_nhan(_q_yaw(_yaw(tr[bien])), _hips_khong_yaw(idle_clip["bones"]["Hips"][k_idle][1:4])), float)
            if bien == N - 1:
                gan = tr[N - nb - 1]
                thu_tu = [N - nb + i for i in range(nb)]
            else:
                gan = tr[nb]
                thu_tu = [nb - 1 - i for i in range(nb)]
            for i, k in enumerate(thu_tu):
                w = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / nb)
                e = _q_sang_euler(_q_slerp(_q(tr[k]), qi, w), gan)
                tr[k] = list(e)
                gan = e
    if giu_chan_the_gioi and hips_cu is not None:
        # Trộn HÔNG về idle làm cả hai chân (con của Hips) quay theo: nghiêng hông lệch 2,7–6,5° dời bàn chân thêm
        # 5–10 cm dù góc chân không đổi (Kimodo lượt 2, 14/09). Bù ĐÙI để chân giữ nguyên hướng THẾ GIỚI:
        # UpLeg' = Hips'⁻¹ ⊗ Hips ⊗ UpLeg — bàn chân đứng yên, hông + thân trên vẫn về idle.
        chi_so = sorted(set(([N - nb + i for i in range(nb)] if cuoi else []) + ([i for i in range(nb)] if dau else [])))
        for x in ("LeftUpLeg", "RightUpLeg"):
            tr = bones.get(x)
            if not tr or len(tr) != N:
                continue
            for k in chi_so:
                hm = _q(bones["Hips"][k])
                nghich = (hm[0], -hm[1], -hm[2], -hm[3])
                q = _q_nhan(nghich, _q_nhan(_q(hips_cu[k]), _q(tr[k])))
                tr[k] = list(_q_sang_euler(np.asarray(q, float), tr[k]))


# NỐI QUÁN TÍNH TẠI MỐI GHÉP ĐỘNG TÁC → LỐI RA (15/09, user: "12: có thấy 2 chân giật vị trí từ trong ra ngoài 1 chút").
# Lối ra sinh bằng `history` = đuôi động tác, nhưng ARDY KHÔNG nối liền tuyệt đối: khung đầu lối ra lệch khung cắt 1–1,35°
# ở đùi/cổ chân (bước thường 0,3–0,6°), bàn chân so hông nhảy 1,1–1,7 cm (thường 0,3), service tự báo seed_jump 1,33°
# (clip 'dance'); 'squat down' nhảy 2,2–2,4 cm · 5,7° khớp chân. Lịch sử dài hơn (16/24 khung) KHÔNG đổi gì — service luôn
# lấy 4 khung model. Quét 22 clip: mọi mối ghép bước chân một khung gấp 10–60 lần trung vị quanh đó. Nghiệm: độ lệch giữa
# khung đầu lối ra và khung cuối động tác NGOẠI SUY một bước vận tốc, tắt dần bậc 5 trong `NOI_S` (inertialization —
# đạo hàm bậc 1–2 bằng 0 ở cuối cửa sổ). Áp cho xương, `hips_pos`, và quỹ đạo bàn chân ARDY mà IK bám (`_chan_clip_ghep`).
NOI_S = float(os.getenv("ARDY_KHO_NOI_S", "0.3"))


def _so_khung_noi(n_ra: int) -> int:
    """Số khung tắt dần độ lệch ở đầu lối ra: `NOI_S`, không dài hơn chính lối ra. ĐƯỢC chồng lên cửa sổ trộn về idle:
    `tron_ve_idle` không đụng xương CHÂN, thân trên thì trộn TỪ tư thế đã nối về idle (khung cuối vẫn đúng idle). Bản đầu cấm
    chồng nên lối ra ngắn (nhảy lò cò 19 khung < cửa sổ trộn 30) không được nối: bước chân tại ghép 0,25 cm/khung giữ nguyên."""
    n = min(int(round(NOI_S * FPS)), int(n_ra) - 1)
    return n if n >= 3 else 0


def _he_tat_dan(n: int) -> np.ndarray:
    """w(i) = 1 − (10u³ − 15u⁴ + 6u⁵), u = i/n, i ∈ [0, n): w(0) = 1, về 0 với vận tốc và gia tốc 0."""
    u = np.arange(n, dtype=float) / float(n)
    return 1.0 - (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)


def _noi_quan_tinh_xuong(bones: dict, s: int, n: int) -> None:
    """Sửa TẠI CHỖ `bones[x]` (danh sách [ex, ey, ez]) từ khung `s` (khung đầu lối ra) trong `n` khung."""
    from loi.dong_thoi_gian import _q_sang_euler, _q_slerp
    from loi.goc_troi import _q_nhan
    if s < 2 or n <= 0:
        return
    w = _he_tat_dan(n)
    lien = lambda q: (q[0], -q[1], -q[2], -q[3])  # noqa: E731 — nghịch đảo quaternion đơn vị
    for tr in bones.values():
        if len(tr) < s + n:
            continue
        q_truoc, q_cuoi, q0 = _q(tr[s - 2]), _q(tr[s - 1]), _q(tr[s])
        dich0 = _q_nhan(q_cuoi, _q_nhan(lien(q_truoc), q_cuoi))      # khung cuối + một bước vận tốc
        lech = _q_nhan(dich0, lien(q0))
        gan = tr[s - 1]
        for i in range(n):
            k = s + i
            q = _q_nhan(_q_slerp((1.0, 0.0, 0.0, 0.0), lech, float(w[i])), _q(tr[k]))
            e = _q_sang_euler(np.asarray(q, float), gan)
            tr[k] = list(e)
            gan = e


def _noi_quan_tinh_vi_tri(P, s: int, n: int):
    """(T,3) hoặc danh sách [x, y, z]: dời khung s.. theo độ lệch so với khung cuối + một bước vận tốc, tắt dần. Trả mảng mới."""
    P = np.array(P, float)
    if s < 2 or n <= 0 or len(P) < s + n:
        return P
    lech = (2.0 * P[s - 1] - P[s - 2]) - P[s]
    P[s:s + n] += _he_tat_dan(n)[:, None] * lech
    return P


NHAC_CHAN_NGUONG_M = 0.02          # bàn chân cao hơn mức đứng quá ngưỡng này = đã nhấc (bắt đầu pha vung)


def _giu_lech_chan(P, s: int, n: int):
    """Bàn chân (T,3) tại mối ghép: độ lệch (khung cuối + một bước vận tốc − khung đầu lối ra) GIỮ NGUYÊN khi chân còn
    đứng, bắt đầu tắt dần bậc 5 trong `n` khung từ lần nhấc chân đầu tiên của lối ra; không nhấc thì giữ tới hết. Trả mảng mới."""
    P = np.array(P, float)
    if s < 2 or n <= 0 or len(P) <= s:
        return P
    lech = (2.0 * P[s - 1] - P[s - 2]) - P[s]
    m = len(P) - s
    cao = P[s:, 1] - float(P[s, 1])
    nhac = np.nonzero(cao > NHAC_CHAN_NGUONG_M)[0]
    w = np.ones(m)
    if len(nhac):
        k = int(nhac[0])
        d = min(n, m - k)
        w[k:k + d] = _he_tat_dan(n)[:d]
        w[k + d:] = 0.0
    P[s:] += w[:, None] * lech
    return P


# XOAY MẶT PHẲNG GỐI (15/09, user "góc xoay của chân chỗ đùi giật xoay mạnh"): đo theo tầng trên dance/kicks — gia tốc góc
# swivel gối quanh trục háng→cổ chân đã 1.800–2.800 °/s² ngay ở khung ARDY THÔ 20 fps, nâng 60 fps dồn 50 % năng lượng vào
# khung nút (đều = 33 %), retarget/IK không thêm. Nhiễu nằm ở nội dung model, không ở nội suy. Làm mượt góc đó bằng cách xoay
# CẢ CHUỖI chân quanh trục háng→cổ chân: cổ chân nằm trên trục nên đứng yên tuyệt đối, hướng bàn chân bù lại giữ world —
# chỗ đặt chân/IK không đổi, chỉ hướng gối. σ 0,05 s: gia tốc −50…60 %, pha về 33 %, gối dời ≤1,9 cm. 0 = tắt.
XOAY_GOI_S = float(os.getenv("ARDY_KHO_XOAY_GOI_S", "0.05"))


def muot_xoay_goi(clip: dict) -> dict:
    """Làm mượt góc swivel gối từng chân (Gauss `XOAY_GOI_S`) bằng phép xoay quanh trục háng→cổ chân. Sửa `clip` tại chỗ
    (UpLeg + Foot; Leg cục bộ giữ nguyên). Gối gần duỗi (mặt phẳng vô nghĩa) thì không xoay. Trả {chân: độ xoay lớn nhất °}."""
    from scipy.ndimage import gaussian_filter1d
    from loi import buoc_chan as BC
    from loi.dong_thoi_gian import _q_sang_euler
    lu = BC._luoi(clip)
    fps_c = float(clip.get("fps") or FPS)
    sig = XOAY_GOI_S * fps_c
    if lu is None or sig <= 0:
        return {}
    b, moc = lu
    n = len(moc)
    if n < 4:
        return {}
    _eq, qmul, _qr, _m = BC._lo()
    inv = lambda q: q * np.array([1.0, -1.0, -1.0, -1.0])  # noqa: E731
    ks = np.arange(n)
    fk = BC._fk(clip, b, moc, ks)
    ra = {}
    for c, (hip, _goi, co) in BC._CHUOI.items():
        F = fk[c]
        U, G, A = (np.asarray(F[k], float) for k in ("p_u", "p_l", "p_f"))
        ax = A - U
        ax /= np.maximum(np.linalg.norm(ax, axis=1), 1e-9)[:, None]
        kn = G - U
        kn -= np.sum(kn * ax, 1)[:, None] * ax
        lech = np.linalg.norm(kn, axis=1)
        kn /= np.maximum(lech, 1e-9)[:, None]
        w = np.clip((lech - 0.015) / 0.03, 0.0, 1.0)          # gối lệch trục < 1,5 cm: không định nghĩa được mặt phẳng
        phi = np.zeros(n)
        for i in range(1, n):                                  # vận chuyển song song hướng gối theo trục đang quay
            a, bb, x = ax[i - 1], ax[i], kn[i - 1]
            cr = np.cross(a, bb)
            s = float(np.linalg.norm(cr))
            if s > 1e-9:
                cr /= s
                g = math.atan2(s, float(np.dot(a, bb)))
                x = x * math.cos(g) + np.cross(cr, x) * math.sin(g) + cr * np.dot(cr, x) * (1.0 - math.cos(g))
            buoc = math.atan2(float(np.dot(np.cross(x, kn[i]), bb)), float(np.dot(x, kn[i])))
            phi[i] = phi[i - 1] + buoc * min(w[i], w[i - 1])
        d = np.clip((gaussian_filter1d(phi, sig, mode="nearest") - phi) * w, -math.radians(15), math.radians(15))
        rq = np.concatenate([np.cos(d / 2)[:, None], ax * np.sin(d / 2)[:, None]], axis=1)
        qw_u2 = qmul(rq, np.asarray(F["qw_u"], float))
        qw_l2 = qmul(rq, np.asarray(F["qw_l"], float))
        moi = {hip: qmul(inv(np.asarray(fk["qw_h"], float)), qw_u2), co: qmul(inv(qw_l2), np.asarray(F["qw_f"], float))}
        for ten, qn in moi.items():
            tr = b[ten]
            gan = tuple(tr[0][1:4])
            for k in range(n):
                e = _q_sang_euler(tuple(float(v) for v in qn[k]), gan)
                tr[k][1], tr[k][2], tr[k][3] = e
                gan = e
        ra[c[0]] = round(math.degrees(float(np.abs(d).max())), 2)
    return ra


# HƯỚNG GỐI Ở HAI ĐẦU CLIP = HƯỚNG GỐI THẾ NGHỈ CLIENT (15/09, trình duyệt nút 6/8/11): thế nghỉ client (IdleDung) giữ gối
# thẳng trước (hệ chậu L −179,5° / R +177,6°), còn idle ARDY đóng băng xoè gối ra 8–10° (−170/+170). Runtime `moi_noi._khop_chan`
# trộn độ xoắn gối về khung đầu clip trong 6 khung → hai đùi xoay 8–10° trong 0,1 s lúc hông đứng yên — đúng cú "đùi giật xoay"
# user thấy, có sẵn từ trước khi làm mượt xoay. Sửa ở clip: kéo góc gối khung đầu/cuối về đúng thế nghỉ client, phần kéo tắt dần
# bậc 5 vào nội dung trong `GOI_NGHI_S`, xoay quanh trục háng→cổ chân (cổ chân đứng yên). 0 = tắt.
GOI_NGHI_S = float(os.getenv("ARDY_KHO_GOI_NGHI_S", "0.5"))
GOI_NGHI_TEP = os.getenv("ARDY_KHO_GOI_NGHI_TEP", "")


def _goc_goi_chau(fk: dict, c: str) -> np.ndarray:
    """Góc hướng gối quanh trục háng→cổ chân trong hệ chậu (rad): 0 = gối chỉ thẳng trước, dương = về phía chân phải."""
    U, G, A = (np.asarray(fk[c][k], float) for k in ("p_u", "p_l", "p_f"))
    ax = A - U
    ax /= np.maximum(np.linalg.norm(ax, axis=1), 1e-9)[:, None]
    kn = G - U
    kn -= np.sum(kn * ax, 1)[:, None] * ax
    lat = np.asarray(fk["RightFoot"]["p_u"], float) - np.asarray(fk["LeftFoot"]["p_u"], float)
    lat -= np.sum(lat * ax, 1)[:, None] * ax
    lat /= np.maximum(np.linalg.norm(lat, axis=1), 1e-9)[:, None]
    return np.arctan2(np.sum(kn * lat, 1), np.sum(kn * np.cross(ax, lat), 1))


_GOC_NGHI: dict = {}


def goc_goi_nghi_client() -> dict | None:
    """{chân: góc gối hệ chậu (rad)} ở khung đầu clip nghỉ client (IdleDung01 của Mira), đọc một lần."""
    if "g" not in _GOC_NGHI:
        from pathlib import Path as _P
        from loi import buoc_chan as BC
        tep = _P(GOI_NGHI_TEP) if GOI_NGHI_TEP else _P(__file__).resolve().parents[2] / "app/motion_clips/mira/IdleDung01.json"
        g = None
        try:
            c = json.loads(tep.read_text())
            lu = BC._luoi(c)
            if lu is not None:
                fk = BC._fk(c, lu[0], lu[1], np.arange(2))
                g = {x: float(_goc_goi_chau(fk, x)[0]) for x in BC.CHAN}
        except (OSError, ValueError):
            g = None
        _GOC_NGHI["g"] = g
    return _GOC_NGHI["g"]


def khop_goi_nghi(clip: dict) -> dict:
    """Kéo hướng gối khung đầu + cuối về thế nghỉ client, tắt dần trong `GOI_NGHI_S`. Sửa `clip` tại chỗ; trả {chân: (lệch đầu°,
    lệch cuối°)} đã bù. Dấu phép xoay TỰ KIỂM bằng FK ở khung đầu (quy ước §2: không tin chiều trục)."""
    from loi import buoc_chan as BC
    from loi.dong_thoi_gian import _q_sang_euler
    dich = goc_goi_nghi_client()
    lu = BC._luoi(clip)
    fps_c = float(clip.get("fps") or FPS)
    if dich is None or lu is None or GOI_NGHI_S <= 0:
        return {}
    b, moc = lu
    n = len(moc)
    if n < 4:
        return {}
    _eq, qmul, _qr, _m = BC._lo()
    inv = lambda q: q * np.array([1.0, -1.0, -1.0, -1.0])  # noqa: E731
    boc = lambda x: (x + math.pi) % (2 * math.pi) - math.pi  # noqa: E731
    fk = BC._fk(clip, b, moc, np.arange(n))
    nT = max(2, min(int(round(GOI_NGHI_S * fps_c)), n // 2))
    s = np.arange(nT) / nT
    tat = 1.0 - (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)          # 1 → 0 bậc 5
    ra = {}
    for c, (hip, _goi, co) in BC._CHUOI.items():
        g = _goc_goi_chau(fk, c)
        gioi = math.radians(20)
        l0 = float(np.clip(boc(dich[c] - g[0]), -gioi, gioi))
        l1 = float(np.clip(boc(dich[c] - g[-1]), -gioi, gioi))
        e = np.zeros(n)
        e[:nT] += l0 * tat
        e[n - nT:] += l1 * tat[::-1]
        F = fk[c]
        U, A = np.asarray(F["p_u"], float), np.asarray(F["p_f"], float)
        ax = A - U
        ax /= np.maximum(np.linalg.norm(ax, axis=1), 1e-9)[:, None]
        # tự kiểm dấu: thử +e ở khung đầu, góc phải tiến về đích; không thì đảo
        dau = 1.0
        if abs(l0) > 1e-4:
            rq0 = np.array([math.cos(l0 / 2), *(ax[0] * math.sin(l0 / 2))])
            G0 = U[0] + _xoay_v(rq0, np.asarray(F["p_l"], float)[0] - U[0])
            fk0 = {"LeftFoot": {kk: np.asarray(vv, float)[:1] for kk, vv in fk["LeftFoot"].items()},
                   "RightFoot": {kk: np.asarray(vv, float)[:1] for kk, vv in fk["RightFoot"].items()}}
            fk0[c]["p_l"] = G0[None]
            if abs(boc(dich[c] - float(_goc_goi_chau(fk0, c)[0]))) > abs(l0):
                dau = -1.0
        d = dau * e
        rq = np.concatenate([np.cos(d / 2)[:, None], ax * np.sin(d / 2)[:, None]], axis=1)
        qw_u2 = qmul(rq, np.asarray(F["qw_u"], float))
        qw_l2 = qmul(rq, np.asarray(F["qw_l"], float))
        moi = {hip: qmul(inv(np.asarray(fk["qw_h"], float)), qw_u2), co: qmul(inv(qw_l2), np.asarray(F["qw_f"], float))}
        for ten, qn in moi.items():
            tr = b[ten]
            gan = tuple(tr[0][1:4])
            for k in range(n):
                ev = _q_sang_euler(tuple(float(v) for v in qn[k]), gan)
                tr[k][1], tr[k][2], tr[k][3] = ev
                gan = ev
        ra[c[0]] = (round(math.degrees(l0), 2), round(math.degrees(l1), 2))
    return ra


def _xoay_v(q, v):
    """Xoay véc-tơ `v` bằng quaternion (w, x, y, z)."""
    w, u = float(q[0]), np.asarray(q[1:], float)
    v = np.asarray(v, float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


# THẾ NGHỈ CLIENT Ở HAI ĐẦU CLIP (15/09 tối, user: "chuẩn hoá lại ardy để có đầu/cuối chuẩn idle client để các motion không phải
# toàn thân thì không cần bước chân"). Đo trước khi sửa (16 clip chân + 6 cử chỉ, so IdleDung01 khung 0 = thế A client): đầu/cuối
# khớp idle ARDY đóng băng (cuối 0°) nhưng idle ARDY ≠ thế nghỉ client — xương tay tới ~53°, chân 12–20°, hai bàn chân rộng
# 30–41 cm (client 25,1), bàn chân lệch hệ hông 7–17 cm, hông cao ±2,6 cm; trình duyệt nhảy múa: hết clip hai chân 29,7 → 25,1 cm
# trong 0,5 s. Runtime phải bước/trượt để bắc cầu. Nay clip TỰ mở/kết ĐÚNG thế A: thân trên + ngón + nghiêng hông (giữ yaw của
# chính khung biên) kéo về A, phần kéo tắt dần bậc 5 `THE_NGHI_S` vào nội dung; hông cao về A; bàn chân IK — clip KHÔNG toàn thân
# ghim đúng chỗ thế A suốt clip (không bước), clip toàn thân giữ độ lệch khi chân đang đứng và chỉ đổi trong pha vung (nhấc
# đầu tiên tắt lệch đầu, nhấc cuối cùng nạp lệch cuối) — không kéo trượt chân đang đặt. 0 = tắt.
THE_NGHI_S = float(os.getenv("ARDY_KHO_THE_NGHI_S", "0.5"))
_THE_NGHI: dict = {}


def the_nghi_client() -> dict | None:
    """Thế nghỉ client (IdleDung01 khung 0 của Mira): {"bones": {x: euler}, "hy": độ cao hông (hips_pos y), "chan": {chân:
    (vị trí cổ chân so hông trong hệ BỎ YAW của hông, y cổ chân tuyệt đối, hướng bàn chân bỏ yaw hông wxyz)}}. Đọc một lần."""
    if "a" not in _THE_NGHI:
        from pathlib import Path as _P
        from loi import buoc_chan as BC
        from loi.goc_troi import _q_nhan, _q_yaw, _yaw_cua
        tep = _P(GOI_NGHI_TEP) if GOI_NGHI_TEP else _P(__file__).resolve().parents[2] / "app/motion_clips/mira/IdleDung01.json"
        a = None
        try:
            c = json.loads(tep.read_text())
            lu = BC._luoi(c)
            if lu is not None:
                fk = BC._fk(c, lu[0], lu[1], np.arange(2))
                qh = tuple(float(v) for v in np.asarray(fk["qw_h"], float)[0])
                g = _yaw_cua(qh)
                ph = np.asarray(fk["p_h"], float)[0]
                chan = {}
                for x in BC.CHAN:
                    pf = np.asarray(fk[x]["p_f"], float)[0]
                    rel = _xoay_v(np.asarray(_q_yaw(-g), float), pf - ph)
                    qrel = np.asarray(_q_nhan(_q_yaw(-g), tuple(float(v) for v in np.asarray(fk[x]["qw_f"], float)[0])), float)
                    chan[x] = (rel, float(pf[1]), qrel)
                hp = c.get("hips_pos") or [[0.0, 0.0, 0.0, 0.0]]
                a = {"bones": {x: list(tr[0][1:4]) for x, tr in c["bones"].items()}, "hy": float(hp[0][2]), "chan": chan}
        except (OSError, ValueError):
            a = None
        _THE_NGHI["a"] = a
    return _THE_NGHI["a"]


def _len5(n: int) -> np.ndarray:
    """0 → 1 bậc 5 trong n khung, khung cuối đúng 1."""
    u = np.arange(1, n + 1, dtype=float) / float(n)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


GHIM_DOI_CM = 4.0     # bàn chân clip dời ngang ≤ ngần này VÀ không nhấc quá `NHAC_CHAN_NGUONG_M` → ghim đúng thế nghỉ suốt clip


def khop_the_nghi(clip: dict, toan_than: bool | None = None) -> dict:
    """Đưa khung đầu + cuối clip về đúng thế nghỉ client (xem ghi chú `THE_NGHI_S`). Sửa `clip` tại chỗ; trả số đo.

    CHẾ ĐỘ GHIM QUYẾT THEO DỮ LIỆU CHÂN CỦA CHÍNH CLIP, không theo cờ toàn thân của chỉ mục: thử trên kho, cờ `toan_than` xếp
    nhảy múa (bàn chân dời 29 cm) và nhảy lên (bật khỏi sàn) là KHÔNG toàn thân → ghim làm mất cú bật (gối Δ² 2,9 → 11,4) và vặn
    gối nhảy múa (gia tốc xoay 865 → 5.123 °/s²). `toan_than` truyền vào chỉ để ghi sổ."""
    from loi import buoc_chan as _BC
    _lu = _BC._luoi(clip)
    if _lu is not None and len(_lu[1]) >= 4:
        _fk0 = _BC._fk(clip, _lu[0], _lu[1], np.arange(len(_lu[1])))
        _doi = max(float(np.linalg.norm((np.asarray(_fk0[x]["p_f"], float) - np.asarray(_fk0[x]["p_f"], float)[0])[:, [0, 2]],
                                        axis=1).max()) for x in _BC.CHAN)
        # chỉ xét dời NGANG: gót nhấc khi ngồi xổm/nhón chân vẫn là "đứng một chỗ" (thử 22 clip: ngồi xổm dời 0,3 cm mà
        # gót nhấc > 2 cm → bị xếp toàn thân và bàn chân bị kéo trượt 7,9 cm). Ghim chỉ khoá (x, z), độ cao theo clip.
        ghim = _doi * 100 <= GHIM_DOI_CM
    else:
        ghim = False
    ra_cha = {"co_toan_than": toan_than, "ghim": ghim}
    kq = _khop_the_nghi(clip, not ghim)
    kq.update(ra_cha)
    return kq


def _khop_the_nghi(clip: dict, toan_than: bool) -> dict:
    """Lõi `khop_the_nghi`: `toan_than` False = ghim bàn chân đúng thế nghỉ suốt clip."""
    from loi import buoc_chan as BC
    from loi.dong_thoi_gian import _q_sang_euler, _q_slerp
    from loi.goc_troi import _q_nhan, _q_yaw, _yaw_cua
    A = the_nghi_client()
    lu = BC._luoi(clip)
    if A is None or lu is None or THE_NGHI_S <= 0:
        return {}
    b, moc = lu
    n = len(moc)
    if n < 4:
        return {}
    fps_c = float(clip.get("fps") or FPS)
    nT = max(2, min(int(round(THE_NGHI_S * fps_c)), n // 2))
    tat = _he_tat_dan(nT)
    w0 = np.zeros(n)
    w0[:nT] = tat
    w1 = np.zeros(n)
    w1[n - nT:] = tat[::-1]
    nghich = lambda q: (float(q[0]), -float(q[1]), -float(q[2]), -float(q[3]))  # noqa: E731
    ra: dict = {"toan_than": bool(toan_than)}
    # 1) thân trên + ngón + nghiêng hông: độ lệch tới A ở mỗi biên nhân TRÁI (hệ cha), tắt dần vào nội dung
    goc_max = 0.0
    for x, tr in b.items():
        if x in CHAN or x not in A["bones"] or len(tr) != n:
            continue
        lech_bien = []
        for k in (0, n - 1):
            qk = _q(tr[k][1:4])
            qa = _q(A["bones"][x])
            if x == "Hips":
                qa = np.asarray(_q_nhan(_q_yaw(_yaw(tr[k][1:4])), tuple(float(v) for v in _hips_khong_yaw(A["bones"]["Hips"]))), float)
            if not any(ng in x for ng in NGON):
                goc_max = max(goc_max, _ang(qa, qk))
            lech_bien.append(_q_nhan(tuple(float(v) for v in qa), nghich(qk)))
        gan = tuple(tr[0][1:4])
        for k in range(n):
            if w0[k] <= 0.0 and w1[k] <= 0.0:
                gan = tuple(tr[k][1:4])
                continue
            q = tuple(float(v) for v in _q(tr[k][1:4]))
            if w1[k] > 0.0:
                q = _q_nhan(_q_slerp((1.0, 0.0, 0.0, 0.0), lech_bien[1], float(w1[k])), q)
            if w0[k] > 0.0:
                q = _q_nhan(_q_slerp((1.0, 0.0, 0.0, 0.0), lech_bien[0], float(w0[k])), q)
            ev = _q_sang_euler(np.asarray(q, float), gan)
            tr[k][1], tr[k][2], tr[k][3] = ev
            gan = ev
    ra["than_tren_lech_bien_do"] = round(goc_max, 2)
    # 2) hông: độ cao về A ở hai biên; clip không toàn thân kéo cả chỗ đứng cuối về chỗ đầu (chân ghim một chỗ)
    hp = clip.get("hips_pos") or []
    if len(hp) == n:
        y0 = A["hy"] - float(hp[0][2])
        y1 = A["hy"] - float(hp[-1][2])
        x1 = 0.0 if toan_than else float(hp[0][1]) - float(hp[-1][1])
        z1 = 0.0 if toan_than else float(hp[0][3]) - float(hp[-1][3])
        for k in range(n):
            hp[k][1] = float(hp[k][1]) + w1[k] * x1
            hp[k][2] = float(hp[k][2]) + w0[k] * y0 + w1[k] * y1
            hp[k][3] = float(hp[k][3]) + w1[k] * z1
        ra["hong_lech_cm"] = (round(y0 * 100, 2), round(y1 * 100, 2), round(math.hypot(x1, z1) * 100, 2))
    # 3) bàn chân IK tới thế A
    ks = np.arange(n)
    fk = BC._fk(clip, b, moc, ks)
    qh = np.asarray(fk["qw_h"], float)
    ph = np.asarray(fk["p_h"], float)
    g = np.array([_yaw_cua(tuple(float(v) for v in q)) for q in qh])
    dich, huong = {}, {}
    for c in BC.CHAN:
        rel, ya, qrel = A["chan"][c]
        P = np.asarray(fk[c]["p_f"], float)
        Q = np.asarray(fk[c]["qw_f"], float)

        def tgt(k, rel=rel, ya=ya, qrel=qrel):
            qy = np.asarray(_q_yaw(float(g[k])), float)
            p = ph[k] + _xoay_v(qy, rel)
            p[1] = ya
            return p, np.asarray(_q_nhan(tuple(qy), tuple(qrel)), float)

        p0, q0 = tgt(0)
        p1, q1 = tgt(n - 1)
        if not toan_than:
            # GHIM chỗ đặt chân (x, z) đúng thế nghỉ suốt clip; độ cao + hướng bàn chân theo clip, độ lệch tới thế nghỉ đổi
            # CHẬM cả clip (nhón gót/ngồi xổm giữ nguyên, không có gì trượt ngang)
            a1 = np.r_[0.0, _len5(n - 1)]
            a0 = 1.0 - a1
            D = np.tile(p0, (n, 1))
            D[:, 1] = P[:, 1] + a0 * (ya - P[0, 1]) + a1 * (ya - P[-1, 1])
            d0 = _q_nhan(tuple(float(v) for v in q0), nghich(Q[0]))
            d1 = _q_nhan(tuple(float(v) for v in q0), nghich(Q[-1]))
            H = np.array([_q_nhan(_q_slerp((1.0, 0.0, 0.0, 0.0), d1, float(a1[k])),
                                  _q_nhan(_q_slerp((1.0, 0.0, 0.0, 0.0), d0, float(a0[k])), tuple(float(v) for v in Q[k])))
                          for k in range(n)], float)
        else:
            nhac = (P[:, 1] - min(float(P[:, 1].min()), ya)) > NHAC_CHAN_NGUONG_M
            idx = np.nonzero(nhac)[0]
            if len(idx):
                a0 = np.ones(n)
                k1 = int(idx[0])
                dd = min(nT, n - k1)
                a0[k1:k1 + dd] = _he_tat_dan(nT)[:dd]
                a0[k1 + dd:] = 0.0
                k2 = int(idx[-1]) + 1
                s = k2 - 1
                while s > 0 and nhac[s - 1]:
                    s -= 1
                L = max(1, min(nT, k2 - s))
                a1 = np.zeros(n)
                a1[k2 - L:k2] = _len5(L)
                a1[k2:] = 1.0
            else:
                a1 = np.r_[0.0, _len5(n - 1)]
                a0 = 1.0 - a1
            o0, o1 = p0 - P[0], p1 - P[-1]
            D = P + a0[:, None] * o0 + a1[:, None] * o1
            d0 = _q_nhan(tuple(float(v) for v in q0), nghich(Q[0]))
            d1 = _q_nhan(tuple(float(v) for v in q1), nghich(Q[-1]))
            H = np.array([_q_nhan(_q_slerp((1.0, 0.0, 0.0, 0.0), d1, float(a1[k])),
                                  _q_nhan(_q_slerp((1.0, 0.0, 0.0, 0.0), d0, float(a0[k])), tuple(float(v) for v in Q[k])))
                          for k in range(n)], float)
            ra[f"{c[0]}_lech_cm"] = (round(float(np.linalg.norm(o0)) * 100, 1), round(float(np.linalg.norm(o1)) * 100, 1))
        dich[c] = _mem_tam_voi(fk[c], D)
        huong[c] = H

    def quy_dao(c, t):
        i = int(min(n - 1, max(0, round(float(t) * fps_c))))
        return dich[c][i], huong[c][i]

    sai = BC._giai_ik(clip, b, moc, ks, np.ones(n), quy_dao)
    ra["sai_ik_cm"] = round(sai * 100, 2)
    return ra


NUT_FPS = 20.0                      # fps model ARDY: khung 60 fps chia hết cho 3 là khung model thật ("nút")


def _r_log(q):
    q = np.asarray(q, float)
    w = float(np.clip(q[0], -1.0, 1.0)); v = q[1:]; s = float(np.linalg.norm(v))
    return (2.0 * math.atan2(s, w) / s) * v if s > 1e-9 else np.zeros(3)


def _r_exp(r):
    a = float(np.linalg.norm(r))
    return np.array([math.cos(a / 2), *((math.sin(a / 2) / a) * r)]) if a > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])


def lam_tron_nut(clip: dict, buoc: int = 3) -> dict:
    """LẤY MẪU LẠI giữa các NÚT model bằng Catmull-Rom (véc-tơ xoay từng xương + hips_pos). Sửa TẠI CHỖ, trả clip.

    GỐC GIẬT của clip ARDY (đo 15/09): service sinh 20 fps rồi nội suy TUYẾN TÍNH (slerp giữa hai khung model) lên 60
    fps → quỹ đạo gãy khúc tại mỗi khung thứ 3, gia tốc nhảy bậc = giật xung 20 Hz (tự tương quan giật theo khung đỉnh
    ở lag 3 và 6: 0,42–0,66). Giật p95 clip kho 1.700–2.800 rad/s³ trong khi bao người thật (BEAT) 470. Đây là NỘI SUY
    ĐÚNG HƠN chứ không phải lọc: giữ nguyên các nút, chỉ vẽ lại đường giữa hai nút — đo 4 clip kho: giật p95 → 600–770,
    tốc độ tay p99 / biên độ / trượt chân KHÔNG đổi, KHÔNG trễ. Lọc thấp hay lò xo (đề xuất "pipeline 3 tầng") giảm giật
    bằng cách cắt đỉnh tốc độ tay và thêm trễ — sai chỗ.
    Nút = khung k với k % buoc == 0 (`generate_clip`: pos = seeded + k·20/60, a = 0 đúng tại các khung đó) + khung cuối.
    Áp NGAY SAU retarget (idle, động tác, lối ra) trước khi cắt/trộn; và áp được lên tệp kho 60 fps có sẵn không cần
    render lại vì nút còn nguyên trong tệp.
    """
    if float(clip.get("fps") or FPS) <= NUT_FPS:
        return clip                        # đã ở fps gốc model — không có khung nội suy nào để vẽ lại
    from loi.dong_thoi_gian import _q_sang_euler
    b = clip.get("bones") or {}
    n = len(next(iter(b.values()), []))
    if n < 4:
        return clip
    nut = list(range(0, n, buoc))
    if nut[-1] != n - 1:
        nut.append(n - 1)
    tn = np.array(nut, float)

    # SPLINE BẬC BA C² (scipy, natural) qua các nút: gia tốc liên tục cả ở nút → giật p95 thấp hơn Catmull-Rom (chỉ C¹,
    # gia tốc nhảy tại nút) thêm 15–20 % (đo 4 clip kho: 784→619, 752→662, 593→499, 2216→1805), tốc độ/biên độ tay không
    # đổi. Thiếu scipy thì lùi về Catmull-Rom (cùng nút, cùng kết luận, chỉ kém mượt hơn ở nút).
    try:
        from scipy.interpolate import CubicSpline

        def cr(P):
            return CubicSpline(tn, P[nut], axis=0, bc_type="natural")(np.arange(n, dtype=float))
    except ImportError:
        he = []
        for i in range(n):
            k = int(min(max(np.searchsorted(tn, i, side="right") - 1, 0), len(nut) - 2))
            u = (i - tn[k]) / max(tn[k + 1] - tn[k], 1e-9)
            he.append((nut[max(k - 1, 0)], nut[k], nut[k + 1], nut[min(k + 2, len(nut) - 1)], u))

        def cr(P):
            out = np.empty((n, P.shape[1]))
            for i, (i0, i1, i2, i3, u) in enumerate(he):
                p0, p1, p2, p3 = P[i0], P[i1], P[i2], P[i3]
                out[i] = 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                                + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u)
            return out

    for x, tr in b.items():
        if len(tr) != n:
            continue
        R = []; gan = None
        for f in tr:
            r = _r_log(_q(f[1:4]))
            if gan is not None and np.linalg.norm(r - gan) > math.pi:      # chọn nhánh gần khung trước
                r = r * (1.0 - 2.0 * math.pi / max(float(np.linalg.norm(r)), 1e-9))
            R.append(r); gan = r
        out = cr(np.array(R))
        gan_e = None
        for i, f in enumerate(tr):
            if i % buoc == 0 or i == n - 1:
                gan_e = tuple(f[1:4]); continue                              # nút giữ nguyên
            e = _q_sang_euler(tuple(_r_exp(out[i])), gan_e or tuple(f[1:4]))
            f[1:4] = [float(v) for v in e]; gan_e = e
    hp = clip.get("hips_pos") or []
    if len(hp) == n:
        out = cr(np.array([f[1:4] for f in hp], float))
        for i, f in enumerate(hp):
            if not (i % buoc == 0 or i == n - 1):
                f[1:4] = [float(v) for v in out[i]]
    return clip



def lap_ghep(clip: dict, k_dau: int, k_cat: int, ra: dict, k_lang: int, idle_clip: dict, k_idle: int = 0) -> dict:
    """Luật 5–6: [k_dau..k_cat] của động tác + [0..k_lang] của lối ra, hips_pos lối ra dời tiếp theo khung cắt
    (retarget trừ root0 từng clip), rồi trộn TRON_S cuối về khung idle cho thân trên + xoay hông (chân giữ nguyên)."""
    bones = {}
    for x, tr in clip["bones"].items():
        a = [list(f[1:4]) for f in tr[k_dau:k_cat + 1]]
        b = [list(f[1:4]) for f in ra["bones"][x][:k_lang + 1]] if x in ra["bones"] else []
        bones[x] = a + b
    N = len(bones["Hips"])
    s_ghep = k_cat - k_dau + 1                               # khung đầu lối ra trong clip ghép
    n_noi = _so_khung_noi(N - s_ghep)
    if n_noi:
        _noi_quan_tinh_xuong(bones, s_ghep, n_noi)          # TRƯỚC trộn về idle (xem NOI_S)
    tron_ve_idle(bones, idle_clip, k_idle, FPS, cuoi=True)
    hpa = [list(f[1:4]) for f in (clip.get("hips_pos") or [])[k_dau:k_cat + 1]]
    hpb = [list(f[1:4]) for f in (ra.get("hips_pos") or [])[:k_lang + 1]]
    if hpa and hpb:
        goc = hpa[-1]
        hpb = [[goc[i] + f[i] for i in range(3)] for f in hpb]
    hp = hpa + hpb
    if n_noi and len(hpa) == s_ghep and len(hp) == N:
        hp = [list(v) for v in _noi_quan_tinh_vi_tri(hp, s_ghep, n_noi)]
    return {"fps": FPS, "duration_s": round((N - 1) / FPS, 4),
            "bones": {x: [[round(k / FPS, 4), *[round(float(v), 5) for v in tr[k]]] for k in range(len(tr))] for x, tr in bones.items()},
            "hips_pos": [[round(k / FPS, 4), *[round(float(v), 5) for v in f]] for k, f in enumerate(hp)]}


def _chan_clip_ghep(hc: dict, tho: list, k_dau: int, k_cat: int, tho2: list, k_lang: int, toan_than: bool | None = None) -> dict:
    """IK bàn chân theo ARDY + nhấc bước lê, chạy MỘT lần trên clip ĐÃ GHÉP (động tác [k_dau..k_cat] + lối ra [0..k_lang]).
    Quỹ đạo ARDY của lối ra dời bằng MỘT tịnh tiến chung sao cho hông nối tiếp khung cắt với đúng vận tốc khung cuối
    (lối ra sinh từ history nên hướng liên tục; chỉ gốc toạ độ có thể lệch). Sửa `hc` tại chỗ.
    Sau IK: đầu/cuối về thế nghỉ client (`khop_the_nghi`) → làm mượt xoay gối → hướng gối hai đầu (`khop_goi_nghi`)."""
    import logging
    from app.modules.body_motion.ardy_retarget import tu_mang
    kq: dict = {}
    try:
        info = _info()
        J = list(info["joints"])
        # làm mượt trên CẢ chuỗi thô rồi mới cắt: nút 20 fps nằm ở khung k % 3 == 0 của chuỗi service trả về,
        # cắt trước thì lệch pha nút (xem `_muot_quy_dao`)
        A1f = _vi_tri_chan_ardy(tu_mang(J, tho), info)
        A2f = _vi_tri_chan_ardy(tu_mang(J, tho2), info)
        A1 = {x: _muot_quy_dao(v)[k_dau:k_cat + 1] for x, v in A1f.items()}
        A2 = {x: _muot_quy_dao(v)[:k_lang + 1] for x, v in A2f.items()}
        H1 = A1["Hips"]
        v = H1[-1] - H1[-2] if len(H1) >= 2 else np.zeros(3)
        dich = H1[-1] + v - A2["Hips"][0]
        A = {x: np.vstack([A1[x], A2[x] + dich]) for x in A1}
        # cùng phép nối quán tính như xương (`lap_ghep`) — IK không được bám lại cú nhảy mối ghép của ARDY. HÔNG tắt dần
        # `NOI_S`; BÀN CHÂN thì KHÔNG tắt khi đang đứng — tắt dần kéo chân đang đặt trượt về chỗ ARDY đặt sau cú nhảy (đo
        # 16 clip: trôi pha đứng nhảy múa 0,16 → 1,07 cm, xoay gót 0,39 → 1,01): giữ nguyên độ lệch tới lần NHẤC CHÂN kế
        # tiếp rồi mới tắt dần trong pha vung; không nhấc lần nào thì giữ tới hết clip (đuôi về nghỉ của runtime bước).
        s = len(A1["Hips"])
        n_noi = _so_khung_noi(len(A2["Hips"]))
        if n_noi:
            for x in ("Hips", "LeftUpLeg", "RightUpLeg"):
                if x in A:
                    A[x] = _noi_quan_tinh_vi_tri(A[x], s, n_noi)
            for x in ("LeftFoot", "RightFoot"):
                A[x] = _giu_lech_chan(A[x], s, n_noi)
        n = len(hc["bones"]["Hips"])
        if len(A["Hips"]) != n:
            logging.getLogger(__name__).warning("IK chân clip ghép bỏ qua: ARDY %d khung ≠ clip %d", len(A["Hips"]), n)
            return {}
        if XOAY_GOI_S > 0:
            kq["xoay_goi"] = muot_xoay_goi(hc)                 # trước IK: IK giữ mặt phẳng gối đã mượt
        if THEO_CHAN:
            kq["theo"] = theo_chan_ardy(hc, [], info, A=A)
        if THE_NGHI_S > 0:
            kq["the_nghi"] = khop_the_nghi(hc, toan_than)      # đầu/cuối = thế nghỉ client; clip đứng một chỗ ghim chân
        if XOAY_GOI_S > 0 and (THEO_CHAN or THE_NGHI_S > 0):
            # IK giải lại mặt phẳng gối theo `truc_on` + làm mượt trường hiệu chỉnh ±4 khung → cấy lại cú xoay gắt ở chân
            # đang vung (bước ngang trái 22.767 °/s² y hệt trước/sau lượt đầu); lượt hai sau IK: 2.390, cổ chân dời 0,0000 cm
            kq["xoay_goi_sau_ik"] = muot_xoay_goi(hc)
        if GOI_NGHI_S > 0:
            kq["goi_nghi"] = khop_goi_nghi(hc)                 # sau cùng: hai đầu clip trùng hướng gối thế nghỉ client
        if NHAC_CHAN:
            kq["nhac"] = nhac_chan_luot(hc)
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).warning("IK chân clip ghép bỏ qua", exc_info=True)
    return kq


def toan_than_theo_noi_dung(do: dict) -> bool:
    """Cùng ngưỡng với runtime (`loi/kho_ardy.toan_than_noi_dung`): DI CHUYỂN THẬT, không phải dồn trọng tâm."""
    from loi.kho_ardy import TOAN_THAN_DI_M, TOAN_THAN_QUAY_DO
    xz = do["xz"]
    return bool((xz.max(axis=0) - xz.min(axis=0)).max() > TOAN_THAN_DI_M or (do["yw"].max() - do["yw"].min()) > TOAN_THAN_QUAY_DO)


# ---- gọi service -----------------------------------------------------------------------------
# NGUỒN SINH CẮM ĐƯỢC (14/09): mặc định HTTP tới ardy_service (`POST /motion`, `/skeleton`). Tool `ardy_render` (image
# RunPod) cắm bản gọi Engine TRONG TIẾN TRÌNH qua `dat_nguon` — công thức clip (cắt · lối ra · trộn · retarget · cổng)
# vẫn chỉ nằm ở tệp này, hai đường không có hai bản.
_NGUON = None


def dat_nguon(nguon) -> None:
    """`nguon.info()` trả skeleton như `/skeleton`; `nguon.motion(than)` nhận đúng thân `POST /motion` và trả
    {"clips": [...]}. `None` = về lại HTTP."""
    global _NGUON
    _NGUON = nguon


def _info() -> dict:
    if _NGUON is not None:
        return _NGUON.info()
    from app.modules.body_motion import ardy_client as ac
    return ac.client()._lay_info()


def _sinh(prompt: str, giay: float, history=None, dich=None, cfg_c=None, theo: bool = True):
    from app.modules.body_motion.ardy_retarget import retarget, tu_mang
    info = _info()
    than = {"text": prompt, "seconds": float(giay), "fps": FPS_ARDY, "chunk_seconds": max(float(giay), 2.5)}
    if history:
        than["history"] = history
    if dich:
        than["target_pose"] = dich
    if cfg_c is not None:
        than["cfg_constraint"] = float(cfg_c)
    if _NGUON is not None:
        d = _NGUON.motion(than)
    else:
        from app.modules.body_motion import ardy_client as ac
        with ac.client()._http() as h:
            r = h.post("/motion", json=than)
            r.raise_for_status()
            d = r.json()
    clips = d.get("clips") or []
    frames = [f for c in clips for f in (c.get("frames") or [])]
    c0 = clips[0] if clips else {}
    fps_ve = float(c0.get("fps") or FPS_ARDY)
    # nâng fps GỐC → lưới kho bằng spline TRONG KHÔNG GIAN ARDY, trước retarget/IK (xem FPS_ARDY/FPS)
    frames = _nang_tho(frames, fps_ve, FPS) if frames else frames
    named = tu_mang(list(c0.get("joints") or info["joints"]), frames) if frames else []
    clip = (retarget(info, named, float(max(FPS, fps_ve)), nguon="ardy", thuoc_hong=THUOC_HONG) if frames else None)
    if clip is not None and LAM_TRON and fps_ve > NUT_FPS:
        lam_tron_nut(clip)                 # chỉ khi service tự nội suy lên fps > gốc (đường cũ)
    # `theo=False`: bên gọi GHÉP nhiều đoạn (render_mot) tự chạy IK bàn chân MỘT lần trên clip đã ghép — IK từng đoạn neo
    # vào khung đầu RIÊNG của đoạn nên tại mối ghép chân nhảy 13–16 cm trong MỘT khung (đo 15/09).
    if clip is not None and THEO_CHAN and theo:
        try:
            theo_chan_ardy(clip, named, info)
        except Exception:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning("retarget theo bàn chân ARDY bỏ qua", exc_info=True)
    if clip is not None and NHAC_CHAN and theo:
        try:
            nhac_chan_luot(clip)
        except Exception:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning("nhấc chân bước lê bỏ qua", exc_info=True)
    return frames, clip, str(c0.get("matched") or ""), float(c0.get("similarity") or 0.0)


# RETARGET GIỮ ĐÚNG CHỖ ĐẶT CHÂN CỦA ARDY (15/09, user: *"phải giải quyết từ ardy… ardy nvidia làm rất chuẩn nên chỉ có bạn
# cấu hình sai"*). Ba chỗ ta dùng sai, đo cùng câu cùng seed (lệch chỗ đặt chân xa nhất trong pha chạm):
#   1. `postprocess` (khử trượt chính thức, bộ giải C++ motion_correction) TẮT ở mọi job — scripts/generate.py của NVIDIA bật
#      mặc định. Bật: xoay gót 3,4 → 2,0 cm, dậm tay 4,6 → 1,0 cm (không gian ARDY).
#   2. Độ dời hông co giãn theo hông→"Head" (0,712) trong khi chân dài như nhau → hông rig đi 71 % quãng thật, bàn chân bị kéo:
#      ARDY 3,4 → rig 10,8 cm. Nay thước "chan" (hông trên cổ chân, 1,016) → 3,1 cm.
#   3. Tư thế nghỉ của chân KHÁC NHAU (ARDY T-pose chân thẳng dưới háng; Mira bàn chân lệch ngang 15,5 cm, lùi 7,7 cm): chép
#      góc qua khung T-pose thì bàn chân rig vẽ cung khác bàn chân ARDY khi hông xoay — quay tại chỗ ARDY 2,5 cm mà rig 8,5 cm.
#      Nghiệm chuẩn của retarget: sau chép góc, IK hai chân cho bàn chân rig đi ĐÚNG quỹ đạo bàn chân ARDY (co giãn cùng hệ số
#      hông, neo tại chỗ đứng khung đầu của rig) — bàn chân đứng yên ở ARDY thì đứng yên ở rig.
THUOC_HONG = os.getenv("ARDY_KHO_THUOC_HONG", "chan")
THEO_CHAN = os.getenv("ARDY_KHO_THEO_CHAN", "1") == "1"


def _muot_quy_dao(P, buoc: int = 3) -> np.ndarray:
    """Quỹ đạo vị trí ARDY (T,3) → spline bậc ba C² qua NÚT model (khung k % buoc == 0 + khung cuối), nút giữ nguyên.

    CÙNG LÝ DO VỚI `lam_tron_nut` (15/09, user: "11, 16 vẫn còn giật lúc rời sàn"): service nội suy TUYẾN TÍNH 20 → 60 fps
    nên FK bàn chân ARDY gãy khúc mỗi 3 khung. `lam_tron_nut` làm mượt XƯƠNG, nhưng IK `theo_chan_ardy` lấy đích từ quỹ
    đạo ARDY THÔ nên cấy lại đúng chỗ gãy: đo trên kho, 30/30 vọt gia tốc cổ chân lớn nhất cùng một pha k % 3, 91–93 %
    năng lượng Δ² ở pha đó (hông không qua IK trải đều 27–43 %). Chuỗi phải là chuỗi service trả về, chưa cắt."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 4 or FPS <= NUT_FPS:            # fps gốc model: quỹ đạo không có đoạn nội suy tuyến tính
        return P
    nut = list(range(0, n, buoc))
    if nut[-1] != n - 1:
        nut.append(n - 1)
    try:
        from scipy.interpolate import CubicSpline
        return CubicSpline(np.array(nut, float), P[nut], axis=0, bc_type="natural")(np.arange(n, dtype=float))
    except ImportError:
        return P


def _nang_tho(frames: list, fps_nguon: float, fps_dich: float) -> list:
    """Khung THÔ ARDY ({t, data: {root, quats xyzw}, contacts}) ở fps gốc → lưới `fps_dich` bằng spline bậc ba C² (natural)
    trên véc-tơ xoay LIÊN TỤC từng khớp + gốc; contacts theo khung gốc gần nhất. Khung gốc giữ nguyên và mang `_nut` = chỉ
    số trong chuỗi gốc (để `lich_su` gửi lại ĐÚNG khung model cho ARDY). `fps_dich <= fps_nguon` thì chỉ đánh dấu nút."""
    if not frames:
        return frames
    fps_nguon, fps_dich = float(fps_nguon), float(fps_dich)
    if fps_dich <= fps_nguon + 1e-6 or len(frames) < 2:
        return [{**f, "_nut": i} for i, f in enumerate(frames)]
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        return [{**f, "_nut": i} for i, f in enumerate(frames)]
    n = len(frames)
    tn = np.arange(n) / fps_nguon
    buoc = int(round(fps_dich / fps_nguon))
    m = (n - 1) * buoc + 1
    tm = np.arange(m) / fps_dich
    root = np.array([f["data"]["root"] for f in frames], float)
    Q = np.array([f["data"]["quats"] for f in frames], float)           # (n, J, 4) xyzw
    J = Q.shape[1]
    R = np.zeros((n, J, 3))
    for j in range(J):
        q_truoc = None
        for i in range(n):
            x, y, z, w = Q[i, j]
            q = np.array([w, x, y, z])
            if q_truoc is not None and float(np.dot(q, q_truoc)) < 0:
                q = -q
            q_truoc = q
            r = _r_log(q)
            if i and np.linalg.norm(r - R[i - 1, j]) > math.pi:
                nr = float(np.linalg.norm(r))
                if nr > 1e-9:
                    r = r * (1.0 - 2.0 * math.pi / nr)
            R[i, j] = r
    Rm = CubicSpline(tn, R, axis=0, bc_type="natural")(tm)
    Pm = CubicSpline(tn, root, axis=0, bc_type="natural")(tm)
    out = []
    for k in range(m):
        i_gan = int(min(n - 1, round(k / buoc)))
        la_nut = k % buoc == 0
        if la_nut:
            data = {"root": list(frames[k // buoc]["data"]["root"]), "quats": [list(v) for v in frames[k // buoc]["data"]["quats"]]}
        else:
            quats = []
            for j in range(J):
                w, x, y, z = _r_exp(Rm[k, j])
                quats.append([float(x), float(y), float(z), float(w)])
            data = {"root": [float(v) for v in Pm[k]], "quats": quats}
        f = {"t": round(float(tm[k]), 4), "data": data, "contacts": frames[i_gan].get("contacts")}
        if la_nut:
            f["_nut"] = k // buoc
        out.append(f)
    return out


def _vi_tri_chan_ardy(named: list, info: dict) -> dict:
    """{LeftFoot/RightFoot/Hips: (T,3)} vị trí world trong không gian ARDY — FK T-pose danh tính (quat local xyzw)."""
    J = list(info["joints"]); PAR = info["parents"]; REST = np.asarray(info["rest_positions"], float)
    IX = {n: i for i, n in enumerate(J)}
    ra = {n: [] for n in ("Hips", "LeftFoot", "RightFoot", "LeftUpLeg", "RightUpLeg")}
    for f in named:
        b = f["data"]["bones"]
        G = [None] * len(J); P = [None] * len(J)
        for i, n in enumerate(J):
            v = b[n]
            x, y, z, w = (v[3:7] if i == 0 else v[0:4])
            Rl = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                           [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                           [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            p = PAR[i]
            if p is None or p < 0:
                G[i] = Rl; P[i] = np.asarray(v[0:3], float)
            else:
                G[i] = G[p] @ Rl; P[i] = P[p] + G[p] @ (REST[i] - REST[p])
        for n in ra:
            ra[n].append(P[IX[n]])
    return {n: np.asarray(v, float) for n, v in ra.items()}


# 20 fps gốc cần vùng mềm RỘNG HƠN: IK chỉ giữ tầm với tại khung gốc, runtime nâng 60 fps bằng spline nội suy TỪNG xương
# độc lập nên giữa hai khung gốc chân duỗi quá mức IK đã giới hạn (dậm chân tại chỗ: 172° ở 20 fps → 178,3° sau nâng,
# gối Δ² 11°/khung²). Cùng seed 15/09 sau nâng 60 fps: 2 cm 11,0 · 3 cm 12,9 · 4 cm 2,5 (kho 60 fps cũ 4,5); đi tới
# 3,4 · 3,2 · 2,9; trôi pha đứng ≤0,5 cm ở mọi mức.
MEM_TAM_VOI_M = float(os.getenv("ARDY_KHO_MEM_TAM_VOI_M", "0.04" if FPS <= 20 else "0.02"))
# IK bám cả ĐỘ CAO bàn chân ARDY (1) hay chỉ vị trí NGANG (0, mặc định — độ cao theo FK retarget). A/B cùng seed
# 15/09 (dậm chân tại chỗ + nhón chân vòng tròn, user "11, 16 vẫn còn giật lúc rời sàn"): bám cả độ cao làm bàn chân
# NẤC lúc rời sàn (nhấc–rơi–nhấc, 1–2 lần/chân; không IK 0) và số lần rời sàn 3–5 → 6–8 — độ cao ARDY khác tỉ lệ chân
# rig; chỉ bám ngang: nấc ≈ không IK, rời sàn về 3–5, trôi pha đứng tốt nhất 0,11–0,38 cm, gối Δ² 2,8–4,6 (≈ không IK).
THEO_CHAN_DOC = os.getenv("ARDY_KHO_THEO_CHAN_DOC", "0") == "1"
# Đích bàn chân = khớp háng rig + k·(bàn chân − khớp háng) của ARDY (1, mặc định) hay neo khung đầu retarget (0, bản cũ).
THEO_CHAN_THE_ARDY = os.getenv("ARDY_KHO_THEO_CHAN_THE_ARDY", "1") == "1"


def _mem_tam_voi(F: dict, D) -> np.ndarray:
    """SOFT IK — kéo đích cổ chân về trong tầm với theo hàm mũ trong `MEM_TAM_VOI_M` cuối (15/09, user: "cổ chân và chân
    khi cách khỏi sàn có hiện tượng giật"). Chân rig dài hơn chân ARDY ~3 % nên đích theo ARDY có lúc vượt tầm với: IK hai
    xương KẸP CỨNG gối ở 178,3° rồi nhả → gối Δ² 13–15°/khung² (không IK 2,4–4,3, A/B cùng seed). Mềm hoá:
    d' = da + s·(1 − e^{−(d−da)/s}) với da = L − s → gối tiến tới duỗi thẳng liên tục, sai lệch ≤ s/e ở đúng tầm với.
    `F` = FK một chân của `buoc_chan._fk` (p_u hông, p_l gối, p_f cổ chân); IK không dời khớp hông nên p_u là tâm đúng."""
    U = np.asarray(F["p_u"], float)
    Lp = np.asarray(F["p_l"], float)
    Fp = np.asarray(F["p_f"], float)
    D = np.asarray(D, float)
    L = float(np.median(np.linalg.norm(U - Lp, axis=1) + np.linalg.norm(Lp - Fp, axis=1)))
    s = MEM_TAM_VOI_M
    da = L - s
    V = D - U
    d = np.linalg.norm(V, axis=1)
    m = d > da
    if not m.any():
        return D
    D2 = D.copy()
    dm = da + s * (1.0 - np.exp(-(d[m] - da) / s))
    D2[m] = U[m] + V[m] * (dm / d[m])[:, None]
    return D2


def theo_chan_ardy(clip: dict, named: list, info: dict, A: dict | None = None) -> dict:
    """IK hai chân rig theo quỹ đạo bàn chân ARDY. Sửa `clip` tại chỗ; trả {sai_cm, k}. `A` = quỹ đạo ARDY đã tính sẵn
    (clip ghép nhiều đoạn), bỏ trống thì FK từ `named`."""
    from loi import buoc_chan as BC
    lu = BC._luoi(clip)
    if lu is None or (not named and A is None):
        return {}
    b, moc = lu
    n = len(moc)
    if A is None:
        A = {x: _muot_quy_dao(v) for x, v in _vi_tri_chan_ardy(named, info).items()}
    if len(A["Hips"]) != n:
        return {"bo_qua": f"số khung ARDY {len(A['Hips'])} ≠ rig {n}"}
    hp = clip.get("hips_pos") or []
    # hệ số = đúng hệ số retarget đã dùng cho hông (đọc lại từ chính clip: độ dời hông rig / độ dời gốc ARDY)
    dA = A["Hips"] - A["Hips"][0]
    dR = np.array([f[1:4] for f in hp], float) if len(hp) == n else None
    k = float(np.sum(dR * dA) / max(np.sum(dA * dA), 1e-9)) if dR is not None and np.sum(dA * dA) > 1e-6 else 1.0
    ks = np.arange(n)
    fk = BC._fk(clip, b, moc, ks)
    goc = {c: np.asarray(fk[c]["p_f"], float)[0] for c in BC.CHAN}
    huong = {c: np.asarray(fk[c]["qw_f"], float) for c in BC.CHAN}
    if THEO_CHAN_THE_ARDY and all(f"{c[:-4]}UpLeg" in A for c in BC.CHAN):
        # THẾ ĐỨNG THEO ARDY (15/09, Playwright nút 12): bàn chân rig = khớp háng RIG + k·(bàn chân ARDY − khớp háng ARDY).
        # Bản neo khung đầu (dưới) giữ nguyên độ dạng chân của tư thế nghỉ Mira: kho đứng rộng 38–41 cm trong khi ARDY
        # 26–30 và idle client 25 → vào clip hai bàn chân lê ra 13 cm trong 0,25 s trên trình duyệt.
        dich = {c: np.asarray(fk[c]["p_u"], float) + k * (A[c] - A[f"{c[:-4]}UpLeg"]) for c in BC.CHAN}
    else:
        dich = {c: goc[c] + k * (A[c] - A[c][0]) for c in BC.CHAN}
    if not THEO_CHAN_DOC:
        # CHỈ BÁM NGANG: chỗ đặt chân (x, z) theo ARDY, độ cao cổ chân giữ đúng FK retarget
        for c in BC.CHAN:
            dich[c][:, 1] = np.asarray(fk[c]["p_f"], float)[:, 1]
    dich = {c: _mem_tam_voi(fk[c], dich[c]) for c in BC.CHAN}

    fps_c = float(clip.get("fps") or FPS)           # fps CỦA CLIP, không phải hằng module (kho 20 fps / clip thử 60 fps)

    def quy_dao(c, t):
        i = int(min(n - 1, max(0, round(float(t) * fps_c))))
        return dich[c][i], huong[c][i]

    sai = BC._giai_ik(clip, b, moc, ks, np.ones(n), quy_dao)
    clip["_theo_chan_ardy"] = {"sai_cm": round(sai * 100, 2), "k": round(k, 3)}
    return clip["_theo_chan_ardy"]


# MẶC ĐỊNH TẮT (15/09): A/B cùng seed clip dậm chân tại chỗ + nhón chân vòng tròn — nhấc không bớt được lần nấc nào
# lúc rời sàn mà làm gối giật thêm (nhón chân, chân phải gối Δ² 2,4 → 8,0°/khung²).
NHAC_CHAN = os.getenv("ARDY_KHO_NHAC_CHAN", "0") == "1"
NC_V = 0.15        # m/s — bàn chân dời ngang nhanh hơn thế là đang BƯỚC (không phải đứng/xoay gót)
NC_DIST = 0.04     # m — quãng bước tối thiểu mới xét
NC_HE = 0.40       # độ nhấc cần = 40 % quãng bước …
NC_MIN, NC_MAX = 0.03, 0.08   # … kẹp 3–8 cm
NC_MEP = 3         # khung mở rộng mỗi đầu cửa sổ nhấc


def nhac_chan_luot(clip: dict) -> dict:
    """ARDY đôi khi sinh BƯỚC LÊ: contacts báo chân rời sàn, bàn chân dời 15–25 cm trong 0,1–0,3 s mà cổ chân không
    nhấc (đo 15/09: xoay gót, quay người — có trong đầu ra thô, bật/tắt postprocess như nhau). Trên màn hình đó là TRƯỢT.
    Mỗi đoạn bàn chân đi nhanh (> NC_V) có độ nhấc thật < NC_HE × quãng thì nâng cổ chân theo cung sin tới đủ độ nhấc,
    giữ nguyên đường ngang + hướng bàn chân, giải IK hai xương. Sửa `clip` tại chỗ; trả sổ các đoạn đã nhấc."""
    from loi import buoc_chan as BC
    lu = BC._luoi(clip)
    if lu is None:
        return {}
    b, moc = lu
    n = len(moc)
    if n < 8:
        return {}
    ks = np.arange(n)
    fk = BC._fk(clip, b, moc, ks)
    P = {c: np.asarray(fk[c]["p_f"], float) for c in BC.CHAN}
    Q = {c: np.asarray(fk[c]["qw_f"], float) for c in BC.CHAN}
    t = np.asarray(moc, float)
    san = min(float(P[c][:, 1].min()) for c in BC.CHAN)
    W, DICH, so = {}, {}, []
    for c in BC.CHAN:
        X = P[c]
        cao = X[:, 1] - san
        v = np.zeros(n)
        v[1:-1] = np.linalg.norm(X[2:, [0, 2]] - X[:-2, [0, 2]], axis=1) / np.maximum(t[2:] - t[:-2], 1e-6)
        v[0], v[-1] = v[1], v[-2]
        di = v > NC_V
        # gộp khe ≤ 3 khung
        i = 0
        doan = []
        while i < n:
            if not di[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and (di[j + 1] or (j + 4 < n and di[j + 2:j + 5].any())):
                j += 1
            doan.append((i, j))
            i = j + 1
        w = np.zeros(n)
        dich = X.copy()
        for i, j in doan:
            dist = float(np.linalg.norm(X[j, [0, 2]] - X[i, [0, 2]]))
            if dist < NC_DIST:
                continue
            can = float(np.clip(NC_HE * dist, NC_MIN, NC_MAX))
            co = float(cao[i:j + 1].max() - max(cao[i], cao[j]))
            them = can - co
            if them <= 0.005:
                continue
            a, z = max(0, i - NC_MEP), min(n - 1, j + NC_MEP)
            s = (np.arange(a, z + 1) - a) / max(z - a, 1)
            dich[a:z + 1, 1] += them * np.sin(np.pi * s)
            w[a:z + 1] = 1.0
            so.append({"chan": c, "tu": round(float(t[a]), 2), "toi": round(float(t[z]), 2),
                       "quang_cm": round(dist * 100, 1), "nhac_cm": round(them * 100, 1)})
        W[c], DICH[c] = w, _mem_tam_voi(fk[c], dich)
    if not so:
        clip["_nhac_chan"] = []
        return {"doan": []}

    fps_c = float(clip.get("fps") or FPS)

    def quy_dao(c, tt):
        k = int(min(n - 1, max(0, round(float(tt) * fps_c))))
        return DICH[c][k], Q[c][k]

    sai = BC._giai_ik(clip, b, moc, ks, W, quy_dao)
    clip["_nhac_chan"] = so
    return {"doan": so, "sai_cm": round(sai * 100, 2)}


def lich_su(tho: list, k_cuoi: int | None = None, n: int = 8) -> list:
    """Đuôi chuyển động làm `history` cho ARDY. Khung đã qua `_nang_tho` thì chỉ gửi KHUNG GỐC (`_nut`) cách nhau 1/FPS_ARDY —
    bài báo ARDY: lịch sử là chính các khung model vừa sinh, không phải khung nội suy."""
    fr = tho[:k_cuoi + 1] if k_cuoi is not None else tho
    if fr and any("_nut" in f for f in fr):
        fr = [f for f in fr if "_nut" in f][-n:]
        return [{"t": round(i / FPS_ARDY, 4), "data": fr[i]["data"]} for i in range(len(fr))]
    fr = fr[-n:]
    return [{"t": round(i / FPS, 4), "data": fr[i]["data"]} for i in range(len(fr))]


def idle_dong_bang(kho: Path) -> tuple[list, dict, dict]:
    """(history idle, target idle, clip idle đã retarget) — sinh MỘT lần, lưu `_idle_tho.json`."""
    p = kho / "_idle_tho.json"
    if p.exists():
        tho = json.loads(p.read_text())
    else:
        tho, _c, _m, _s = _sinh(CAU_IDLE, 4.0)
        kho.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(tho))
        print(f"idle đóng băng: {len(tho)} khung thô → {p}")
    from app.modules.body_motion.ardy_retarget import retarget, tu_mang
    info = _info()
    if not any("_nut" in f for f in tho) and len(tho) >= int(3.5 * 60):
        # `_idle_tho.json` đóng băng từ đường 60 fps cũ (service nội suy): lấy đúng KHUNG NÚT model (k % 3 == 0 của chuỗi
        # service trả về) — cùng một idle, không sinh lại — rồi nâng lên lưới kho bằng spline như mọi clip
        tho = _nang_tho(tho[::3], FPS_ARDY, FPS)
    clip = retarget(info, tu_mang(list(info["joints"]), tho), FPS, nguon="ardy")
    if LAM_TRON:
        lam_tron_nut(clip)
    return lich_su(tho), {"root": tho[-1]["data"]["root"], "quats": tho[-1]["data"]["quats"]}, clip


# ---- render một clip ---------------------------------------------------------------------------
def render_mot(tag: str, prompt: str, giay: float, hist: list, dich: dict, idle_clip: dict) -> dict:
    """Trả {"clip", "meta"} hoặc {"loi": ...}."""
    from app.modules.body_motion import ardy_song
    k_idle = len(idle_clip["bones"]["Hips"]) - 1
    idle_pose = tu_the(idle_clip, k_idle)
    t0 = time.time()
    tho, clip, khop, sim = _sinh(prompt, giay, hist, theo=False)
    if not clip:
        return {"loi": "service không trả clip"}
    # khung 0 phải trùng idle (history là điều kiện cứng) — không trùng là idle đóng băng đã đổi
    d0 = lech(tu_the(clip, 0), idle_pose)[0]
    ct = diem_cat_dong_tac(clip, idle_pose)
    if ct["k_cat"] is None:
        return {"loi": f"không có động tác (khớp '{khop}' {sim:.2f})", "khop": khop, "sim": sim}
    k_cat = ct["k_cat"]
    if tho and "_nut" in tho[0]:
        # cắt ĐÚNG khung model gốc: `lich_su` chỉ gửi khung gốc cho lối ra, cắt ở khung nội suy thì lối ra nối từ khung gốc
        # trước đó tới 2 khung 60 fps
        buoc_nut = max(1, int(round(FPS / FPS_ARDY)))
        k_cat = max(1, k_cat - k_cat % buoc_nut)
    tho2, ra, _m2, _s2 = _sinh(CAU_LOI_RA, GIAY_LOI_RA, lich_su(tho, k_cat), dich, CFG_LOI_RA, theo=False)
    if not ra:
        return {"loi": "lối ra không có clip"}
    noi = lech(tu_the(ra, 0), tu_the(clip, k_cat))[0]
    k_lang = diem_lang_loi_ra(ra, idle_pose)
    het_lang = k_lang is None
    if k_lang is None:
        k_lang = len(ra["bones"]["Hips"]) - 1
    k_dau = diem_dau(ct["do"]["dd"], int(ct["onset"]))
    hc = lap_ghep(clip, k_dau, k_cat, ra, k_lang, idle_clip, k_idle)
    if THEO_CHAN or NHAC_CHAN or THE_NGHI_S > 0:
        _chan_clip_ghep(hc, tho, k_dau, k_cat, tho2, k_lang, toan_than=toan_than_theo_noi_dung(ct["do"]))
    n = len(hc["bones"]["Hips"])
    dinh_hc = max(lech(tu_the(hc, k), idle_pose)[0] for k in range(0, n, 3))
    dinh_goc = float(ct["do"]["dd"].max())
    # khung cuối đo THÂN TRÊN: chân KHÔNG trộn về idle (user chốt 12/09 "clip thì chân tự do"), gộp chân vào thì
    # số đo bị chi phối bởi thế đứng của chính động tác và cổng kiểm định kỳ mất nghĩa
    _tren = lambda p: {x: v for x, v in p.items() if x not in CHAN}  # noqa: E731
    # mốc "idle" của đầu/cuối = THẾ NGHỈ CLIENT khi `khop_the_nghi` bật (idle ARDY đóng băng lệch thế client ~53° tay —
    # đo vào idle ARDY thì mọi clip mới "lệch 53°" và lượt kiểm định kỳ `KIEM_CUOI_MAX` tự dừng lô oan)
    A_ng = the_nghi_client() if THE_NGHI_S > 0 else None
    pose_nghi = ({x: (_hips_khong_yaw(e) if x == "Hips" else _q(e)) for x, e in A_ng["bones"].items()
                  if not any(g in x for g in NGON)} if A_ng else idle_pose)
    cuoi = lech(_tren(tu_the(hc, n - 1)), _tren(pose_nghi))[0]
    sd = ardy_song.do(hc)
    song = ardy_song.song(hc)
    meta = {"tag": tag, "prompt": prompt, "khop": khop, "sim": round(sim, 3), "giay_xin": giay, "dai_s": hc["duration_s"],
            "loai": ct["loai"], "onset_s": round(ct["onset"] / FPS, 2), "dinh1_s": round(ct["dinh1"] / FPS, 2),
            "cat_s": round(k_cat / FPS, 2), "cua_so_bac": ct["cua_so_bac"], "lap_sau": ct["lap_sau"],
            "khung0_lech_idle": round(d0, 1), "dau_lech_idle": round(float(ct["do"]["dd"][k_dau]), 1),
            "noi_loi_ra": round(noi, 1), "loi_ra_lang_s": None if het_lang else round(k_lang / FPS, 2),
            "dinh_do": round(dinh_hc, 1), "dinh_goc": round(dinh_goc, 1), "cuoi_lech_idle": round(cuoi, 2),
            "quay_tong": round(float(ct["do"]["yw"][k_cat]), 1), "toan_than": toan_than_theo_noi_dung(ct["do"]),
            "di_m": round(float((ct["do"]["xz"].max(axis=0) - ct["do"]["xz"].min(axis=0)).max()), 3),
            "quay_bien_do": round(float(ct["do"]["yw"].max() - ct["do"]["yw"].min()), 1),
            "song": bool(song), "song_do": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sd.items()},
            "sinh_s": round(time.time() - t0, 1)}
    if d0 > 5.0:
        return {"loi": f"khung 0 lệch idle {d0:.1f}° — idle đóng băng khác lịch sử?", "meta": meta}
    if not song:
        return {"loi": "cổng sống: không có động tác thật", "meta": meta}
    if dinh_hc < 0.6 * dinh_goc:
        return {"loi": f"đỉnh sau cắt {dinh_hc:.0f}° < 60 % bản gốc {dinh_goc:.0f}°", "meta": meta}
    if hc["duration_s"] > DAI_TOI_DA:
        return {"loi": f"dài {hc['duration_s']:.1f}s > {DAI_TOI_DA}", "meta": meta}
    # NGẮN HƠN CỬA SỔ TRỘN = KHÔNG CÓ ĐOẠN TRỘN NÀO (13/09). `lap_ghep` bỏ qua trộn khi `N <= nb + 1` để khỏi
    # vỡ chỉ số — an toàn cho máy nhưng clip xuất ra KHÔNG về idle, phá đúng bất biến mà cả kho dựa vào.
    # Đo trên 5.904 clip sinh trong ngày: đúng 2 cái hỏng đuôi (9,42° và 9,11°), cả hai dài 0,42–0,45 s, còn
    # 5.884 clip từ 1 s trở lên thì 0 cái hỏng. Mảnh 0,4 s cũng không phải động tác dùng được — loại thẳng.
    if hc["duration_s"] <= TRON_S:
        return {"loi": f"dài {hc['duration_s']:.2f}s ≤ cửa sổ trộn {TRON_S}s — đuôi không kịp về idle", "meta": meta}
    hc["source"] = f"ardy_kho:{prompt}"
    hc["_khop"], hc["_similarity"] = khop, round(sim, 3)
    hc["_kho_toan_than"] = meta["toan_than"]
    return {"clip": hc, "meta": meta}


def render_tag(tag: str, prompt: str, kho: Path = KHO, bien_the: int = 0) -> dict | None:
    """Render MỘT biến thể cho `tag` rồi ghi vào kho (dùng cho 8010 tự bổ sung tag thiếu lúc chạy). Trả mục chỉ mục
    hoặc None (trượt cổng/hỏng — đã ghi `kho_loi.jsonl`)."""
    kho = Path(kho)
    hist, dich, idle_clip = idle_dong_bang(kho)
    giay = GIAY_BIEN_THE[bien_the % len(GIAY_BIEN_THE)]
    try:
        r = render_mot(tag, prompt, giay, hist, dich, idle_clip)
    except Exception as e:  # noqa: BLE001
        r = {"loi": f"hỏng: {str(e)[:120]}"}
    if "clip" not in r:
        with (kho / "kho_loi.jsonl").open("a") as f:
            f.write(json.dumps({"tag": tag, "bien_the": bien_the, "prompt": prompt, "tu_bo_sung": True,
                                **{k: v for k, v in r.items() if k != "clip"}}, ensure_ascii=False) + "\n")
        return None
    tep = _ten_tep(tag, bien_the)
    (kho / (tep + ".tmp")).write_text(json.dumps(r["clip"]))
    (kho / (tep + ".tmp")).replace(kho / tep)
    m = {"tep": tep, "bien_the": bien_the, "tu_bo_sung": True, **r["meta"]}
    cu = [x for x in doc_chi_muc(kho).get(tag, []) if x.get("bien_the") != bien_the]
    ghi_chi_muc(kho, {tag: sorted(cu + [m], key=lambda x: x["bien_the"])}, [tag])
    return m


def kiem_dinh_ky(metas: list, so_loi: int) -> tuple[bool, str]:
    """Chấm lô `metas` vừa render (meta của chỉ mục, không mở lại tệp). Trả (đạt, mô tả một dòng).

    Ba đại lượng đủ bắt mọi kiểu hỏng đã gặp: khung ĐẦU xa idle (luật cắt đầu sai — lô 12/09), khung CUỐI thân trên
    xa idle (lối ra/trộn hỏng), tỉ lệ TRƯỢT CỔNG (service hỏng, hoặc lô câu toàn tư thế tĩnh).
    """
    import statistics as st
    if not metas:
        return True, "lô rỗng"
    dau = [float(m.get("dau_lech_idle") or 0.0) for m in metas]
    cuoi = [float(m.get("cuoi_lech_idle") or 0.0) for m in metas]
    dai = [float(m.get("dai_s") or 0.0) for m in metas]
    ti_loi = so_loi / max(1, so_loi + len(metas))
    mo_ta = (f"{len(metas)} clip · đầu so idle {st.median(dau):.1f}° (max {max(dau):.1f}) · "
             f"cuối {st.median(cuoi):.2f}° (max {max(cuoi):.2f}) · dài {st.median(dai):.1f}s · "
             f"trượt cổng {ti_loi * 100:.0f}%")
    if max(dau) > KIEM_DAU_MAX:
        return False, f"{mo_ta} → ĐẦU CLIP không về idle (>{KIEM_DAU_MAX}°)"
    if max(cuoi) > KIEM_CUOI_MAX:
        return False, f"{mo_ta} → ĐUÔI CLIP không về idle (>{KIEM_CUOI_MAX}°)"
    if ti_loi > KIEM_LOI_MAX:
        return False, f"{mo_ta} → TRƯỢT CỔNG quá nhiều (>{KIEM_LOI_MAX * 100:.0f}%)"
    return True, mo_ta


def _ten_tep(tag: str, bt: int) -> str:
    return f"{tag}__{bt}.json"


def doc_chi_muc(kho: Path) -> dict:
    p = kho / "kho.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def ghi_chi_muc(kho: Path, cm: dict, tags=None) -> dict:
    """Ghi chỉ mục bằng cách HỢP NHẤT vào bản trên đĩa dưới khoá tệp: CLI render lô và 8010 tự bổ sung tag thiếu
    (`loi/kho_ardy.Kho.bo_sung`) có thể cùng ghi — ai cũng chỉ thay đúng các tag mình vừa render. Trả bản đã hợp nhất."""
    import fcntl
    kho.mkdir(parents=True, exist_ok=True)
    with (kho / "kho.lock").open("a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            dia = doc_chi_muc(kho)
            for t in (tags if tags is not None else list(cm)):
                if t in cm:
                    dia[t] = cm[t]
            tmp = kho / "kho.json.tmp"
            tmp.write_text(json.dumps(dia, ensure_ascii=False, indent=1))
            tmp.replace(kho / "kho.json")
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    return dia


def danh_sach_tag(nhom: str) -> list[str]:
    from app.modules.body_motion import ardy_cue
    if nhom == "day":
        from loi.ham_ardy import ds_day_llm
        return ds_day_llm()
    if nhom == "bang":
        from loi.ham_ardy import ds_day_llm
        day = ds_day_llm()
        return list(dict.fromkeys(day + list(ardy_cue.bang_prompt().keys())))
    if nhom == "loi":                # render lại các tag đã trượt (kho_loi.jsonl) sau khi sửa luật
        p = KHO / "kho_loi.jsonl"
        ra = []
        for dong in (p.read_text().splitlines() if p.exists() else []):
            try:
                ra.append(json.loads(dong)["tag"])
            except Exception:  # noqa: BLE001
                pass
        return list(dict.fromkeys(ra))
    raise SystemExit(f"--nhom {nhom!r} không biết (day | bang | loi)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tags", default="", help="tag,tag,… (không có = theo --nhom)")
    ap.add_argument("--cau", default="", help="tệp văn bản, mỗi dòng một CÂU tiếng Anh của catalog → render thẳng câu đó (tag = câu viết snake_case)")
    ap.add_argument("--nhom", default="day", help="day = 47 tag dạy LLM · bang = cả ardy_prompts.json")
    ap.add_argument("--bien-the", type=int, default=1, help=f"số biến thể mỗi tag (độ dài xin {GIAY_BIEN_THE})")
    ap.add_argument("--lam-lai", action="store_true", help="render lại cả clip đã có")
    ap.add_argument("--tu-kho", action="store_true",
                    help="lấy (tag, câu, số biến thể) từ CHỈ MỤC kho hiện có và render lại đúng vậy — sửa lô cũ")
    ap.add_argument("--phan", default="", help="chia phần 'i/N': giữ tag thứ i, i+N, i+2N… (chạy song song nhiều tiến trình)")
    ap.add_argument("--kiem-moi", type=int, default=KIEM_MOI, help="kiểm định kỳ sau mỗi ngần này clip (0 = tắt)")
    ap.add_argument("--kho", default=str(KHO))
    a = ap.parse_args(argv)
    from app.modules.body_motion import ardy_cue
    kho = Path(a.kho)
    kho.mkdir(parents=True, exist_ok=True)
    hist, dich, idle_clip = idle_dong_bang(kho)
    cau_theo_tag: dict[str, str] = {}
    so_bt: dict[str, int] = {}
    if a.tu_kho:
        cm0 = doc_chi_muc(kho)
        cau_theo_tag = {t: v[0]["prompt"] for t, v in cm0.items() if v and v[0].get("prompt")}
        so_bt = {t: len(cm0[t]) for t in cau_theo_tag}
        tags = list(cau_theo_tag)
        # ARDY_KHO_CHI_XAU=1: chỉ render lại tag THẬT SỰ hỏng (khung đầu > KIEM_DAU_MAX hoặc chưa có số đo),
        # thay vì quét cả chỉ mục. 13/09: kho 22.119 clip nhưng chỉ 2.756 cái lệch mốc đầu — quét hết là làm lại
        # 8 phần việc đã tốt để sửa 1. Đo trên tệp bằng `kiem_tai_ve.py` cho cùng kết luận: lô cũ hỏng KHUNG ĐẦU
        # (trung vị 8,4°, max 115°), đuôi thì 0,00° cả lô cũ lẫn lô mới.
        if os.getenv("ARDY_KHO_CHI_XAU") == "1":
            def _xau(v: list) -> bool:
                # MỌI biến thể, không chỉ `v[0]`. Bản đầu (13/09 chiều) chỉ soi biến thể đầu nên tag có
                # `__0` lành mà `__1/__2` hỏng thì cả tag bị loại khỏi việc — quét tệp tối hôm đó còn đúng
                # 5 clip lọt lưới (bow_greeting __1/__2, open_palm __1, cross_arms __2, và một tag mà thứ
                # tự mục trong chỉ mục không trùng thứ tự hậu tố). Lọc theo phần tử ĐẠI DIỆN của một nhóm
                # là mù với phần còn lại của nhóm.
                if not v:
                    return True
                for m in v:
                    d = m.get("dau_lech_idle")
                    if d is None or float(d) > KIEM_DAU_MAX:
                        return True
                return False
            tags = [t for t in tags if _xau(cm0.get(t) or [])]
            print(f"chỉ render lại tag hỏng: {len(tags)}/{len(cau_theo_tag)} (khung đầu > {KIEM_DAU_MAX}°)", flush=True)
        a.lam_lai = True
    elif a.cau:
        import re as _re
        for dong in Path(a.cau).read_text(encoding="utf-8").splitlines():
            s = dong.strip()
            if s and not s.startswith("#"):
                cau_theo_tag[_re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:80]] = s
        tags = list(cau_theo_tag)
    else:
        tags = [t.strip() for t in a.tags.split(",") if t.strip()] or danh_sach_tag(a.nhom)
    if a.phan:
        _i, _n = (int(x) for x in a.phan.split("/"))
        tags = tags[_i::_n]
        print(f"phần {_i}/{_n}: {len(tags)} tag", flush=True)
    cm = doc_chi_muc(kho)
    loi_p = kho / "kho_loi.jsonl"
    n_ok = n_loi = n_bo = 0
    gan_day: list = []                 # meta của lô đang chờ kiểm định kỳ
    loi_moc = 0                        # số trượt tại lần kiểm trước
    t_bd = time.time()
    for i, tag in enumerate(tags):
        if tag in cau_theo_tag:
            prompt = cau_theo_tag[tag]
        else:
            d = ardy_cue.don(tag, False, None)
            if not d:
                n_bo += 1
                print(f"[{i + 1}/{len(tags)}] {tag}: không dựng được câu — bỏ", flush=True)
                continue
            prompt = d[0]
        muc = [m for m in cm.get(tag, []) if (kho / m["tep"]).exists()] if not a.lam_lai else []
        for bt in range(so_bt.get(tag, a.bien_the)):
            if any(m.get("bien_the") == bt for m in muc):
                continue
            giay = GIAY_BIEN_THE[bt % len(GIAY_BIEN_THE)]
            try:
                r = render_mot(tag, prompt, giay, hist, dich, idle_clip)
            except Exception as e:  # noqa: BLE001
                r = {"loi": f"hỏng: {str(e)[:120]}"}
            if "clip" not in r:
                n_loi += 1
                with loi_p.open("a") as f:
                    f.write(json.dumps({"tag": tag, "bien_the": bt, "prompt": prompt, **{k: v for k, v in r.items() if k != "clip"}},
                                       ensure_ascii=False) + "\n")
                print(f"[{i + 1}/{len(tags)}] {tag} bt{bt}: TRƯỢT — {r['loi']}", flush=True)
                continue
            tep = _ten_tep(tag, bt)
            (kho / (tep + ".tmp")).write_text(json.dumps(r["clip"]))      # ghi nguyên tử: vòng đẩy R2 chạy song song
            (kho / (tep + ".tmp")).replace(kho / tep)
            m = {"tep": tep, "bien_the": bt, **r["meta"]}
            muc = [x for x in muc if x.get("bien_the") != bt] + [m]
            cm[tag] = sorted(muc, key=lambda x: x["bien_the"])
            ghi_chi_muc(kho, cm, [tag])
            n_ok += 1
            print(f"[{i + 1}/{len(tags)}] {tag} bt{bt}: {m['dai_s']:.2f}s {m['loai']} cắt {m['cat_s']}s đỉnh {m['dinh_do']}° "
                  f"cuối {m['cuoi_lech_idle']}° nối lối ra {m['noi_loi_ra']}° quay {m['quay_tong']:+.0f}° "
                  f"{'TOÀN THÂN ' if m['toan_than'] else ''}({m['sinh_s']}s)", flush=True)
            gan_day.append(m)
            if a.kiem_moi and len(gan_day) >= a.kiem_moi:
                dat, mo_ta = kiem_dinh_ky(gan_day, n_loi - loi_moc)
                print(f"  KIỂM ĐỊNH KỲ: {mo_ta}", flush=True)
                if not dat:
                    print("  DỪNG — lô lệch chuẩn, sửa rồi chạy lại; clip đã render giữ nguyên", flush=True)
                    return 2
                gan_day.clear()
                loi_moc = n_loi
    if a.kiem_moi and gan_day:
        print(f"KIỂM cuối: {kiem_dinh_ky(gan_day, n_loi - loi_moc)[1]}", flush=True)
    print(f"\nxong: {n_ok} clip mới · {n_loi} trượt (kho_loi.jsonl) · {n_bo} tag bỏ · {time.time() - t_bd:.0f}s · "
          f"kho {sum(len(v) for v in cm.values())} clip / {len(cm)} tag → {kho}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
