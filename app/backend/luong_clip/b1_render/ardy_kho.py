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
FPS = 60
CAU_IDLE = "A person stands still."
CAU_LOI_RA = "A person stands still with arms relaxed at the sides."
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


def lap_ghep(clip: dict, k_dau: int, k_cat: int, ra: dict, k_lang: int, idle_clip: dict, k_idle: int = 0) -> dict:
    """Luật 5–6: [k_dau..k_cat] của động tác + [0..k_lang] của lối ra, hips_pos lối ra dời tiếp theo khung cắt
    (retarget trừ root0 từng clip), rồi trộn TRON_S cuối về khung idle cho thân trên + xoay hông (chân giữ nguyên)."""
    bones = {}
    for x, tr in clip["bones"].items():
        a = [list(f[1:4]) for f in tr[k_dau:k_cat + 1]]
        b = [list(f[1:4]) for f in ra["bones"][x][:k_lang + 1]] if x in ra["bones"] else []
        bones[x] = a + b
    N = len(bones["Hips"])
    tron_ve_idle(bones, idle_clip, k_idle, FPS, cuoi=True)
    hpa = [list(f[1:4]) for f in (clip.get("hips_pos") or [])[k_dau:k_cat + 1]]
    hpb = [list(f[1:4]) for f in (ra.get("hips_pos") or [])[:k_lang + 1]]
    if hpa and hpb:
        goc = hpa[-1]
        hpb = [[goc[i] + f[i] for i in range(3)] for f in hpb]
    hp = hpa + hpb
    return {"fps": FPS, "duration_s": round((N - 1) / FPS, 4),
            "bones": {x: [[round(k / FPS, 4), *[round(float(v), 5) for v in tr[k]]] for k in range(len(tr))] for x, tr in bones.items()},
            "hips_pos": [[round(k / FPS, 4), *[round(float(v), 5) for v in f]] for k, f in enumerate(hp)]}


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


def _sinh(prompt: str, giay: float, history=None, dich=None, cfg_c=None):
    from app.modules.body_motion.ardy_retarget import retarget, tu_mang
    info = _info()
    than = {"text": prompt, "seconds": float(giay), "fps": FPS, "chunk_seconds": max(float(giay), 2.5)}
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
    clip = (retarget(info, tu_mang(list(c0.get("joints") or info["joints"]), frames), float(c0.get("fps") or FPS), nguon="ardy")
            if frames else None)
    return frames, clip, str(c0.get("matched") or ""), float(c0.get("similarity") or 0.0)


def lich_su(tho: list, k_cuoi: int | None = None, n: int = 8) -> list:
    fr = tho[:k_cuoi + 1] if k_cuoi is not None else tho
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
    clip = retarget(info, tu_mang(list(info["joints"]), tho), FPS, nguon="ardy")
    return lich_su(tho), {"root": tho[-1]["data"]["root"], "quats": tho[-1]["data"]["quats"]}, clip


# ---- render một clip ---------------------------------------------------------------------------
def render_mot(tag: str, prompt: str, giay: float, hist: list, dich: dict, idle_clip: dict) -> dict:
    """Trả {"clip", "meta"} hoặc {"loi": ...}."""
    from app.modules.body_motion import ardy_song
    k_idle = len(idle_clip["bones"]["Hips"]) - 1
    idle_pose = tu_the(idle_clip, k_idle)
    t0 = time.time()
    tho, clip, khop, sim = _sinh(prompt, giay, hist)
    if not clip:
        return {"loi": "service không trả clip"}
    # khung 0 phải trùng idle (history là điều kiện cứng) — không trùng là idle đóng băng đã đổi
    d0 = lech(tu_the(clip, 0), idle_pose)[0]
    ct = diem_cat_dong_tac(clip, idle_pose)
    if ct["k_cat"] is None:
        return {"loi": f"không có động tác (khớp '{khop}' {sim:.2f})", "khop": khop, "sim": sim}
    k_cat = ct["k_cat"]
    tho2, ra, _m2, _s2 = _sinh(CAU_LOI_RA, GIAY_LOI_RA, lich_su(tho, k_cat), dich, CFG_LOI_RA)
    if not ra:
        return {"loi": "lối ra không có clip"}
    noi = lech(tu_the(ra, 0), tu_the(clip, k_cat))[0]
    k_lang = diem_lang_loi_ra(ra, idle_pose)
    het_lang = k_lang is None
    if k_lang is None:
        k_lang = len(ra["bones"]["Hips"]) - 1
    k_dau = diem_dau(ct["do"]["dd"], int(ct["onset"]))
    hc = lap_ghep(clip, k_dau, k_cat, ra, k_lang, idle_clip, k_idle)
    n = len(hc["bones"]["Hips"])
    dinh_hc = max(lech(tu_the(hc, k), idle_pose)[0] for k in range(0, n, 3))
    dinh_goc = float(ct["do"]["dd"].max())
    # khung cuối đo THÂN TRÊN: chân KHÔNG trộn về idle (user chốt 12/09 "clip thì chân tự do"), gộp chân vào thì
    # số đo bị chi phối bởi thế đứng của chính động tác và cổng kiểm định kỳ mất nghĩa
    _tren = lambda p: {x: v for x, v in p.items() if x not in CHAN}  # noqa: E731
    cuoi = lech(_tren(tu_the(hc, n - 1)), _tren(idle_pose))[0]
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
