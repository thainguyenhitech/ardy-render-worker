"""GHÉP NỀN VỚI ĐOẠN ARDY TRÊN MỘT TRỤC THỜI GIAN — đè, KHÔNG trộn.

LUẬT SỐ 1 của project (xem README): mỗi khung thuộc về ĐÚNG MỘT nguồn. Trong khoảng một đoạn
ARDY chạy, nền transformer bị ghi đè hoàn toàn; ngoài khoảng đó, nền là nguồn duy nhất. Không
có trọng số, không có dốc, không cắt clip ARDY.

VÌ SAO (đo 09/09 trên `backend/`): tầng trộn cũ slerp giữa tư thế "đang đứng" của nền và tư thế
"giữa sải bước" của ARDY suốt dốc 0,30 s. Tư thế lai đó có bàn chân KHÔNG nằm ở điểm chạm của
bên nào, trong khi `hips_pos` đã đi được nửa đường — bàn chân quét sàn. ARDY sinh gốc và góc
khớp NHẤT QUÁN với nhau; giữ nguyên cả cụm thì tiếp xúc của nó sống sót.

CÁI GIÁ PHẢI TRẢ, CÓ CHỦ Ý: hai mép đoạn ARDY có bước nhảy (vị trí liên tục nhưng vận tốc thì
không). Ở bước này ta ĐỂ NGUYÊN cho thấy, rồi xử bằng một tầng riêng có tên và đo được — chứ
không giấu nó vào trọng số trộn như bản cũ.

Mọi phép lấy mẫu đi qua QUATERNION (CLAUDE.md §4 bẫy 0/1: nội suy tuyến tính trên Euler đi sai
trắc địa của phép quay và nhảy nhánh ±π).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

FPS = 60


# ---- quaternion (w, x, y, z) — cùng quy ước với backend/luong_clip/b1_render/build_clip ----
def _euler_sang_q(ex: float, ey: float, ez: float):
    """Euler XYZ intrinsic (quy ước Three.js) → quaternion (w,x,y,z)."""
    cx, sx = math.cos(ex / 2), math.sin(ex / 2)
    cy, sy = math.cos(ey / 2), math.sin(ey / 2)
    cz, sz = math.cos(ez / 2), math.sin(ez / 2)
    return (cx * cy * cz - sx * sy * sz,
            sx * cy * cz + cx * sy * sz,
            cx * sy * cz - sx * cy * sz,
            cx * cy * sz + sx * sy * cz)


def _q_sang_euler(q, gan=None):
    """Quaternion → Euler XYZ. `gan` = bộ ba khung TRƯỚC, để chọn nhánh LIÊN TỤC.

    Không có `gan` thì hàm trả nhánh chính, và track ghi ra có thể nhảy ±π giữa hai khung —
    đúng bẫy đã trả giá nhiều lần (§2b "TRACK EULER GHI RA PHẢI LIÊN TỤC").
    """
    w, x, y, z = q
    m02 = 2 * (x * z + w * y)
    m02 = max(-1.0, min(1.0, m02))
    ey = math.asin(m02)
    if abs(m02) < 0.999999:
        ex = math.atan2(-2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
        ez = math.atan2(-2 * (x * y - w * z), 1 - 2 * (y * y + z * z))
    else:                                   # suy biến: gộp hai trục còn lại
        ex = math.atan2(2 * (y * z + w * x), 1 - 2 * (x * x + z * z))
        ez = 0.0
    if gan is None:
        return ex, ey, ez
    # hai nhánh Euler tương đương: (ex, ey, ez) và (ex±π, π−ey, ez±π). Chọn nhánh gần khung
    # trước, rồi bỏ bội 2π còn lại.
    kia = (ex + math.pi, math.pi - ey, ez + math.pi)
    def _lech(a, b):
        return sum(abs((a[i] - b[i] + math.pi) % (2 * math.pi) - math.pi) for i in range(3))
    ra = kia if _lech(kia, gan) < _lech((ex, ey, ez), gan) else (ex, ey, ez)
    return tuple(gan[i] + ((ra[i] - gan[i] + math.pi) % (2 * math.pi) - math.pi) for i in range(3))


def _q_slerp(a, b, t: float):
    d = sum(a[i] * b[i] for i in range(4))
    if d < 0:
        b, d = tuple(-v for v in b), -d
    if d > 0.9995:
        q = tuple(a[i] + t * (b[i] - a[i]) for i in range(4))
    else:
        th = math.acos(max(-1.0, min(1.0, d)))
        s = math.sin(th)
        k0, k1 = math.sin((1 - t) * th) / s, math.sin(t * th) / s
        q = tuple(k0 * a[i] + k1 * b[i] for i in range(4))
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    return tuple(v / n for v in q)


# ---- trục thời gian --------------------------------------------------------------------
@dataclass
class Doan:
    """Một đoạn chuyển động đặt tại mốc TUYỆT ĐỐI `t0` của lượt nói.

    `clip`: {"bones": {tên: [[t, ex, ey, ez], …]}, "hips_pos": [[t, x, y, z], …], "duration_s"}
    `nguon`: "nen" (transformer) | "ardy". Đoạn ARDY ĐÈ lên nền trong đúng khoảng của nó.
    """
    t0: float
    clip: dict
    nguon: str = "nen"
    ten: str = ""
    # Sổ sách của `OngStream`, gắn THẲNG vào đoạn. Bản đầu tra bằng `id(Doan)` trong một set:
    # đoạn đã kết sổ bị gỡ khỏi lịch rồi bị thu hồi, Python cấp lại ĐÚNG id đó cho đoạn MỚI —
    # đoạn mới bị coi là "đã kết sổ", không bao giờ được cộng vào gốc (đo 10/09: cú
    # 'walk_forward' thứ 5 của lượt, câu sau lệch 184 cm).
    da_gop: bool = False            # đã cộng vào gốc trôi — cấm cộng hai lần
    da_hien: bool = False           # đã hiện ở ít nhất một khung

    @property
    def dai(self) -> float:
        d = self.clip.get("duration_s")
        if d:
            return float(d)
        tr = next(iter((self.clip.get("bones") or {}).values()), None)
        return float(tr[-1][0]) if tr else 0.0

    @property
    def t1(self) -> float:
        return self.t0 + self.dai

    def phu(self, t: float) -> bool:
        return self.t0 - 1e-9 <= t < self.t1 - 1e-9


def _mau_track(track: list, t: float, gan=None):
    """Lấy mẫu một track xương tại thời điểm CỤC BỘ `t` — qua quaternion, không nội suy Euler."""
    if not track:
        return None
    if t <= track[0][0]:
        q = _euler_sang_q(track[0][1], track[0][2], track[0][3])
        return _q_sang_euler(q, gan)
    if t >= track[-1][0]:
        q = _euler_sang_q(track[-1][1], track[-1][2], track[-1][3])
        return _q_sang_euler(q, gan)
    lo, hi = 0, len(track) - 1
    while hi - lo > 1:                      # nhị phân: track có thể hàng nghìn khung
        gi = (lo + hi) // 2
        if track[gi][0] <= t:
            lo = gi
        else:
            hi = gi
    a, b = track[lo], track[hi]
    u = 0.0 if b[0] <= a[0] else (t - a[0]) / (b[0] - a[0])
    q = _q_slerp(_euler_sang_q(a[1], a[2], a[3]), _euler_sang_q(b[1], b[2], b[3]), u)
    return _q_sang_euler(q, gan)


def _mau_hp(hp: list, t: float):
    """hips_pos tại thời điểm cục bộ — nội suy tuyến tính (đây là VỊ TRÍ, không phải phép quay)."""
    if not hp:
        return [0.0, 0.0, 0.0]
    if t <= hp[0][0]:
        return [float(v) for v in hp[0][1:4]]
    if t >= hp[-1][0]:
        return [float(v) for v in hp[-1][1:4]]
    lo, hi = 0, len(hp) - 1
    while hi - lo > 1:
        gi = (lo + hi) // 2
        if hp[gi][0] <= t:
            lo = gi
        else:
            hi = gi
    a, b = hp[lo], hp[hi]
    u = 0.0 if b[0] <= a[0] else (t - a[0]) / (b[0] - a[0])
    return [float(a[1 + k] + (b[1 + k] - a[1 + k]) * u) for k in range(3)]


@dataclass
class Ghep:
    """Kết quả ghép: clip + SỔ NGUỒN từng khung (để đo và để nhìn thấy chỗ nhảy)."""
    clip: dict
    nguon_khung: list = field(default_factory=list)   # "nen" | "ardy" cho mỗi khung
    moi_noi: list = field(default_factory=list)       # mốc (giây) nơi nguồn đổi
    # ĐOẠN nào cấp khung đó. Bên gọi cần ĐÚNG ĐOẠN chứ không chỉ tên nguồn: một cue mới có thể
    # ĐÈ NGANG cue cũ khi cue cũ chưa chạy hết (luật "đặt sau thắng"), lúc đó `nguon_khung` vẫn
    # là "ardy" suốt nên nhìn vào nó KHÔNG thấy đoạn cũ đã thôi diễn — mà đúng chỗ đó là chỗ
    # phải kết sổ chỗ đứng, nếu không nhân vật nhảy về gốc của clip mới (đo 10/09: −2,50 m → 0).
    doan_khung: list = field(default_factory=list)


def ghep(nen: list[Doan], ardy: list[Doan], t_bd: float, t_kt: float,
         fps: int = FPS) -> Ghep:
    """Ghép một lát trục thời gian [t_bd, t_kt) thành MỘT clip liên tục.

    Luật chọn nguồn cho mỗi khung, theo đúng thứ tự này:
      1. đoạn ARDY nào phủ mốc đó → lấy đoạn MỚI NHẤT (đặt sau thắng, cue mới đè cue cũ);
      2. không có → đoạn nền phủ mốc đó;
      3. không có nữa → bỏ khung (nền chưa tới, không bịa).

    KHÔNG có nhánh nào trộn hai nguồn. Đó là toàn bộ nội dung của hàm này.
    """
    ten_xuong: list[str] = []
    for d in list(nen) + list(ardy):
        for x in (d.clip.get("bones") or {}):
            if x not in ten_xuong:
                ten_xuong.append(x)
    ra_bones: dict[str, list] = {x: [] for x in ten_xuong}
    ra_hp: list = []
    nguon_khung: list[str] = []
    doan_khung: list = []
    moi_noi: list = []
    truoc: dict[str, tuple] = {}
    nguon_truoc = None

    n = max(0, int(round((t_kt - t_bd) * fps)))
    for i in range(n):
        t_abs = t_bd + i / fps
        d = None
        for cs in reversed(ardy):           # cue đặt sau thắng
            if cs.phu(t_abs):
                d = cs
                break
        if d is None:
            for cs in nen:
                if cs.phu(t_abs):
                    d = cs
                    break
        if d is None:
            continue
        # MỐI NỐI = ĐỔI ĐOẠN, không chỉ đổi LOẠI nguồn (12/09 chiều, user "giật vị trí, chân cong quẹo"):
        # cue ARDY đè cue ARDY ("xoay người" → "chạy vòng") từng không được ghi nên `chuyen_tiep.lam_lien`
        # không làm liên → cắt cứng, đo câu 28: thân 38× · đầu 74× trần đúng mốc 1,4 s. Nền là MỘT đoạn cho
        # cả cửa sổ nên nền→nền không bao giờ thành mối nối.
        if nguon_truoc is not None and d is not nguon_truoc:
            moi_noi.append(round(t_abs - t_bd, 4))
        nguon_truoc = d
        nguon_khung.append(d.nguon)
        doan_khung.append(d)
        t_cb = t_abs - d.t0
        t_ra = round(i / fps, 4)
        b = d.clip.get("bones") or {}
        for x in ten_xuong:
            e = _mau_track(b.get(x) or [], t_cb, truoc.get(x))
            if e is None:
                # nguồn này không có xương đó: giữ giá trị khung trước, không bịa tư thế nghỉ
                e = truoc.get(x)
                if e is None:
                    continue
            truoc[x] = e
            ra_bones[x].append([t_ra, float(e[0]), float(e[1]), float(e[2])])
        ra_hp.append([t_ra, *_mau_hp(d.clip.get("hips_pos") or [], t_cb)])

    clip = {"bones": {x: v for x, v in ra_bones.items() if v},
            "hips_pos": ra_hp,
            "fps": fps,
            "duration_s": round(n / fps, 4) if n else 0.0,
            "nguon": "ghep"}
    return Ghep(clip=clip, nguon_khung=nguon_khung, moi_noi=moi_noi, doan_khung=doan_khung)
