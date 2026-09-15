"""RETARGET ARDY (NVIDIA, CoreSkeleton27, T-pose identity, quaternion xyzw, mét, Y lên,
+Z trước) -> rig Ready Player Me của ta (78 xương, A-pose, clip = Euler XYZ local
tuyệt đối theo khung, `hips_pos` = ĐỘ DỜI so tư thế nghỉ, `contacts` = khoảng thời
gian chạm sàn từng bàn chân) — mọi thứ ở BACKEND (§1 CLAUDE.md), không dùng
client Three.js của ARDY.

Công thức (guide §6): world(bone) = G_ardy(joint) ⊗ bindWorld(bone). G_ardy tích luỹ
quaternion local dọc chuỗi ARDY (rest = identity nên G chính là xoay THẾ GIỚI so
T-pose); bindWorld = xoay thế giới của xương rig ở tư thế nghỉ (A-pose) đọc từ
`hinh_hoc._rig()` (`rest_world`). Xương rig không có khớp ARDY (ngón tay, mắt…)
giữ xoay local nghỉ. Kiểm bằng FK độc lập trong `tools/kiem_ardy_retarget.py`.

Bẫy đã biết (CLAUDE.md §7/§9): retarget theo HƯỚNG xương bỏ mất xoay quanh trục
— ở đây dùng quaternion đầy đủ nên không mất; Euler ghi ra phải LIÊN TỤC
(`ong.euler_lien_tuc` ở cửa ra); quaternion của build_clip là WXYZ, của ARDY là XYZW.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# khớp ARDY -> xương rig (Spine3 -> Spine2 như MIXAMO_MAP của ARDY; ARDY Spine2,
# HandThumb1, HandEnd bỏ)
ANH_XA = {
    "Hips": "Hips", "Spine": "Spine", "Spine1": "Spine1", "Spine3": "Spine2",
    "Neck": "Neck", "Head": "Head",
    "LeftShoulder": "LeftShoulder", "LeftArm": "LeftArm", "LeftForeArm": "LeftForeArm", "LeftHand": "LeftHand",
    "RightShoulder": "RightShoulder", "RightArm": "RightArm", "RightForeArm": "RightForeArm", "RightHand": "RightHand",
    "LeftUpLeg": "LeftUpLeg", "LeftLeg": "LeftLeg", "LeftFoot": "LeftFoot", "LeftToeBase": "LeftToeBase",
    "RightUpLeg": "RightUpLeg", "RightLeg": "RightLeg", "RightFoot": "RightFoot", "RightToeBase": "RightToeBase",
}
# 42 xương của clip/dataset (đúng bộ dong_goi_mau ghi) — ngón giữ nghỉ
XUONG_CLIP = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
              "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
              "RightShoulder", "RightArm", "RightForeArm", "RightHand",
              "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
              "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
              "LeftHandThumb1", "LeftHandThumb2", "LeftHandIndex1", "LeftHandIndex2",
              "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandRing1", "LeftHandRing2",
              "LeftHandPinky1", "LeftHandPinky2",
              "RightHandThumb1", "RightHandThumb2", "RightHandIndex1", "RightHandIndex2",
              "RightHandMiddle1", "RightHandMiddle2", "RightHandRing1", "RightHandRing2",
              "RightHandPinky1", "RightHandPinky2"]


def _xyzw_to_wxyz(q):
    q = np.asarray(q, float)
    return np.array([q[3], q[0], q[1], q[2]])


def _the_gioi_ardy(info: dict, local_wxyz: np.ndarray, B) -> np.ndarray:
    """(J,4) xoay thế giới của từng khớp ARDY từ local (rest identity)."""
    joints, parents = info["joints"], info["parents"]
    G = np.zeros_like(local_wxyz)
    for j in range(len(joints)):
        p = parents[j]
        G[j] = local_wxyz[j] if p < 0 else B.q_mul(G[p], local_wxyz[j])
    return G


def tu_mang(joints: list, frames: list) -> list:
    """Đổi khung dạng MẢNG của ARDY motion service sang dạng ĐẶT TÊN mà `retarget` nhận.

    Service cục bộ (`ardy_service`, cổng 8090) trả `format: "array"`:
        {"t": …, "data": {"root": [x,y,z], "quats": [[x,y,z,w] × 27]}, "contacts": [4 cờ]}
    còn `retarget` đọc dạng của server RunPod (`format: "named"`):
        {"data": {"bones": {"<tên khớp>": [qx,qy,qz,qw] — riêng khớp gốc [px,py,pz,qx,qy,qz,qw]}}}
    Thứ tự `quats` khớp thứ tự `joints` (guide §4), khớp gốc là chỉ số 0.
    """
    ra = []
    for f in frames:
        d = f.get("data") or {}
        q = d.get("quats") or []
        root = list(d.get("root") or (0.0, 0.0, 0.0))
        if len(q) != len(joints):
            raise ValueError(f"khung ARDY có {len(q)} quat, xương {len(joints)}")
        bones = {n: (root + list(q[j]) if j == 0 else list(q[j])) for j, n in enumerate(joints)}
        ra.append({"t": f.get("t"), "data": {"bones": bones}, "contacts": f.get("contacts")})
    return ra


# VỊ TRÍ ĐÃ DỜI SANG `ardy_goc.py` (09/09). Tệp này chỉ còn lo ĐỔI KHÔNG GIAN (ARDY 27 khớp ⇄
# rig 78 xương). Kế toán vị trí từng rải bảy tệp và lỗi trôi 1,16 m sống đúng ở khe giữa chúng.
# Giữ hai tên cũ làm cầu vì tool ngoài repo còn gọi:
def he_do_doi(clip: dict) -> float:
    from app.modules.body_motion.ardy_goc import he_do_doi as _f
    return _f(clip)


def sua_do_doi(clip: dict) -> dict:
    from app.modules.body_motion.ardy_goc import sua
    return sua(clip)


@lru_cache(maxsize=1)
def _khung_rig():
    """(B, n2i, par, rr, rp, i2n, rest_world_q, tpose_world_q, tg) — dùng CHUNG cho CẢ HAI chiều.

    Phải dùng chung: hai chiều mà dựng T-pose lệch nhau dù một chút thì đổi xuôi rồi đổi ngược
    không về chỗ cũ, và sai số đó rơi thẳng vào mối nối — đúng thứ ta đang đi chữa.

    T-POSE THẾ GIỚI CHO CHUỖI TAY (07/09, user "2 tay bị bẻ ngược"): G của ARDY là xoay thế giới
    SO VỚI T-POSE (rest identity), còn rig nghỉ ở A-POSE (cánh tay đã hạ 35°). Ghép thẳng
    G ⊗ rest_world(A) là cộng cú hạ tay HAI LẦN. Đúng: G ⊗ world_T với world_T = R_A ⊗
    rest_world, R_A = phép quay nhỏ nhất đưa Arm→ForeArm về nằm ngang (±X), áp cho cả chuỗi.

    CẢ CHUỖI PHẢI THẲNG HÀNG, KHÔNG CHỈ CÁNH TAY (12/09, user "giơ tay trái" → bàn tay xuyên đầu):
    rig nghỉ KHUỶU gập 27,6° (góc 152,4°) và cổ tay 8,6°, còn T-pose của ARDY thẳng tay. Bản
    07/09 chỉ đưa cánh tay về ngang nên mọi clip ARDY mang thêm đúng cú gập đó — đo 6 clip cùng
    mẫu ở hai không gian: khuỷu rig gập hơn thô 27–28° ở CẢ hai tay (111,8→84,3 · 61,0→33,5 ·
    63,5→36,0), tay giơ lên là cẳng tay gập 159° chạm đầu. Nay đi dọc chuỗi Arm→ForeArm→Hand→
    ngón giữa: mỗi đốt xoay tối thiểu về ±X TRONG HỆ ĐÃ XOAY của đốt trước, áp cho cả nhánh
    dưới nó (đốt sau tích luỹ phép quay của đốt trước).

    §4 bẫy 8: trả BẢN SAO (`np.array`), không trả thẳng mảng của rig cache — một phép chuẩn hoá
    tại chỗ ở nơi khác là hỏng FK của cả tiến trình.
    """
    from app.modules.motion.hinh_hoc import _rig, _GLB
    B, n2i, par, rr, rp, _ = _rig()
    tg = B.prep_target(str(_GLB))
    rest_world_q = {i: np.array(tg["rest_world"][i][0], float) for i in rr}
    i2n = {i: n for n, i in n2i.items()}
    tpose_world_q = dict(rest_world_q)

    def _thuoc(i, goc) -> bool:
        k = i
        while k is not None:
            if k == goc:
                return True
            k = par.get(k)
        return False

    for ben, dT in (("Left", np.array([1.0, 0.0, 0.0])), ("Right", np.array([-1.0, 0.0, 0.0]))):
        chuoi = [(f"{ben}Arm", f"{ben}ForeArm"), (f"{ben}ForeArm", f"{ben}Hand"),
                 (f"{ben}Hand", f"{ben}HandMiddle1")]
        R = np.array([1.0, 0.0, 0.0, 0.0])          # phép quay tích luỹ của các đốt trên (wxyz)
        for goc_xuong, ngon in chuoi:
            ia, ic = n2i.get(goc_xuong), n2i.get(ngon)
            if ia is None or ic is None:
                break
            pa = np.array(tg["rest_world"][ia][1], float)
            pc = np.array(tg["rest_world"][ic][1], float)
            d = pc - pa
            d = B.q_rot(R, d / max(np.linalg.norm(d), 1e-9))     # hướng đốt SAU khi đốt trên đã xoay
            truc = np.cross(d, dT)
            s = np.linalg.norm(truc)
            c = float(np.clip(np.dot(d, dT), -1.0, 1.0))
            if s < 1e-6:
                continue
            goc = float(np.arctan2(s, c))
            truc = truc / s
            R_b = np.array([np.cos(goc / 2), *(np.sin(goc / 2) * truc)], float)      # wxyz
            R = B.q_mul(R_b, R)
            for i in rr:
                if _thuoc(i, ia):
                    tpose_world_q[i] = B.q_mul(R, rest_world_q[i])
    return B, n2i, par, rr, rp, i2n, rest_world_q, tpose_world_q, tg


def sang_ardy(info: dict, goc_xoay: dict, root: list | None = None) -> dict:
    """CHIỀU NGƯỢC của retarget: một tư thế RIG → một khung ARDY.

    Trả `{"quats": [[x,y,z,w] × J] local so T-pose, "root": [x,y,z]}` — đúng định dạng
    `ardy_service` nhận cho `history` (ghim khung ĐẦU) và `target_pose` (ghim khung CUỐI).

    VÌ SAO CẦN (user 09/09): "input frame đầu vào để ARDY bắt đầu từ frame đó, target frame cuối
    để ARDY kết thúc bằng frame đó". Neo hai đầu vào tư thế nền của nhân vật thì clip nào cũng
    ghép được vào clip nào, mà bếp VẪN nấu sẵn được (clip không phụ thuộc clip trước) — đo
    09/09 qua HTTP: mối nối clip↔clip max 91,3° → 7,0°, tb 10,4° → 0,7°.

    Công thức là nghịch đảo đúng của `retarget`: W = G ⊗ tpose_world ⇒ G = W ⊗ tpose_world⁻¹,
    rồi hạ về local theo cây ARDY: L[j] = G[cha]⁻¹ ⊗ G[j].

    `goc_xoay` = {tên xương rig: (x, y, z) Euler XYZ local, radian}. Khớp ARDY không có xương
    rig tương ứng nhận local = đơn vị (thừa hưởng cha) chứ KHÔNG nhận xoay đơn vị THẾ GIỚI —
    đặt sai chỗ này là bẻ gãy chuỗi từ đó trở xuống.
    """
    B, n2i, par, rr, rp, i2n, rest_world_q, tpose_world_q, _tg = _khung_rig()
    joints = list(info["joints"])
    cha = list(info.get("parents") or [-1] * len(joints))
    don_vi = np.array([1.0, 0.0, 0.0, 0.0])

    rot = dict(rr)
    for n, e in (goc_xoay or {}).items():
        i = n2i.get(n)
        if i is not None:
            rot[i] = B.q_from_euler_xyz(float(e[0]), float(e[1]), float(e[2]))

    W: dict = {}

    def _w(i):
        if i in W:
            return W[i]
        p = par.get(i)
        q = rot.get(i, rr[i])
        W[i] = B.q_mul(_w(p), q) if (p is not None and p in rr) else q
        return W[i]

    for i in rr:
        _w(i)

    G: list = [None] * len(joints)
    for j, jn in enumerate(joints):
        n = ANH_XA.get(jn)
        i = n2i.get(n) if n else None
        if i is not None:
            G[j] = B.q_mul(W[i], B.q_inv(tpose_world_q[i]))
        else:
            p = cha[j]
            # khớp ARDY không có xương rig: local = đơn vị ⇒ G nhận đúng G của cha
            G[j] = G[p] if (0 <= p < j and G[p] is not None) else don_vi

    quats = []
    for j in range(len(joints)):
        p = cha[j]
        L = G[j] if not (0 <= p < len(joints)) else B.q_mul(B.q_inv(G[p]), G[j])
        L = L / max(float(np.linalg.norm(L)), 1e-9)
        quats.append([float(L[1]), float(L[2]), float(L[3]), float(L[0])])   # wxyz -> xyzw
    return {"quats": quats, "root": [float(x) for x in (root or [0.0, 0.0, 0.0])]}


def retarget(info: dict, frames: list[dict], fps: float, nguon: str = "ardy", thuoc_hong: str = "dau") -> dict:
    """frames = motion_frame của ARDY (format named, space local), đều `fps` khung/s.
    Trả clip dict đúng định dạng kho: fps, duration_s, bones{name:[[t,x,y,z]]},
    hips_pos[[t,dx,dy,dz]], contacts{LeftFoot/RightFoot:[[t0,t1]]}, nguon."""
    # T-pose thế giới + khung rig dựng bởi `_khung_rig` — DÙNG CHUNG với chiều ngược `sang_ardy`
    # (xem lời giải thích ở đó: hai chiều dựng lệch nhau là đổi xuôi–ngược không về chỗ cũ).
    B, n2i, par, rr, rp, i2n, rest_world_q, tpose_world_q, tg = _khung_rig()
    joints = list(info["joints"]); jidx = {n: i for i, n in enumerate(joints)}
    # thứ tự rig cha trước con
    thu_tu = []
    def _them(i):
        if i in thu_tu:
            return
        p = par.get(i)
        if p is not None and p in rr:
            _them(p)
        thu_tu.append(i)
    for i in rr:
        _them(i)
    ten_xuong = [n for n in XUONG_CLIP if n in n2i]
    # tỉ lệ chiều cao hông (ARDY rest T-pose vs rig) cho độ dời hông
    # rest_positions của ARDY đặt Hips ở y=0 (gốc tại hông) -> lấy chiều cao ĐẦU so hông làm thước
    rpos = np.asarray(info.get("rest_positions") or [[0, 0, 0]] * len(joints), float)
    cao_ardy = float(rpos[jidx["Head"]][1] - rpos[jidx["Hips"]][1]) if "Head" in jidx else 0.0
    cao_rig = float(tg["rest_world"][n2i["Head"]][1][1] - tg["rest_world"][n2i["Hips"]][1][1])
    k_hong = (cao_rig / cao_ardy) if cao_ardy > 0.2 else 1.0
    # THƯỚC "chan" (15/09, tool kho ARDY): độ dời hông co giãn theo CHIỀU CAO HÔNG TRÊN CỔ CHÂN — chân mới là thứ đỡ
    # thân trên sàn. Khớp "Head" của hai bộ xương KHÔNG cùng định nghĩa (hông→Head ARDY 73,0 cm, Mira 52,0) nên thước
    # "dau" cho 0,712 trong khi chân dài như nhau (86,8 / 89,7 cm; hông trên cổ chân 89,6 / 91,0 → 1,016): hông rig chỉ
    # đi 71 % quãng thật mà góc chân chép nguyên → bàn chân bị kéo lê. Đo cùng câu cùng seed: xoay gót 3,4 cm (ARDY) →
    # 10,8 cm (thước đầu) → 3,1 cm (thước chân). Mặc định giữ "dau" cho đường 8000/8002 (hệ số `sua_do_doi` hiệu chuẩn trên nó).
    if thuoc_hong == "chan" and "LeftFoot" in jidx and "LeftFoot" in n2i:
        cao_a = float(rpos[jidx["Hips"]][1] - rpos[jidx["LeftFoot"]][1])
        cao_r = float(tg["rest_world"][n2i["Hips"]][1][1] - tg["rest_world"][n2i["LeftFoot"]][1][1])
        if cao_a > 0.3:
            k_hong = cao_r / cao_a
    T = len(frames)
    bones = {n: [] for n in ten_xuong}
    hips_pos = []
    root0 = None
    lien_lac = {"LeftFoot": [], "RightFoot": []}
    for k, f in enumerate(frames):
        t = round(k / fps, 4)
        d = f["data"]["bones"]
        J = len(joints)
        lq = np.zeros((J, 4))
        for j, n in enumerate(joints):
            v = d[n]
            lq[j] = _xyzw_to_wxyz(v[3:7] if j == 0 else v[0:4])
        G = _the_gioi_ardy(info, lq, B)
        W = {}
        for i in thu_tu:
            n = i2n[i]
            p = par.get(i)
            j = jidx.get(next((a for a, b in ANH_XA.items() if b == n), None), None)
            if j is not None:
                W[i] = B.q_mul(G[j], tpose_world_q[i])
            elif p is not None and p in W:
                W[i] = B.q_mul(W[p], rr[i])
            else:
                W[i] = rest_world_q[i]
        for n in ten_xuong:
            i = n2i[n]; p = par.get(i)
            Wp = W[p] if (p is not None and p in W) else np.array([1.0, 0, 0, 0])
            L = B.q_mul(B.q_inv(Wp), W[i])
            e = B.q_to_euler_xyz(L)
            bones[n].append([t, float(e[0]), float(e[1]), float(e[2])])
        root = np.asarray(d["Hips"][0:3], float)
        if root0 is None:
            root0 = root.copy()
        dh = (root - root0) * k_hong
        hips_pos.append([t, float(dh[0]), float(dh[1]), float(dh[2])])
        c = list(f.get("contacts") or [0, 0, 0, 0])
        lien_lac["LeftFoot"].append(bool(c[0] or c[1])); lien_lac["RightFoot"].append(bool(c[2] or c[3]))
    contacts = {}
    for ten, arr in lien_lac.items():
        kh, mo = [], None
        for k, v in enumerate(arr + [False]):
            if v and mo is None:
                mo = k
            elif not v and mo is not None:
                kh.append([round(mo / fps, 3), round(k / fps, 3)]); mo = None
        contacts[ten] = kh
    # ĐỘ DỜI GỐC KHỚP SẢI CHÂN: hiệu chỉnh ở ĐẦU RA ỐNG (`sua_do_doi`), không ở đây — ống còn
    # thu biên độ chân nên hệ số đo tại nguồn (1,31) khác hẳn hệ số đúng sau ống (0,45).
    return {"fps": float(fps), "duration_s": round((T - 1) / fps, 4), "bones": bones,
            "hips_pos": hips_pos, "contacts": contacts, "nguon": nguon, "source": nguon}
