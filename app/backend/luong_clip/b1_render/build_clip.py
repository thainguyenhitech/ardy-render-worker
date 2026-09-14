"""Tính bone tracks cho rig đích TRỰC TIẾP trên node glTF (không qua Blender) —
đúng không gian local mà Three.js áp dụng, hết lệch quy ước bone.

Chạy: python build_clip.py --positions pos.json --target characters/mira.glb \
        --out clip.json

Phương pháp FK direction-matching (position-based):
- Đọc node glTF: rest local rotation (quat) + translation, ghép hierarchy.
- Mỗi frame: từ vai xuống bàn tay, xoay node (góc nhỏ nhất, quaternion) sao cho
  hướng xương (node -> node con) trùng hướng đoạn chi nguồn (đổi Z-up -> Y-up).
- Xuất euler XYZ local từng frame — client set thẳng vào bone.rotation.

ANCHOR THEO TAG (motion-graph rút gọn): kho có K tư thế neo (_anchors.json,
sinh bởi build_anchors.py). Đầu/cuối mỗi clip được gán tag tư thế GẦN NHẤT
(khoảng cách vị trí khớp tay trong không gian NGUỒN — rig-invariant) rồi slerp
0.4s về đúng tư thế neo đó -> clip JSON mang pose_in/pose_out; runtime chỉ
matching tag là chuyển tiếp liền mạch. Chân + Hips luôn neo về pose chuẩn
(foot-lock IK giữ nguyên mốc _footBase).
"""
import argparse
import os
import json
import struct
import sys
from pathlib import Path

import numpy as np

_CLIPS_DIR = Path(__file__).parents[2] / "app" / "motion_clips"

# (node đích, khớp nguồn đầu, khớp nguồn cuối); node con để lấy hướng xương đích
# TOÀN BỘ khung xương: hips (giải 2-vector riêng) + cột sống + cổ + 2 tay
# (kèm ngón) + 2 chân. Head vẫn procedural (kênh gật-theo-lời-nói).
_FINGER_CHAINS = []
for _side_t, _side_s in (("Left", "L"), ("Right", "R")):
    for _f in ("Index", "Middle", "Ring", "Pinky", "Thumb"):
        _FINGER_CHAINS.append([
            (f"{_side_t}Hand{_f}1", f"{_side_s}_{_f}1", f"{_side_s}_{_f}2",
             f"{_side_t}Hand{_f}2"),
            (f"{_side_t}Hand{_f}2", f"{_side_s}_{_f}2", f"{_side_s}_{_f}3",
             f"{_side_t}Hand{_f}3"),
        ])

CHAINS = [
    [("Spine", "Spine1", "Spine2", "Spine1"),
     ("Spine1", "Spine2", "Spine3", "Spine2"),
     ("Spine2", "Spine3", "Neck", "Neck"),
     ("Neck", "Neck", "Head", "Head")],
    [("LeftShoulder", "L_Collar", "L_Shoulder", "LeftArm"),
     ("LeftArm", "L_Shoulder", "L_Elbow", "LeftForeArm"),
     ("LeftForeArm", "L_Elbow", "L_Wrist", "LeftHand"),
     ("LeftHand", "L_Wrist", "L_Middle1", "LeftHandMiddle1")],
    [("RightShoulder", "R_Collar", "R_Shoulder", "RightArm"),
     ("RightArm", "R_Shoulder", "R_Elbow", "RightForeArm"),
     ("RightForeArm", "R_Elbow", "R_Wrist", "RightHand"),
     ("RightHand", "R_Wrist", "R_Middle1", "RightHandMiddle1")],
    [("LeftUpLeg", "L_Hip", "L_Knee", "LeftLeg"),
     ("LeftLeg", "L_Knee", "L_Ankle", "LeftFoot"),
     ("LeftFoot", "L_Ankle", "L_Foot", None)],
    [("RightUpLeg", "R_Hip", "R_Knee", "RightLeg"),
     ("RightLeg", "R_Knee", "R_Ankle", "RightFoot"),
     ("RightFoot", "R_Ankle", "R_Foot", None)],
] + _FINGER_CHAINS

# xương luôn neo về POSE CHUẨN bất kể tag (giữ mốc chân cho foot-lock IK)
LEG_BONES = {"Hips", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg",
             "LeftFoot", "RightFoot"}

# khớp nguồn dùng làm "đặc trưng tư thế" khi gán tag (rel. Pelvis, Z-up)
FEATURE_JOINTS = ["L_Wrist", "R_Wrist", "L_Elbow", "R_Elbow", "Head"]


_FINGER_JOINT_NAMES = [f"{{s}}_{f}{i}" for f in ("Index", "Middle", "Ring",
                                                 "Pinky", "Thumb")
                       for i in (1, 2, 3)]


def push_crossed_arms(frames, allow_behind=False):
    """Khoanh tay / tay vắt qua thân: SMPL nguồn MỎNG hơn rig đích nên cẳng
    tay bị lún vào ngực + hai tay đan vào nhau quá sâu khi retarget. Sửa tại
    positions (áp cho cả clip lẫn tư thế neo):
    - đẩy bàn tay (100%) + khuỷu (50%) ra PHÍA TRƯỚC (blender -Y)
    - NÉN độ vắt qua trục thân: quá CROSS_MAX thì kéo về 60%
    - hai tay cùng khoanh -> SO LE trước-sau (tay trái ngoài, phải trong)
      như khoanh tay thật, không đan cùng một mặt phẳng"""
    PUSH_MAX = 0.07      # m khi vắt hẳn sang bên kia
    CROSS_MAX = 0.05     # m vắt qua midline tối đa, phần sâu hơn nén 60%
    SEP_MIN = 0.04       # m tách trước-sau tối thiểu giữa 2 cẳng tay khi khoanh

    def _restore_lengths(fr, orig):
        """Sau mọi luật chỉnh: ép lại ĐÚNG chiều dài xương tay như nguồn
        (vai->khuỷu rồi khuỷu->cổ tay; bàn tay+ngón dịch rigid theo cổ tay).
        Xương giãn/nén là nguồn gốc cổ tay gập góc quái sau direction-match."""
        import math as _m
        for s in ("L", "R"):
            sh, eo, wo = (orig.get(f"{s}_Shoulder"), orig.get(f"{s}_Elbow"),
                          orig.get(f"{s}_Wrist"))
            e, w = fr.get(f"{s}_Elbow"), fr.get(f"{s}_Wrist")
            shn = fr.get(f"{s}_Shoulder")
            if not all(x is not None for x in (sh, eo, wo, e, w, shn)):
                continue
            # khuỷu: giữ hướng mới, chiều dài cũ (vai không bị luật nào đụng)
            L0 = _m.dist(sh, eo)
            v = [e[i] - shn[i] for i in range(3)]
            L = _m.sqrt(sum(x * x for x in v)) or 1e-9
            e_new = [shn[i] + v[i] * L0 / L for i in range(3)]
            de = [e_new[i] - e[i] for i in range(3)]
            fr[f"{s}_Elbow"] = e_new
            w = [w[i] + de[i] for i in range(3)]   # cổ tay trôi theo khuỷu
            # cổ tay: chiều dài cẳng tay cũ
            F0 = _m.dist(eo, wo)
            v = [w[i] - e_new[i] for i in range(3)]
            F = _m.sqrt(sum(x * x for x in v)) or 1e-9
            w_new = [e_new[i] + v[i] * F0 / F for i in range(3)]
            dw = [w_new[i] - fr[f"{s}_Wrist"][i] for i in range(3)]
            for name in [f"{s}_Wrist"] + [j.format(s=s)
                                          for j in _FINGER_JOINT_NAMES]:
                if name in fr:
                    p = fr[name]
                    fr[name] = [p[0] + dw[0], p[1] + dw[1], p[2] + dw[2]]

    for fr in frames:
        _orig = {k: list(v) for k, v in fr.items()
                 if k[0] in "LR" and ("Shoulder" in k or "Elbow" in k
                                      or "Wrist" in k)}
        cross_k = {}
        for s, sign in (("L", 1.0), ("R", -1.0)):
            w = fr.get(f"{s}_Wrist")
            cross_k[s] = 0.0 if w is None else \
                max(0.0, min(1.0, (0.02 - sign * w[0]) / 0.10))
        # ÔM/QUẤN TAY QUANH THÂN (SelfHug...): cả 2 cổ tay vắt SÂU là tư thế
        # có chủ đích — fade tắt các chỉnh sửa (kẻo kéo tay khỏi vòng ôm +
        # nén cẳng tay làm bàn tay gập góc quái dị); thay bằng rigid push
        # nhẹ CẢ KHỐI tay ra trước (giữ nguyên hình học cẳng tay)
        depths = []
        for s, sign in (("L", 1.0), ("R", -1.0)):
            w = fr.get(f"{s}_Wrist")
            depths.append(-sign * w[0] if w is not None else -1.0)
        wrap = min(depths)
        fade = 1.0 if wrap <= 0.10 else max(0.0, (0.16 - wrap) / 0.06)
        if fade < 1.0:
            rigid = 0.04 * (1.0 - fade)
            for s in ("L", "R"):
                for name in ([f"{s}_Wrist", f"{s}_Elbow"]
                             + [j.format(s=s) for j in _FINGER_JOINT_NAMES]):
                    if name in fr:
                        pnt = fr[name]
                        fr[name] = [pnt[0], pnt[1] - rigid, pnt[2]]
        for s, sign in (("L", 1.0), ("R", -1.0)):
            k = cross_k[s] * fade
            if k <= 0:
                continue
            push = PUSH_MAX * k
            w = fr[f"{s}_Wrist"]
            over = (-sign * w[0]) - CROSS_MAX
            dx = sign * over * 0.6 * fade if over > 0 else 0.0
            for name in [f"{s}_Wrist"] + [j.format(s=s)
                                          for j in _FINGER_JOINT_NAMES]:
                if name in fr:
                    p = fr[name]
                    fr[name] = [p[0] + dx, p[1] - push, p[2]]
            e = fr.get(f"{s}_Elbow")
            if e is not None:
                # khuỷu bám 80% (không phải 50%): push vi sai làm GIÃN cẳng
                # tay -> direction-matching sinh cổ tay gập góc quái
                fr[f"{s}_Elbow"] = [e[0] + dx * 0.8, e[1] - push * 0.8, e[2]]
        # MẶT PHẲNG LƯNG: tay vung RA SAU thân (HY-Motion hay hất tay qua đầu
        # ra sau — celebrate/chop) retarget thành ưỡn/vặn người "dẹo" -> kéo
        # cổ tay về trước mặt phẳng lưng (blender: sau = +Y so pelvis), trừ
        # clip chủ đích tay-sau-lưng (allow_behind)
        if not allow_behind:
            pel = fr.get("Pelvis")
            if pel is not None:
                for s in ("L", "R"):
                    w = fr.get(f"{s}_Wrist")
                    if w is None:
                        continue
                    # kéo HẲN về trước mặt phẳng thân (đích -2cm, full) —
                    # HY-Motion hất tay sau tới 35cm, retarget còn khuếch đại
                    behind = w[1] - (pel[1] - 0.02)
                    if behind <= 0:
                        continue
                    dy = behind
                    for name in [f"{s}_Wrist"] + [j.format(s=s)
                                                  for j in _FINGER_JOINT_NAMES]:
                        if name in fr:
                            p = fr[name]
                            fr[name] = [p[0], p[1] - dy, p[2]]
                    e = fr.get(f"{s}_Elbow")
                    if e is not None:
                        fr[f"{s}_Elbow"] = [e[0], e[1] - dy * 0.75, e[2]]
        # TAY BUÔNG THẤP (ngang hông trở xuống, không vắt qua thân): rig có
        # đùi/hông/bụng dày hơn SMPL nên tay hay dính vào body -> ép 2 khoảng
        # thở: PHÍA TRƯỚC trục chậu (blender: trước = -Y) và RA BÊN CẠNH khi
        # tay trôi về giữa thân (trước bụng)
        HANG_MIN_FWD = 0.14
        HANG_MIN_OUT = 0.12
        pelvis = fr.get("Pelvis")
        if pelvis is not None:
            for s, sign in (("L", 1.0), ("R", -1.0)):
                w = fr.get(f"{s}_Wrist")
                if w is None or w[2] > pelvis[2] + 0.05:
                    continue          # chỉ áp cho tay từ ngang hông trở xuống
                out_now = sign * w[0]
                if out_now < -0.06:
                    continue          # vắt sâu ở tầm thấp (hiếm) — để luật vắt lo
                # vắt NÔNG ở tầm thấp = nhiễu retarget -> vẫn đẩy về bên mình
                need = max(0.0, w[1] - (pelvis[1] - HANG_MIN_FWD))
                dx = sign * (HANG_MIN_OUT - out_now) \
                    if out_now < HANG_MIN_OUT else 0.0
                if need <= 0 and dx == 0.0:
                    continue
                for name in [f"{s}_Wrist"] + [j.format(s=s)
                                              for j in _FINGER_JOINT_NAMES]:
                    if name in fr:
                        p = fr[name]
                        fr[name] = [p[0] + dx, p[1] - need, p[2]]
                e = fr.get(f"{s}_Elbow")
                if e is not None:
                    fr[f"{s}_Elbow"] = [e[0] + dx * 0.7, e[1] - need * 0.7,
                                        e[2]]
        # hai tay cùng khoanh: ÉP tách trước-sau tối thiểu (tay trái nằm
        # NGOÀI = trước hơn) — không đan cùng mặt phẳng, hết lún vào nhau
        # (fade tắt khi là tư thế ôm quấn sâu)
        if cross_k["L"] > 0 and cross_k["R"] > 0 and fade > 0:
            need = fr["L_Wrist"][1] - (fr["R_Wrist"][1] - SEP_MIN)
            if need > 0:
                scale = min(1.0, cross_k["L"], cross_k["R"]) * fade
                shift = need * scale
                for name in ["L_Wrist"] + [j.format(s="L")
                                           for j in _FINGER_JOINT_NAMES]:
                    if name in fr:
                        fr[name] = [fr[name][0], fr[name][1] - shift,
                                    fr[name][2]]
                e = fr.get("L_Elbow")
                if e is not None:
                    fr["L_Elbow"] = [e[0], e[1] - shift * 0.5, e[2]]
        _restore_lengths(fr, _orig)
    return frames


def pose_feature(joints):
    """Vector đặc trưng tư thế thân trên của 1 frame nguồn (rig-invariant)."""
    pelvis = np.array(joints["Pelvis"])
    return np.concatenate([np.array(joints[j]) - pelvis
                           for j in FEATURE_JOINTS if j in joints])


def frame_from_two_vectors(up, left):
    """Ma trận orientation trực giao từ trục lên + trục ngang (dùng cho Hips)."""
    up = up / np.linalg.norm(up)
    fwd = np.cross(left, up)
    fwd /= np.linalg.norm(fwd)
    left = np.cross(up, fwd)
    return np.stack([left, up, fwd], axis=1)   # cột: X=left, Y=up, Z=fwd


def mat_to_quat(m):
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    if w < 1e-8:
        # fallback trục lớn nhất
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1e-12, 1 + m[i, i] - m[j, j] - m[k, k])) * 2
        q = np.zeros(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
        return q / np.linalg.norm(q)
    x = (m[2, 1] - m[1, 2]) / (4 * w)
    y = (m[0, 2] - m[2, 0]) / (4 * w)
    z = (m[1, 0] - m[0, 1]) / (4 * w)
    return np.array([w, x, y, z])


# ---------- quaternion helpers (w, x, y, z) ----------

def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw])


def q_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)


def q_rot(q, v):
    return q_mul(q_mul(q, np.array([0.0, *v])), q_inv(q))[1:]


def q_between(a, b):
    """Quaternion nhỏ nhất xoay vector a -> b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d > 0.999999:
        return np.array([1.0, 0, 0, 0])
    if d < -0.999999:  # ngược chiều: xoay 180° quanh trục vuông góc bất kỳ
        axis = np.cross(a, np.array([1.0, 0, 0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0, 1.0, 0]))
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])
    axis = np.cross(a, b)
    q = np.array([1.0 + d, *axis])
    return q / np.linalg.norm(q)


def q_from_euler_xyz(ex, ey, ez):
    """Euler XYZ intrinsic (Three.js) -> quat: qx * qy * qz."""
    cx, sx = np.cos(ex / 2), np.sin(ex / 2)
    cy, sy = np.cos(ey / 2), np.sin(ey / 2)
    cz, sz = np.cos(ez / 2), np.sin(ez / 2)
    qx = np.array([cx, sx, 0.0, 0.0])
    qy = np.array([cy, 0.0, sy, 0.0])
    qz = np.array([cz, 0.0, 0.0, sz])
    return q_mul(q_mul(qx, qy), qz)


def q_slerp(a, b, t):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - t) * th) * a + np.sin(t * th) * b) / np.sin(th)


def q_to_euler_xyz(q):
    """Quat -> euler XYZ intrinsic (thứ tự mặc định của Three.js)."""
    w, x, y, z = q
    m00 = 1 - 2 * (y * y + z * z)
    m01 = 2 * (x * y - z * w)
    m02 = 2 * (x * z + y * w)
    m11 = 1 - 2 * (x * x + z * z)
    m12 = 2 * (y * z - x * w)
    m21 = 2 * (y * z + x * w)
    m22 = 1 - 2 * (x * x + y * y)
    ey = np.arcsin(np.clip(m02, -1, 1))
    if abs(m02) < 0.99999:
        ex = np.arctan2(-m12, m22)
        ez = np.arctan2(-m01, m00)
    else:
        ex = np.arctan2(m21, m11)
        ez = 0.0
    return ex, ey, ez


# ---------- glTF ----------

def load_gltf_nodes(path):
    raw = Path(path).read_bytes()
    assert raw[:4] == b"glTF"
    clen, _ = struct.unpack("<II", raw[12:20])
    gltf = json.loads(raw[20:20 + clen])
    nodes = gltf["nodes"]
    name2idx = {n.get("name"): i for i, n in enumerate(nodes)}
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    rest_rot, rest_pos = {}, {}
    for i, n in enumerate(nodes):
        r = n.get("rotation", [0, 0, 0, 1])          # gltf: x,y,z,w
        rest_rot[i] = np.array([r[3], r[0], r[1], r[2]])
        rest_pos[i] = np.array(n.get("translation", [0, 0, 0]), dtype=float)
    return name2idx, parent, rest_rot, rest_pos


def world_transform(idx, parent, rot, pos, cache):
    if idx in cache:
        return cache[idx]
    if idx not in parent:
        out = (rot[idx], pos[idx])
    else:
        pq, pp = world_transform(parent[idx], parent, rot, pos, cache)
        out = (q_mul(pq, rot[idx]), pp + q_rot(pq, pos[idx]))
    cache[idx] = out
    return out


def blender_to_gltf(v):  # Z-up -> Y-up
    return np.array([v[0], v[2], -v[1]])


def prep_target(path):
    """Nạp rig đích + tính sẵn rest world và frame Hips (dùng lại mọi frame)."""
    name2idx, parent, rest_rot, rest_pos = load_gltf_nodes(path)
    cache = {}
    rest_world = {i: world_transform(i, parent, rest_rot, rest_pos, cache)
                  for i in rest_rot}
    hips_rest_frame = None
    if all(n in name2idx for n in ("Hips", "Spine", "LeftUpLeg", "RightUpLeg")):
        h = rest_world[name2idx["Hips"]][1]
        up_t = rest_world[name2idx["Spine"]][1] - h
        left_t = (rest_world[name2idx["LeftUpLeg"]][1]
                  - rest_world[name2idx["RightUpLeg"]][1])
        hips_rest_frame = frame_from_two_vectors(up_t, left_t)
    return {"name2idx": name2idx, "parent": parent, "rest_rot": rest_rot,
            "rest_pos": rest_pos,          # cần cho FK khi giữ tiếp xúc
            "rest_world": rest_world, "hips_rest_frame": hips_rest_frame}


def solve_frame(joints, tg):
    """Giải local euler XYZ cho MỌI xương của 1 frame nguồn trên rig đích.
    Trả {bone: (ex, ey, ez)}. cur_world share giữa chuỗi: hips + cột sống
    trước -> vai/tay/chân tính trên thân đang xoay."""
    name2idx, parent = tg["name2idx"], tg["parent"]
    rest_rot, rest_world = tg["rest_rot"], tg["rest_world"]
    out = {}
    cur_world = {}

    if tg["hips_rest_frame"] is not None and \
            all(j in joints for j in ("Pelvis", "Spine1", "L_Hip", "R_Hip")):
        idx_h = name2idx["Hips"]
        up_s = blender_to_gltf(np.array(joints["Spine1"])
                               - np.array(joints["Pelvis"]))
        left_s = blender_to_gltf(np.array(joints["L_Hip"])
                                 - np.array(joints["R_Hip"]))
        m_src = frame_from_two_vectors(up_s, left_s)
        q_delta = mat_to_quat(m_src @ tg["hips_rest_frame"].T)
        new_world = q_mul(q_delta, rest_world[idx_h][0])
        p_idx = parent.get(idx_h)
        p_q = rest_world[p_idx][0] if p_idx is not None \
            else np.array([1.0, 0, 0, 0])
        out["Hips"] = q_to_euler_xyz(q_mul(q_inv(p_q), new_world))
        cur_world[idx_h] = new_world

    for chain in CHAINS:
        for t_name, s_head, s_tail, child_name in chain:
            if t_name not in name2idx or s_head not in joints:
                continue
            idx = name2idx[t_name]
            child_idx = name2idx.get(child_name)
            p_idx = parent.get(idx)
            p_rest_q = rest_world[p_idx][0] if p_idx is not None \
                else np.array([1.0, 0, 0, 0])
            p_cur_q = cur_world.get(p_idx, p_rest_q)
            rigid_q = q_mul(p_cur_q, rest_rot[idx]) if p_idx is not None \
                else rest_rot[idx]
            d_rest = q_rot(q_inv(rest_world[idx][0]),
                           rest_world[child_idx][1] - rest_world[idx][1]) \
                if child_idx is not None else np.array([0, 1.0, 0])
            d_rigid = q_rot(rigid_q, d_rest)
            d_want = blender_to_gltf(
                np.array(joints[s_tail]) - np.array(joints[s_head]))
            if t_name == "Neck" and child_idx is not None:
                # HY-Motion hay NGỬA CỔ quá đà (đo tới 46°) -> nén: cho ngửa
                # tự do ~6° so với rest, phần vượt chỉ giữ 25%
                rd = rest_world[child_idx][1] - rest_world[idx][1]
                rd = rd / np.linalg.norm(rd)
                dwn = d_want / np.linalg.norm(d_want)
                back = float(rd[2] - dwn[2])   # ngả sau = -Z tăng
                if back > 0.10:
                    dwn[2] += (back - 0.10) * 0.75
                    d_want = dwn / np.linalg.norm(dwn)
            q_corr = q_between(d_rigid, d_want)
            new_world_q = q_mul(q_corr, rigid_q)
            out[t_name] = q_to_euler_xyz(q_mul(q_inv(p_cur_q), new_world_q))
            cur_world[idx] = new_world_q
    return out


# ---------------------------------------------------------------- TIẾP XÚC --
# GIỮ TIẾP XÚC KHI RETARGET. Khâu retarget khớp theo HƯỚNG xương, không theo VỊ
# TRÍ đầu mút — đúng cho cử chỉ vung tay, SAI HẲN cho cử chỉ chạm vào cơ thể.
# Đo được: prompt "xoa hai tay vào nhau", HY sinh đúng (hai cổ tay cách nhau
# 3,7 cm ở bản gốc SMPL-H) nhưng clip retarget ra hai tay cách 44,8 cm — nhìn
# thành "xoa không khí". Tương tự xoa gáy 54,7 cm, tay áp ngực 27,5 cm.
# Cách sửa: phát hiện khung có TIẾP XÚC Ở NGUỒN, tính vị trí bàn tay MONG MUỐN
# trên rig đích (mốc đích + offset nguồn nhân tỉ lệ dài tay), rồi giải IK HAI
# XƯƠNG cho chuỗi vai-khuỷu để bàn tay tới đúng đó. Trọng số vào/ra mềm để
# không bụp ở mép cửa sổ tiếp xúc.
NGUONG_TIEP_XUC = 0.20      # m, khoảng cách NGUỒN coi là đang chạm/gần chạm
DAI_MEM_TIEP_XUC = 0.08     # m, dải chuyển tiếp mềm quanh ngưỡng
# TỰ BẬT: chỉ cặp nào TỪNG chạm thật (dưới ngưỡng này ở ít nhất một khung) mới
# được sửa. Không có nó thì mọi clip có tay quét ngang thân đều bị kéo, vì tay
# đi qua trong vòng 20 cm của thân là chuyện thường.
NGUONG_CHAM_THAT = 0.09     # m
# (tay, mốc nguồn, xương đích của mốc)
CAP_TIEP_XUC = [
    ("Right", "L_Wrist", "LeftHand"), ("Left", "R_Wrist", "RightHand"),
    ("Right", "Neck", "Neck"), ("Left", "Neck", "Neck"),
    ("Right", "Head", "Head"), ("Left", "Head", "Head"),
    ("Right", "Spine3", "Spine2"), ("Left", "Spine3", "Spine2"),
    ("Right", "Pelvis", "Hips"), ("Left", "Pelvis", "Hips"),
]


def _q_truc_goc(truc, goc):
    n = np.linalg.norm(truc)
    if n < 1e-9:
        return np.array([1.0, 0, 0, 0])
    truc = truc / n
    h = goc * 0.5
    return np.array([np.cos(h), *(truc * np.sin(h))])


def _goc_giua(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(float(np.dot(u, v)) / (nu * nv), -1.0, 1.0)))


def ik_hai_xuong(S, E, H, P, truc_on=None):
    """Trả (q_vai, q_khuyu) — hai phép xoay THÊM (world) để bàn tay H tới P.

    Chuẩn hai xương: gập khuỷu theo định lý cosin cho đúng khoảng cách, rồi
    xoay cả tay cho hướng vai->tay trùng hướng vai->đích. Trục gập lấy pháp
    tuyến mặt phẳng vai-khuỷu-tay hiện tại nên KHÔNG đổi mặt phẳng khuỷu
    (giữ nguyên dáng tay mà khâu khớp hướng đã dựng).

    `truc_on` (tuỳ chọn): trục gập THAM CHIẾU cho chuỗi gần DUỖI THẲNG. Chân
    người đứng gần collinear nên pháp tuyến đùi-gối-cổ chân toàn nhiễu — ngưỡng
    1e-7 cũ gần như không bao giờ kích hoạt, hướng trục lật tự do giữa hai
    khung và nghiệm NHẢY NHÁNH: đo 2026-08-23, ghim chân qua đây tạo bước
    13,8°/khung trong khi dữ liệu vào mượt 0,72°. Có truc_on thì (a) pháp
    tuyến yếu -> dùng thẳng truc_on trực giao hoá, (b) pháp tuyến đo được ->
    LẬT DẤU về cùng bán cầu với truc_on, nghiệm liên tục giữa các khung.
    """
    l1 = float(np.linalg.norm(E - S))
    l2 = float(np.linalg.norm(H - E))
    if l1 < 1e-6 or l2 < 1e-6:
        return None
    d = float(np.clip(np.linalg.norm(P - S), 1e-4, l1 + l2 - 1e-4))
    n = np.cross(E - S, H - S)
    nn = float(np.linalg.norm(n))
    if truc_on is not None:
        # TRỘN TRỤC LIÊN TỤC theo độ suy biến — KHÔNG ngưỡng nhị phân. Ba bản
        # trước đều tạo bậc thang ở chính ngưỡng mình đặt (1e-7 -> lật tự do
        # 13,8°/khung; if dot<0 -> chập chờn quanh 0 ra 40,8°; if |cos|<0,3 ->
        # nhảy 18° đúng tại biên 0,3). Bài học: trong chuỗi thời gian liên
        # tục, mọi NGƯỠNG chọn nhánh đều thành điểm gãy. Thay bằng trộn:
        #   - đo được rõ (chuỗi gập hẳn, mặt phẳng thuận trục) -> tin pháp tuyến
        #   - suy biến dần (gần thẳng, hoặc mặt phẳng xoay vuông góc trục)
        #     -> trượt dần về trục ổn định trực giao hoá
        to = np.asarray(truc_on, float)
        to = to / max(float(np.linalg.norm(to)), 1e-9)
        u = (E - S) / max(l1, 1e-9)
        t_orth = to - float(to @ u) * u
        t_orth = t_orth / max(float(np.linalg.norm(t_orth)), 1e-9)
        if nn > 1e-12:
            n_hat = n / nn
            if float(n_hat @ to) < 0:
                n_hat = -n_hat
        else:
            n_hat = t_orth
        c_ = abs(float(n_hat @ to))
        s_thang = min(1.0, max(0.0, 1.0 - nn / (0.05 * l1 * l2)))
        s_vuong = min(1.0, max(0.0, (0.45 - c_) / 0.35))
        be = max(s_thang, s_vuong)
        be = be * be * (3 - 2 * be)
        n = n_hat * (1.0 - be) + t_orth * be
    else:
        if nn < 1e-7:
            n = np.cross(E - S, P - S)
        if np.linalg.norm(n) < 1e-7:
            n = np.array([0.0, 0.0, 1.0])
    if np.linalg.norm(n) < 1e-7:
        n = np.array([1.0, 0.0, 0.0])
    n = n / np.linalg.norm(n)
    d_cur = float(np.clip(np.linalg.norm(H - S), 1e-4, l1 + l2 - 1e-4))
    def _a1(dd):
        return float(np.arccos(np.clip((l1 * l1 + dd * dd - l2 * l2)
                                       / (2 * l1 * dd), -1.0, 1.0)))
    def _a2(dd):
        return float(np.arccos(np.clip((l1 * l1 + l2 * l2 - dd * dd)
                                       / (2 * l1 * l2), -1.0, 1.0)))
    def _thu(nn_):
        qv = _q_truc_goc(nn_, _a1(d) - _a1(d_cur))
        qk = _q_truc_goc(nn_, _a2(d) - _a2(d_cur))
        E2 = S + q_rot(qv, E - S)
        H2 = E2 + q_rot(q_mul(qv, qk), H - E)
        return qv, qk, H2

    # DẤU CỦA TRỤC GẬP PHẢI TỰ KIỂM. `n` = (E-S)x(H-S) chỉ định nghĩa MỘT mặt
    # phẳng, không định nghĩa CHIỀU: xoay +Δa2 quanh n làm DUỖI hay GẬP là tuỳ
    # dấu n, mà dấu đó ngược nhau giữa khuỷu tay và gối. Hàm này ra đời cho
    # TAY với Δ nhỏ nên chưa bao giờ lộ; đem ghim CHÂN thì nó GẬP ĐÔI chân
    # thay vì duỗi — đo 26/08 trên clip Kimodo: đích cách 0,913 m mà bàn chân
    # ra ở 0,423 m (gập thêm 62,8°), FK lên 0,76 m = "chân dị tật, người bay".
    # Nghiệm: thử cả hai dấu, giữ dấu cho |S->tay| ĐÚNG bằng d cần đạt. Đây là
    # phép kiểm rẻ (vài phép nhân quaternion) và tự đúng cho mọi chuỗi.
    qv, qk, H2 = _thu(n)
    if abs(float(np.linalg.norm(H2 - S)) - d) > 1e-4:
        qv2, qk2, H2b = _thu(-n)
        if abs(float(np.linalg.norm(H2b - S)) - d) < abs(float(np.linalg.norm(H2 - S)) - d):
            qv, qk, H2 = qv2, qk2, H2b
    q_huong = q_between(H2 - S, P - S)
    return q_mul(q_huong, qv), qk


def _muot_tiep_xuc(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def giu_tiep_xuc(tracks, frames, tg, fps, ti_le=None):
    """Sửa track tay ở những khung mà NGUỒN đang chạm cơ thể. Sửa tại chỗ.

    Trả số khung đã sửa (0 = clip này không có tiếp xúc nào)."""
    n2i, parent = tg["name2idx"], tg["parent"]
    rest_rot, rest_world, rest_pos = tg["rest_rot"], tg["rest_world"], tg["rest_pos"]
    if ti_le is None:
        a_s, a_t = _arm_len_src(frames), _arm_len_target(tg)
        ti_le = (a_t / a_s) if (a_s and a_t) else 1.0
    # cặp nào TỪNG chạm thật trong clip này?
    cap_that = []
    for ben, moc_src, moc_dich in CAP_TIEP_XUC:
        w_src = f"{'L' if ben == 'Left' else 'R'}_Wrist"
        kc_min = min((float(np.linalg.norm(np.array(j[w_src]) - np.array(j[moc_src])))
                      for j in frames if w_src in j and moc_src in j), default=9e9)
        if kc_min <= NGUONG_CHAM_THAT:
            cap_that.append((ben, moc_src, moc_dich))
    if not cap_that:
        return 0
    da_sua = 0
    sua_khung: dict = {}
    for fi, joints in enumerate(frames):
        if fi >= len(next(iter(tracks.values()))):
            break
        # dựng rot hiện tại của khung này
        rot = dict(rest_rot)
        for ten, tr in tracks.items():
            if ten in n2i and fi < len(tr):
                rot[n2i[ten]] = q_from_euler_xyz(*tr[fi][1:])
        cache = {}
        def _w(i):
            return world_transform(i, parent, rot, rest_pos, cache)
        for ben, moc_src, moc_dich in cap_that:
            w_src = f"{'L' if ben == 'Left' else 'R'}_Wrist"
            if w_src not in joints or moc_src not in joints:
                continue
            kc = float(np.linalg.norm(np.array(joints[w_src])
                                      - np.array(joints[moc_src])))
            if kc > NGUONG_TIEP_XUC:
                continue
            w = _muot_tiep_xuc((NGUONG_TIEP_XUC - kc) / DAI_MEM_TIEP_XUC)
            if w <= 1e-3:
                continue
            ten_vai, ten_khuyu, ten_tay = f"{ben}Arm", f"{ben}ForeArm", f"{ben}Hand"
            if any(t not in n2i for t in (ten_vai, ten_khuyu, ten_tay, moc_dich)):
                continue
            S = _w(n2i[ten_vai])[1]
            E = _w(n2i[ten_khuyu])[1]
            H = _w(n2i[ten_tay])[1]
            M = _w(n2i[moc_dich])[1]
            lech = blender_to_gltf(np.array(joints[w_src])
                                   - np.array(joints[moc_src])) * ti_le
            P = M + lech
            P = H + (P - H) * w          # trộn mềm theo trọng số tiếp xúc
            kq = ik_hai_xuong(S, E, H, P)
            if kq is None:
                continue
            q_vai, q_khuyu = kq
            for ten, q_them in ((ten_vai, q_vai), (ten_khuyu, q_khuyu)):
                idx = n2i[ten]
                p_idx = parent.get(idx)
                p_q = _w(p_idx)[0] if p_idx is not None else np.array([1.0, 0, 0, 0])
                cu_world = q_mul(p_q, rot[idx]) if p_idx is not None else rot[idx]
                moi_world = q_mul(q_them, cu_world)
                rot[idx] = q_mul(q_inv(p_q), moi_world)
                cache.clear()
            # giữ nguyên HƯỚNG bàn tay trong world (lòng bàn tay vẫn như cũ)
            idx_h = n2i[ten_tay]
            p_h = parent.get(idx_h)
            if p_h is not None:
                rot[idx_h] = q_mul(q_inv(_w(p_h)[0]),
                                   q_mul(q_mul(q_vai, q_khuyu),
                                         q_mul(rest_world[p_h][0], rest_rot[idx_h])))
                cache.clear()
            da_sua += 1
            sua_khung.setdefault(ten_vai, set()).add(fi)
            sua_khung.setdefault(ten_khuyu, set()).add(fi)
            sua_khung.setdefault(ten_tay, set()).add(fi)
            for ten in (ten_vai, ten_khuyu, ten_tay):
                ex, ey, ez = q_to_euler_xyz(rot[n2i[ten]])
                tracks[ten][fi][1:] = [round(float(ex), 4), round(float(ey), 4),
                                       round(float(ez), 4)]
    # LÀM MƯỢT PHẦN ĐÃ SỬA. IK giải ĐỘC LẬP từng khung nên nhiễu tần cao: đo
    # được giật p95 78 -> 722 (trần mocap 470) dù khoảng cách tiếp xúc đã đúng.
    # Trung bình trượt trên QUATERNION (slerp đôi một, ép dấu liên tục) quanh
    # các khung đã sửa, cộng thêm 2 khung đệm mỗi bên để không gãy ở mép.
    for ten, khung in sua_khung.items():
        tr = tracks.get(ten)
        if not tr:
            continue
        rong = set()
        for fi in khung:
            rong.update(range(max(0, fi - 2), min(len(tr), fi + 3)))
        qs = [q_from_euler_xyz(*f[1:]) for f in tr]
        for i in range(1, len(qs)):
            if float(np.dot(qs[i - 1], qs[i])) < 0:
                qs[i] = -qs[i]
        for _ in range(2):                    # hai lượt cho êm
            moi = list(qs)
            for i in sorted(rong):
                if i == 0 or i >= len(qs) - 1:
                    continue
                tb = q_slerp(qs[i - 1], qs[i + 1], 0.5)
                moi[i] = q_slerp(qs[i], tb, 0.5)
            qs = moi
        for i in sorted(rong):
            ex, ey, ez = q_to_euler_xyz(qs[i])
            tr[i][1:] = [round(float(ex), 4), round(float(ey), 4),
                         round(float(ez), 4)]
    return da_sua


def foot_contacts(frames, fps):
    """Contact curves (pha chân BÁM ĐẤT) precompute từ vị trí khớp nguồn —
    client chỉ đọc nhãn để foot-lock IK, không tự dò (backend-first).
    Bám đất = độ cao khớp bàn chân sát mức thấp nhất của clip + vận tốc
    ngang nhỏ. Trả {"LeftFoot": [[t0,t1],...], "RightFoot": ...} (giây)."""
    H_THRESH = 0.05      # m trên mức sàn của clip
    V_THRESH = 0.30      # m/s vận tốc ngang
    MIN_LEN, MAX_GAP = 0.15, 0.12   # s: bỏ interval ngắn, nối khe hở nhỏ
    out = {}
    for target, src in (("LeftFoot", "L_Foot"), ("RightFoot", "R_Foot")):
        if src not in frames[0]:
            continue
        pos = np.array([f[src] for f in frames])       # (N,3) Z-up
        ground = pos[:, 2].min()
        vel = np.zeros(len(pos))
        vel[1:] = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) * fps
        vel[0] = vel[1] if len(vel) > 1 else 0.0
        flags = (pos[:, 2] - ground < H_THRESH) & (vel < V_THRESH)
        iv = []
        start = None
        for i, f in enumerate(flags):
            if f and start is None:
                start = i
            elif not f and start is not None:
                iv.append([start, i - 1])
                start = None
        if start is not None:
            iv.append([start, len(flags) - 1])
        merged = []
        for a, b in iv:
            if merged and (a - merged[-1][1]) / fps <= MAX_GAP:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        out[target] = [[round(a / fps, 3), round(b / fps, 3)]
                       for a, b in merged if (b - a) / fps >= MIN_LEN]
    return out


def smooth_frames(frames, window=3):
    """LỌC NHIỄU THỜI GIAN bằng Savitzky-Golay (kỹ thuật HY-Motion dùng nội
    bộ — motion_diffusion.smooth_with_savgol window 9/poly 5): khử nhiễu
    1-frame NHƯNG giữ nguyên đỉnh/hình dáng chuyển động, không bào tròn cú
    nhấn như trung bình trượt. window tham số cũ giữ nguyên ý nghĩa cường độ:
    3 = savgol(7,3) nhẹ; 5 = savgol(11,3) mạnh."""
    if len(frames) < 11:
        return frames
    from scipy.signal import savgol_filter
    import numpy as _np
    win, poly = (7, 3) if window <= 3 else (11, 3)
    keys = list(frames[0].keys())
    out = [dict() for _ in frames]
    for k in keys:
        pts = _np.array([fr[k] for fr in frames if k in fr])
        if len(pts) != len(frames):
            for i, fr in enumerate(frames):
                if k in fr:
                    out[i][k] = fr[k]
            continue
        sm = savgol_filter(pts, win, poly, axis=0)
        for i in range(len(frames)):
            out[i][k] = [float(sm[i][0]), float(sm[i][1]), float(sm[i][2])]
    return out


def _arm_len_src(frames):
    """Chiều dài tay nguồn (vai->khuỷu->cổ tay, trung bình 2 bên, frame 0)."""
    import math as _m
    f = frames[0]
    ls = []
    for s in ("L", "R"):
        try:
            ls.append(_m.dist(f[f"{s}_Shoulder"], f[f"{s}_Elbow"])
                      + _m.dist(f[f"{s}_Elbow"], f[f"{s}_Wrist"]))
        except KeyError:
            pass
    return sum(ls) / len(ls) if ls else None


def _arm_len_target(tg):
    """Chiều dài tay rig đích từ rest world (LeftArm->ForeArm->Hand)."""
    n2i, rw = tg["name2idx"], tg["rest_world"]
    ls = []
    for s in ("Left", "Right"):
        try:
            a = rw[n2i[f"{s}Arm"]][1]
            b = rw[n2i[f"{s}ForeArm"]][1]
            c = rw[n2i[f"{s}Hand"]][1]
            ls.append(float(np.linalg.norm(b - a) + np.linalg.norm(c - b)))
        except KeyError:
            pass
    return sum(ls) / len(ls) if ls else None


def _fk_hand_speeds(tracks, fps, tg):
    """Vận tốc tay per khung (FK rig đích) — dùng chung cho chuẩn hóa toàn
    cục lẫn giới hạn cục bộ."""
    n2i, parent = tg["name2idx"], tg["parent"]
    rest_rot = tg["rest_rot"]
    rest_pos = {}
    for i in rest_rot:
        pa = parent.get(i)
        if pa is None:
            rest_pos[i] = tg["rest_world"][i][1]
        else:
            pq, pp = tg["rest_world"][pa]
            w, x, y, z = pq
            inv = np.array([w, -x, -y, -z])
            rest_pos[i] = q_rot(inv, tg["rest_world"][i][1] - pp)
    hands = [n2i[h] for h in ("LeftHand", "RightHand") if h in n2i]
    nframes = len(next(iter(tracks.values())))
    prev, out = None, []
    for fi in range(nframes):
        rot = dict(rest_rot)
        for name, tr in tracks.items():
            if name in n2i and fi < len(tr):
                f = tr[fi]
                rot[n2i[name]] = q_from_euler_xyz(f[1], f[2], f[3])
        cache = {}
        cur = [world_transform(h, parent, rot, rest_pos, cache)[1] for h in hands]
        if prev is not None:
            out.append(max(float(np.linalg.norm(b - a)) for a, b in zip(prev, cur)) * fps)
        prev = cur
    return out


def local_speed_warp(tracks, contacts, fps, tg, v_max=3.0, m_cap=2.0):
    """GIỚI HẠN TỐC ĐỘ CỤC BỘ: chuẩn hóa toàn cục (p95) không bắt được ĐỈNH
    cục bộ (vd cú vung tay lên đầu 3.4 m/s giữa IdleAdjust — QA hộp đen).
    Khung nào tay vượt v_max thì GIÃN thời gian đúng khung đó (mult tới
    m_cap), làm mượt cửa sổ ±2 khung để không đổi nhịp đột ngột."""
    speeds = _fk_hand_speeds(tracks, fps, tg)
    if not speeds:
        return contacts
    mult = [1.0] + [max(1.0, min(m_cap, v / v_max)) for v in speeds]
    sm = [sum(mult[max(0, i-2):i+3]) / len(mult[max(0, i-2):i+3])
          for i in range(len(mult))]
    if all(m <= 1.001 for m in sm):
        return contacts
    any_tr = next(iter(tracks.values()))
    old_ts = [f[0] for f in any_tr]
    new_ts = [old_ts[0]]
    for i in range(1, len(old_ts)):
        new_ts.append(new_ts[-1] + (old_ts[i] - old_ts[i-1]) * sm[min(i, len(sm)-1)])
    remap = lambda t: float(np.interp(t, old_ts, new_ts))
    for tr in tracks.values():
        for i, f in enumerate(tr):
            f[0] = round(new_ts[i] if i < len(new_ts) else remap(f[0]), 4)
    return {k: [[round(remap(a), 3), round(remap(b), 3)] for a, b in iv]
            for k, iv in contacts.items()}


def speed_factor_fk(tracks, fps, tg):
    """CHUẨN HOÁ DẢI TỐC ĐỘ đo bằng FK TRÊN RIG ĐÍCH — đúng vận tốc world mà
    mắt người thấy trên màn hình (gồm cả phần vai/nghiêng thân cộng vào tay;
    đo trên cổ tay nguồn từng thiếu phần này nên clip render ra vẫn nhanh hơn
    cap). QA 2026-08-16: p95 world lên 3.3-4.0 m/s dù nguồn đã cap 2.8.
    Vượt V_HI thì GIÃN thời lượng (tối đa 1.6x), dưới V_LO nén nhẹ (tối đa
    0.8x — clip chậm thường chậm CÓ CHỦ ĐÍCH)."""
    V_HI, V_LO = 2.6, 0.25
    n2i, parent = tg["name2idx"], tg["parent"]
    rest_rot = tg["rest_rot"]
    rest_pos = {i: tg["rest_world"][i][1] * 0 for i in rest_rot}  # placeholder
    # rest_pos local cần bản gốc — lấy lại từ rest_world qua parent
    for i in rest_rot:
        p = parent.get(i)
        if p is None:
            rest_pos[i] = tg["rest_world"][i][1]
        else:
            pq, pp = tg["rest_world"][p]
            # local = rot(parent)^-1 * (world_i - world_p)
            w, x, y, z = pq
            inv = np.array([w, -x, -y, -z])
            rest_pos[i] = q_rot(inv, tg["rest_world"][i][1] - pp)
    hands = [n2i[h] for h in ("LeftHand", "RightHand") if h in n2i]
    nframes = len(next(iter(tracks.values())))
    prev = None
    speeds = []
    for fi in range(nframes):
        rot = dict(rest_rot)
        for name, tr in tracks.items():
            if name in n2i and fi < len(tr):
                f = tr[fi]
                rot[n2i[name]] = q_from_euler_xyz(f[1], f[2], f[3])
        cache = {}
        cur = [world_transform(h, parent, rot, rest_pos, cache)[1] for h in hands]
        if prev is not None:
            for a, b in zip(prev, cur):
                speeds.append(float(np.linalg.norm(b - a)) * fps)
        prev = cur
    if not speeds:
        return 1.0
    speeds.sort()
    p95 = speeds[int(len(speeds) * 0.95)]
    if p95 > V_HI:
        return min(1.6, p95 / V_HI)
    if 1e-6 < p95 < V_LO:
        return max(0.8, p95 / V_LO)
    return 1.0


def load_anchors(char_stem):
    """(features nguồn per tag, pose rig per tag) — rỗng nếu chưa build."""
    src_p = _CLIPS_DIR / "_anchors_src.json"
    rig_p = _CLIPS_DIR / char_stem / "_anchors.json"
    if not src_p.exists() or not rig_p.exists():
        return {}, {}
    src = {t: np.array(v) for t, v in json.loads(src_p.read_text()).items()}
    return src, json.loads(rig_p.read_text())


# MẶT NẠ BỘ PHẬN cho clip "diễn theo yêu cầu": HY-Motion luôn sinh toàn thân
# (tay vung, thân lắc kèm) — user review đòi ĐÚNG bộ phận đó thôi. factor <1
# nén về tư thế nền, >1 khuếch đại (gật đầu nguồn ~7-13 độ hơi nhẹ).
_MASK_GROUPS = {
    "head": lambda b: b in ("Head", "Neck"),
    "torso": lambda b: b == "Hips" or b.startswith("Spine"),
    "armL": lambda b: b.startswith("Left") and not any(
        x in b for x in ("Leg", "Foot", "Toe")),
    "armR": lambda b: b.startswith("Right") and not any(
        x in b for x in ("Leg", "Foot", "Toe")),
    "legs": lambda b: any(x in b for x in ("UpLeg", "Leg", "Foot", "Toe"))
        and b != "Hips",
}
_BOTH_ARMS = {"armL": 1.0, "armR": 1.0, "head": 0.5, "torso": 0.35, "legs": 0.1}
# dấu twist per bone (hiệu chuẩn bằng ảnh: build 2 dấu, soi lòng bàn tay)
ARM_TWIST_SIGN = {"LeftArm": 1.0, "LeftForeArm": 1.0, "LeftHand": 1.0,
                  "RightArm": 1.0, "RightForeArm": 1.0, "RightHand": 1.0}

_HEAD_REST = {"torso": 0.15, "armL": 0.08, "armR": 0.08, "legs": 0.05}
CLIP_MASKS = {
    # HIỆU CHỈNH per clip theo audit biên độ 2026-08-16 (dải chuẩn người thật:
    # gật nhẹ 8-14°, lắc chậm 18-28°, quay 45-65°):
    "HeadBobAgree": {"head": 3.0, **_HEAD_REST},     # 5.6° -> ~10.5°
    "HeadShakeSlow": {"head": 2.2, **_HEAD_REST},    # 17.1° -> ~23°
    "HeadTurnLeft": {"head": 0.70, **_HEAD_REST},    # 125° -> ~55°
    "HeadTurnRight": {"head": 0.78, **_HEAD_REST},   # 112° -> ~55°
    # ĐẦU còn lại: boost chung 1.6, phần khác gần đứng yên
    "Head": {"head": 1.6, **_HEAD_REST},
    "ActionRaiseLeftHand": {"armL": 1.0, "head": 0.5, "torso": 0.3,
                            "armR": 0.1, "legs": 0.08},
    "ActionRaiseRightHand": {"armR": 1.0, "head": 0.5, "torso": 0.3,
                             "armL": 0.1, "legs": 0.08},
    "ActionSaluteArmy": {"armR": 1.0, "head": 0.6, "torso": 0.3,
                         "armL": 0.12, "legs": 0.08},
    "ActionRaiseBothHands": _BOTH_ARMS, "ActionWaveBothArms": _BOTH_ARMS,
    "ActionAirPunch": _BOTH_ARMS, "ActionHeartOverhead": _BOTH_ARMS,
    "ActionHandsBehindHead": _BOTH_ARMS,
    "ActionStompFoot": {"legs": 1.0, "torso": 0.5, "armL": 0.35,
                        "armR": 0.35, "head": 0.4},
    "ActionKickLight": {"legs": 1.0, "torso": 0.6, "armL": 0.45,
                        "armR": 0.45, "head": 0.4},
    "ActionTiptoe": {"legs": 1.0, "torso": 0.6, "armL": 0.5, "armR": 0.5,
                     "head": 0.4},
    "ActionMarchInPlace": {"legs": 1.0, "torso": 0.6, "armL": 0.7,
                           "armR": 0.7, "head": 0.4},
    # hành động toàn thân (Bow/Sit/Spin/Jump/Lean...) không mask — cả người
    # chuyển động là ĐÚNG ngữ nghĩa
}


def apply_mask(tracks, base_local, mask):
    """Scale delta-so-với-base per bone: factor <1 nén, >1 khuếch đại."""
    for name, tr in tracks.items():
        if name == "HipsPos":
            continue
        factor = None
        for grp, fac in mask.items():
            if _MASK_GROUPS[grp](name):
                factor = fac
                break
        if factor is None or abs(factor - 1.0) < 1e-3:
            continue
        qb = base_local.get(name)
        if qb is None:
            continue
        w0, x0, y0, z0 = qb
        inv = np.array([w0, -x0, -y0, -z0])
        for f in tr:
            q = q_from_euler_xyz(f[1], f[2], f[3])
            d = q_mul(q, inv)              # delta = q * base^-1
            if d[0] < 0:
                d = -d
            ang = 2.0 * np.arccos(min(1.0, d[0]))
            s_ = np.sqrt(max(0.0, 1.0 - d[0] * d[0]))
            if s_ < 1e-8:
                continue
            axis = d[1:] / s_
            na = ang * factor
            nd = np.array([np.cos(na / 2), *(axis * np.sin(na / 2))])
            nq = q_mul(nd, qb)
            ex, ey, ez = q_to_euler_xyz(nq)
            f[1], f[2], f[3] = round(float(ex), 4), round(float(ey), 4), \
                round(float(ez), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    # ÉP tag tư thế đầu/cuối (thay vì tự gán theo neo gần nhất) — dùng để
    # build NHIỀU PHIÊN BẢN một motion, mỗi phiên bản vào từ một tư thế neo
    # khác nhau -> đồ thị tư thế phủ kín, matching không bao giờ kẹt
    ap.add_argument("--pose-in", dest="pose_in")
    ap.add_argument("--smooth", type=int, default=3,
                    help="cửa sổ lọc trung bình trượt vị trí nguồn (3=nhẹ, 5=mạnh)")
    ap.add_argument("--pose-out", dest="pose_out")
    # ĐỒNG BỘ VỚI TIẾNG: ease-in/out bên dưới kéo giãn trục thời gian để clip
    # rời rạc kết thúc mềm. Với chuyển động sinh TỪ ÂM THANH thì phép giãn đó
    # làm lệch nhịp so với giọng nói (10.5s motion cho 9.9s tiếng) — phải tắt.
    # Chốt cứng thân dưới. Zero hoá góc khớp chân ở bước npz chưa đủ: build_clip
    # suy góc xoay TỪ VỊ TRÍ khớp, nên làm mượt và cong vênh tốc độ vẫn để lại
    # dao động ~11 độ ở chân. Ghim thẳng track về khung đầu mới thật sự đứng yên.
    ap.add_argument("--lock-lower", dest="lock_lower", action="store_true",
                    help="ghim xương chân về tư thế khung đầu (chuyển động sinh "
                         "từ giọng nói không mang thông tin ở thân dưới)")
    # KHÔNG KÉO VỀ TƯ THẾ NEO ở hai đầu. Neo là cơ chế motion-graph của KHO
    # CLIP (clip nối nhau qua tư thế A-E). Clip GRU sinh theo từng câu mà vẫn
    # bị neo thì mỗi câu bị kéo về đúng một tư thế trong 0,4s cuối — đo được
    # khung đầu/cuối mọi câu trùng nhau tới từng mm: "tay xếp gọn sau mỗi câu".
    # Giữa các câu GRU nay backend nối bằng nuong_noi(), không cần neo nữa.
    ap.add_argument("--no-anchor", dest="no_anchor", action="store_true",
                    help="giữ nguyên tư thế đầu/cuối của nguồn, không slerp về neo")
    # Với chuyển động sinh TỪ GIỌNG NÓI, clip không có nghĩa nếu tách khỏi
    # đoạn tiếng đã sinh ra nó. Ghi đường dẫn tiếng vào luôn để trang thử tự
    # chọn đúng file, khỏi phải nhớ clip nào đi với tiếng nào.
    ap.add_argument("--audio", help="đường dẫn tiếng đã sinh ra chuyển động này "
                                    "(vd /vocal_test/mira_nh_long.wav)")
    ap.add_argument("--giu_nhip", action="store_true",
                    help="không co giãn thời gian (bỏ speed_factor_fk + local_speed_warp) — clip Kimodo tự do")
    ap.add_argument("--no-retime", dest="no_retime", action="store_true",
                    help="giữ nguyên trục thời gian gốc (dùng cho motion sinh "
                         "từ audio, cần khớp nhịp với giọng nói)")
    args = ap.parse_args()

    data = json.load(open(args.positions))
    _base = Path(args.positions).name.replace(".pos.json", "")
    fps, frames = data["fps"], push_crossed_arms(
        smooth_frames(data["frames"], window=args.smooth),
        allow_behind=_base.startswith("IdleHandsBehind"))
    tg = prep_target(args.target)
    name2idx, rest_rot = tg["name2idx"], tg["rest_rot"]

    tracks = {}
    for fi, joints in enumerate(frames):
        t_sec = round(fi / fps, 4)
        for name, (ex, ey, ez) in solve_frame(joints, tg).items():
            tracks.setdefault(name, []).append(
                [t_sec, round(float(ex), 4), round(float(ey), 4),
                 round(float(ez), 4)])

    # GIỮ TIẾP XÚC: nguồn chạm vào người thì đích cũng phải chạm (xem khối
    # "TIẾP XÚC" phía trên). Tự bật theo nội dung clip, clip không có tiếp xúc
    # thì hàm trả 0 và không sờ vào track nào.
    _n_tx = (giu_tiep_xuc(tracks, frames, tg, fps)
             if os.environ.get("GIU_TIEP_XUC", "1") != "0" else 0)
    if _n_tx:
        print(f"  giữ tiếp xúc: sửa {_n_tx} lượt khung-tay")

    # TỊNH TIẾN HÔNG (HipsPos): ngồi xổm/nhảy/quỳ cần hông LÊN-XUỐNG —
    # rotation-only làm crouch/jump đứng đơ (QA ảnh đã dính). Xuất offset
    # Pelvis so với frame đầu, scale theo tỉ lệ chân đích/nguồn; client áp
    # vào bones.Hips.position, legIK tự gập gối khi hông hạ.
    def _leg_len_src(fr0):
        import math as _m
        try:
            return (_m.dist(fr0["Pelvis"], fr0["L_Knee"])
                    + _m.dist(fr0["L_Knee"], fr0["L_Ankle"]))
        except KeyError:
            return None

    def _leg_len_target():
        n2i, rw = tg["name2idx"], tg["rest_world"]
        try:
            a = rw[n2i["LeftUpLeg"]][1]
            b = rw[n2i["LeftLeg"]][1]
            c = rw[n2i["LeftFoot"]][1]
            return float(np.linalg.norm(b - a) + np.linalg.norm(c - b))
        except KeyError:
            return None

    l_src, l_tgt = _leg_len_src(frames[0]), _leg_len_target()
    if l_src and l_tgt and "Pelvis" in frames[0]:
        k = l_tgt / l_src
        p0 = np.array(frames[0]["Pelvis"])
        hips_tr = []
        for fi, joints in enumerate(frames):
            t_sec = round(fi / fps, 4)
            d = (np.array(joints["Pelvis"]) - p0) * k
            # nguồn Z-up (Blender schema) -> Y-up glTF: (x, z, -y)
            hips_tr.append([t_sec, round(float(d[0]), 4),
                            round(float(d[2]), 4), round(float(-d[1]), 4)])
        tracks["HipsPos"] = hips_tr

    # TWIST ĐẦU/CỔ từ SMPL poses (head_twist trong positions JSON):
    # direction-matching không quan sát được xoay quanh trục xương -> quay
    # đầu trái/phải từng bị MẤT trên mọi clip. Cộng twist cổ vào Neck.y và
    # tạo track Head (trước giờ Head thuần procedural). Đổi dấu vì nguồn
    # Z-up đã lật (blender_to_gltf: y -> -z).
    twist = data.get("head_twist")
    if twist and "Neck" in tracks:
        nfr = len(tracks["Neck"])
        head_tr = []
        for fi in range(nfr):
            tw = twist[min(fi, len(twist) - 1)]
            f = tracks["Neck"][fi]
            f[2] = round(f[2] + float(tw[0]), 4)
            head_tr.append([f[0], 0.0, round(float(tw[1]), 4), 0.0])
        tracks["Head"] = head_tr

    # POSE CHUẨN per-rig (tag "A"): tay từ solver standard_pose, còn lại rest
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from app.modules.pose.standard_pose import get_base_pose
    bp = get_base_pose(Path(args.target).stem)
    base_local = {}
    for name in tracks:
        if name == "HipsPos":
            continue
        base_local[name] = q_from_euler_xyz(*bp[name]) if name in bp \
            else rest_rot[name2idx[name]]

    # TWIST CHUỖI TAY từ SMPL poses (PHASE 2 full-rotation — arm_twist trong
    # positions JSON): direction-matching là shortest-arc nên twist quanh
    # trục xương = 0 theo cấu trúc — lật lòng bàn tay/xoay cẳng tay bị mất
    # trên MỌI clip từ trước. Compose q' = q ⊗ Ry(tw) (quay quanh trục Y
    # local = dọc xương, quy ước Mixamo/RPM) — KHÔNG cộng euler thô.
    atw = data.get("arm_twist")
    if atw:
        _ARM_TWIST_MAP = [("LeftArm", 0), ("LeftForeArm", 1), ("LeftHand", 2),
                          ("RightArm", 3), ("RightForeArm", 4),
                          ("RightHand", 5)]
        for name, col in _ARM_TWIST_MAP:
            tr = tracks.get(name)
            if not tr:
                continue
            sign = ARM_TWIST_SIGN.get(name, 1.0)
            for fi, f in enumerate(tr):
                tw = float(atw[min(fi, len(atw) - 1)][col]) * sign
                if abs(tw) < 1e-4:
                    continue
                q = q_from_euler_xyz(f[1], f[2], f[3])
                h = tw / 2.0
                qy = np.array([np.cos(h), 0.0, np.sin(h), 0.0])
                nq = q_mul(q, qy)
                ex, ey, ez = q_to_euler_xyz(nq)
                f[1], f[2], f[3] = round(float(ex), 4), round(float(ey), 4), \
                    round(float(ez), 4)

    # MẶT NẠ BỘ PHẬN (nếu clip thuộc bảng CLIP_MASKS)
    masked = False
    for prefix, mask in CLIP_MASKS.items():
        if _base.startswith(prefix):
            apply_mask(tracks, base_local, mask)
            masked = True
            break

    # Gán tag tư thế cho ĐẦU/CUỐI clip: anchor gần nhất trong không gian nguồn
    src_anchors, rig_anchors = load_anchors(Path(args.target).stem)

    def nearest_tag(joints):
        if not src_anchors:
            return "A"
        f = pose_feature(joints)
        return min(src_anchors, key=lambda t: float(
            np.linalg.norm(f - src_anchors[t])))

    pose_in = args.pose_in or nearest_tag(frames[0])
    pose_out = args.pose_out or nearest_tag(frames[-1])
    if masked:
        # clip đã mask: thân đứng ở tư thế chuẩn -> neo BẮT BUỘC A/A, kẻo
        # cửa sổ neo kéo bộ phận-đã-nén về tư thế neo khác (tay bật lên
        # vài chục độ dù mask 0.08 — QA đã dính)
        pose_in = pose_out = "A"
    # QUY TẮC: clip TALKING không được kết ở C (khoanh tay) / D (tay cằm) —
    # khoảng ngắt giữa beat sẽ đứng khoanh tay khi đang nói. Ép về tư thế
    # hội thoại {A,B,E} gần nhất. C/D vẫn hợp lệ cho idle/thinking.
    _TALK_PREFIX = ("Talking", "CalmTalk", "Gesture", "Surprised", "Welcome",
                    "HandOnChest", "Shrug", "Pointing", "Waving")
    base_name = Path(args.positions).name.replace(".pos.json", "")
    rule_forced = False
    if not args.pose_out and base_name.startswith(_TALK_PREFIX) \
            and pose_out in ("C", "D") and src_anchors:
        f_end = pose_feature(frames[-1])
        pose_out = min(("A", "B", "E"), key=lambda t: float(
            np.linalg.norm(f_end - src_anchors[t])))
        rule_forced = True
    forced = bool(args.pose_in or args.pose_out) or rule_forced

    def anchor_quat(tag, name):
        # chân/Hips luôn về pose chuẩn (mốc foot-lock); tag A = pose chuẩn
        if tag != "A" and name not in LEG_BONES and \
                name in rig_anchors.get(tag, {}):
            return q_from_euler_xyz(*rig_anchors[tag][name])
        return base_local[name]

    # ANCHOR: slerp 0.4s đầu về tư thế neo pose_in, 0.4s cuối về pose_out —
    # chuyển tiếp liền mạch từ DỮ LIỆU, runtime chỉ matching tag.
    dur = (len(frames) - 1) / fps
    # tag ép có thể xa tư thế gốc -> cửa sổ neo dài hơn cho cú kéo êm
    anchor = min(0.6 if forced else 0.4, dur * 0.4)
    if args.no_anchor:
        anchor = 0.0          # cửa sổ neo rỗng -> mọi khung giữ nguyên
    # CHIA CHO 0: mốc thời gian ghi ra đã LÀM TRÒN 4 số (89/30 = 2,9667) trong
    # khi dur tính đủ (2,96666...), nên khung cuối cho edge = -3e-5 < anchor = 0
    # -> lọt vào nhánh neo rồi chia cho 0. Chặn ở gốc: anchor = 0 thì bỏ hẳn
    # bước neo, đúng ý --no-anchor.
    _can_neo = tracks.items() if anchor > 1e-9 else []
    for name, tr in _can_neo:
        if name == "HipsPos":
            # neo về 0 (đứng chuẩn) ở hai đầu — cùng cửa sổ với rotation
            for f in tr:
                w = 1.0
                if f[0] < anchor:
                    w = (f[0] / anchor) ** 2
                edge = dur - f[0]
                if edge < anchor:
                    w = min(w, (edge / anchor) ** 2)
                if w < 1.0:
                    f[1] = round(f[1] * w, 4)
                    f[2] = round(f[2] * w, 4)
                    f[3] = round(f[3] * w, 4)
            continue
        q_in, q_out = anchor_quat(pose_in, name), anchor_quat(pose_out, name)
        for f in tr:
            w = 1.0
            qb = None
            if f[0] < anchor:
                u = f[0] / anchor
                w, qb = u * u * (3 - 2 * u), q_in
            edge = dur - f[0]
            if edge < anchor:
                u = edge / anchor
                w2 = u * u * (3 - 2 * u)
                if w2 < w:
                    w, qb = w2, q_out
            if qb is None or w >= 1.0:
                continue
            qm = q_slerp(qb, q_from_euler_xyz(f[1], f[2], f[3]), w)
            ex, ey, ez = q_to_euler_xyz(qm)
            f[1], f[2], f[3] = round(float(ex), 4), round(float(ey), 4), \
                round(float(ez), 4)

    # chuẩn hoá tốc độ: scale TOÀN BỘ trục thời gian theo factor (timestamps,
    # duration, contacts, fps hiệu dụng) — client đọc theo timestamp nên chạy
    # trong suốt; admin preview dùng fps đã scale khớp khoảng cách frame
    # --giu_nhip (clip Kimodo tự do, 14/09): KHÔNG đổi nhịp — cú nhảy/xoay nhanh là NỘI DUNG, kéo chậm là méo.
    factor = 1.0 if args.giu_nhip else speed_factor_fk(tracks, fps, tg)
    if abs(factor - 1.0) > 1e-3:
        for tr in tracks.values():
            for f in tr:
                f[0] = round(f[0] * factor, 4)
    contacts = foot_contacts(frames, fps)
    if abs(factor - 1.0) > 1e-3:
        contacts = {k: [[round(a * factor, 3), round(b * factor, 3)]
                        for a, b in iv] for k, iv in contacts.items()}

    if not args.giu_nhip:
        contacts = local_speed_warp(tracks, contacts, fps / factor, tg)
    # duration sau local warp = timestamp cuối
    # EASE-IN/OUT RETIME: HY-Motion cắt cứng ở 3s nên nhiều clip đang đà thì
    # hết frame -> kết thúc khựng lại đột ngột, nhìn cứng. Bung giãn trục thời
    # gian ở W_OUT cuối (tốc độ phát giảm dần về 1/S_OUT) + nhẹ ở W_IN đầu —
    # mọi clip tự giảm tốc vào tư thế neo như người thật thu tay về.
    D_lin = max(f[0] for tr in tracks.values() for f in tr[-1:])
    W_OUT = min(0.9, D_lin * 0.4)
    S_OUT = 3.0          # cuối clip chậm còn ~33% tốc độ gốc
    W_IN = min(0.3, D_lin * 0.15)
    S_IN = 1.7           # đầu clip vào chậm ~60% rồi tăng dần

    def _warp(t):
        if args.no_retime:      # giữ nguyên trục thời gian, không chia cho 0
            return t
        add_in = (S_IN - 1) * W_IN / 3
        if t <= 0:
            return 0.0
        if t < W_IN:
            u = t / W_IN
            return W_IN * (u + (S_IN - 1) / 3 * (1 - (1 - u) ** 3))
        base = t + add_in
        if t <= D_lin - W_OUT:
            return base
        u = (t - (D_lin - W_OUT)) / W_OUT
        return (D_lin - W_OUT) + add_in + \
            W_OUT * (u + (S_OUT - 1) / 3 * u ** 3)

    for tr in tracks.values():
        for f in tr:
            f[0] = round(_warp(f[0]), 4)
    contacts = {k: [[round(_warp(a), 3), round(_warp(b), 3)] for a, b in iv]
                for k, iv in contacts.items()}
    duration_s = round(_warp(D_lin), 3)

    if args.lock_lower:
        # PHẢI ĐẶT CUỐI: hoà vào tư thế neo ở đầu/cuối clip ghi đè lại giá trị
        # góc, nên ghim sớm hơn là vô ích (đã thử, chân vẫn dao động ~11 độ).
        LOWER = ("UpLeg", "Leg", "Foot", "Toe")
        n = 0
        for name, tr in tracks.items():
            if name != "HipsPos" and any(k in name for k in LOWER) and tr:
                a, b, c = tr[0][1], tr[0][2], tr[0][3]
                for f in tr:
                    f[1], f[2], f[3] = a, b, c
                n += 1
        print(f"  ghim {n} xương thân dưới về tư thế khung đầu")

    hips_pos = tracks.pop("HipsPos", None)
    out = {"fps": round(fps / factor, 3),
           "duration_s": duration_s,
           "source": Path(args.positions).name, "bones": tracks,
           **({"audio": args.audio} if args.audio else {}),
           **({"hips_pos": hips_pos} if hips_pos else {}),
           "pose_in": pose_in, "pose_out": pose_out,
           "contacts": contacts}
    Path(args.out).write_text(json.dumps(out))
    print(f"OK {args.out}: {len(tracks)} bones x {len(frames)} frames "
          f"[{pose_in}->{pose_out}]")


if __name__ == "__main__":
    main()
