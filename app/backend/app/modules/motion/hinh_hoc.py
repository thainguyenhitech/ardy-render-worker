"""HÌNH HỌC XOAY — quaternion, véc-tơ xoay, lấy mẫu track, nạp rig.

Hai quy tắc sống còn (đã trả giá để học được):
1. KHÔNG đạo hàm trên góc Euler — Euler quấn qua ±360° sinh đỉnh giả.
   Mọi phép so/đạo hàm đi qua quaternion ép dấu liên tục hoặc log map.
2. _lech tính trong KHUNG CHA: d = qa · qb⁻¹. Nhân sai phía là tư thế dựng
   lại không trùng và mối nối còn 3-7 m/s dù về lý phải bằng 0.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from app.modules.motion.cau_hinh import _BUILD, _GLB



@lru_cache(maxsize=1)
def _rig():
    import sys
    sys.path.insert(0, str(_BUILD))
    import build_clip as B
    tg = B.prep_target(str(_GLB))
    n2i, par, rr = tg["name2idx"], tg["parent"], tg["rest_rot"]
    rp = {}
    for i in rr:
        p = par.get(i)
        if p is None:
            rp[i] = tg["rest_world"][i][1]
        else:
            pq, pp = tg["rest_world"][p]
            w, x, y, z = pq
            rp[i] = B.q_rot(np.array([w, -x, -y, -z]),
                            tg["rest_world"][i][1] - pp)
    nghi = {n: [float(v) for v in B.q_to_euler_xyz(rr[i])]
            for n, i in n2i.items()}
    return B, n2i, par, rr, rp, nghi

def _mau(B, tr, t):
    n = len(tr)
    t = max(tr[0][0], min(tr[-1][0], t))
    lo, hi = 0, n - 1
    while lo < hi:
        m = (lo + hi + 1) // 2
        if tr[m][0] <= t:
            lo = m
        else:
            hi = m - 1
    f = tr[lo]
    return B.q_from_euler_xyz(f[1], f[2], f[3])

def _q_mul(a, b):
    return (a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
            a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
            a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0])

def _lech(qa, qb):
    """Góc và trục của phép xoay đưa qb về qa, TRONG KHUNG CHA: d = qa · qb⁻¹.

    Thứ tự nhân là chỗ dễ sai nhất ở đây. Bản đầu tôi tính conj(qb)·qa — đó là
    độ lệch trong khung CỤC BỘ của qb — rồi lại áp bằng phép nhân bên TRÁI
    (khung cha). Hai khung không khớp nên tư thế dựng lại không trùng cuối A,
    và mối ghép còn 3,3-6,9 m/s dù về lý phải bằng 0.
    """
    d = _q_mul(qa, (qb[0], -qb[1], -qb[2], -qb[3]))
    dw = abs(d[0])
    goc = 2 * math.acos(min(1.0, dw))
    s = math.sqrt(max(1e-12, 1 - dw * dw))
    sg = 1.0 if d[0] >= 0 else -1.0
    return goc, [d[1] * sg / s, d[2] * sg / s, d[3] * sg / s]

def _log_q(q):
    """log map: quaternion -> véc-tơ xoay (rad)."""
    w = max(-1.0, min(1.0, float(q[0])))
    v = np.array([float(q[1]), float(q[2]), float(q[3])])
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3)
    return v / n * (2.0 * math.atan2(n, w))

def _exp_q(v):
    """exp map: véc-tơ xoay -> quaternion."""
    g = float(np.linalg.norm(v))
    if g < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(g / 2) / g
    return (math.cos(g / 2), float(v[0]) * s, float(v[1]) * s, float(v[2]) * s)

# ---------------------------------------------------------------- tầng 6
def _catmull(p0, p1, p2, p3, u):
    """Catmull-Rom: đi QUA mọi điểm gốc và có đạo hàm liên tục — không còn
    'góc' ở mỗi khung như nội suy tuyến tính."""
    u2 = u * u
    u3 = u2 * u
    return 0.5 * ((2 * p1) + (-p0 + p2) * u
                  + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)


# ---------------------------------------------------------------- vector hoá theo thời gian
def _q_euler_vec(ex, ey, ez):
    """Euler XYZ intrinsic -> quat (T,4), cùng quy ước build_clip.q_from_euler_xyz."""
    cx, sx = np.cos(ex / 2), np.sin(ex / 2)
    cy, sy = np.cos(ey / 2), np.sin(ey / 2)
    cz, sz = np.cos(ez / 2), np.sin(ez / 2)
    w, x, y, z = cx * cy, sx * cy, cx * sy, sx * sy          # qx*qy
    return np.stack([w * cz - z * sz, x * cz + y * sz, y * cz - x * sz, w * sz + z * cz], axis=1)


def _euler_q_vec(q):
    """quat (T,4) -> Euler XYZ intrinsic (T,3), cùng quy ước build_clip.q_to_euler_xyz."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m00 = 1 - 2 * (y * y + z * z)
    m01 = 2 * (x * y - z * w)
    m02 = 2 * (x * z + y * w)
    m11 = 1 - 2 * (x * x + z * z)
    m12 = 2 * (y * z - x * w)
    m21 = 2 * (y * z + x * w)
    m22 = 1 - 2 * (x * x + y * y)
    ey = np.arcsin(np.clip(m02, -1, 1))
    binh = np.abs(m02) < 0.99999
    ex = np.where(binh, np.arctan2(-m12, m22), np.arctan2(m21, m11))
    ez = np.where(binh, np.arctan2(-m01, m00), 0.0)
    return np.stack([ex, ey, ez], axis=1)


def _log_q_vec(q):
    """log map (T,4) -> véc-tơ xoay (T,3)."""
    w = np.clip(q[:, 0], -1.0, 1.0)
    v = q[:, 1:]
    n = np.linalg.norm(v, axis=1)
    goc = 2.0 * np.arctan2(n, w)
    he = np.where(n < 1e-9, 0.0, goc / np.where(n < 1e-9, 1.0, n))
    return v * he[:, None]


def _exp_q_vec(v):
    """exp map (T,3) -> quat (T,4)."""
    g = np.linalg.norm(v, axis=1)
    s = np.where(g < 1e-12, 0.5, np.sin(g / 2) / np.where(g < 1e-12, 1.0, g))
    return np.concatenate([np.cos(g / 2)[:, None], v * s[:, None]], axis=1)


def _lien_tuc(q):
    """Ép dấu quaternion liên tục theo thời gian (q và -q là một phép xoay)."""
    d = np.sum(q[1:] * q[:-1], axis=1)
    dau = np.cumprod(np.where(d < 0, -1.0, 1.0))
    q = q.copy()
    q[1:] *= dau[:, None]
    return q
