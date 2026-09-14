"""Đo theo bộ phận — vận tốc/gia tốc WORLD ở điểm đo, vận tốc/gia tốc GÓC của
xương. Mọi đạo hàm trên MỐC GỐC của clip; góc trên véc-tơ xoay LIÊN TỤC."""
from __future__ import annotations

import numpy as np

from app.modules.motion.hinh_hoc import _log_q, _mau, _rig


def moc_goc(clip: dict) -> list[float]:
    b = clip.get("bones") or {}
    return sorted({f[0] for tr in b.values() for f in tr})


_FK = {"khoa": None, "moc": None, "pos": None, "n2i": None}


def _dau_van(clip: dict) -> tuple:
    """Dấu vân rẻ tiền của nội dung clip — đổi một số là đổi dấu. Dùng để
    dùng lại FK giữa các lần đo liên tiếp (đo là phần đắt nhất của ống)."""
    b = clip.get("bones") or {}
    tong = 0.0
    n = 0
    for tr in b.values():
        n += len(tr)
        for f in tr:
            tong += f[0] * 0.37 + f[1] + f[2] * 1.71 + f[3] * 2.29
    return (id(clip), n, round(tong, 6))


def _q_euler_vec(ex, ey, ez):
    """Euler XYZ intrinsic -> quat (T,4), cùng quy ước build_clip.q_from_euler_xyz."""
    cx, sx = np.cos(ex / 2), np.sin(ex / 2)
    cy, sy = np.cos(ey / 2), np.sin(ey / 2)
    cz, sz = np.cos(ez / 2), np.sin(ez / 2)
    # qx*qy = (cx cy, sx cy, cx sy, -sx sy)  ;  (.)*qz
    w, x, y, z = cx * cy, sx * cy, cx * sy, sx * sy
    return np.stack([w * cz - z * sz, x * cz + y * sz, y * cz - x * sz, w * sz + z * cz], axis=1)


def _q_mul_vec(a, b):
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=1)


def _q_rot_vec(q, v):
    """Xoay véc-tơ hằng v (3,) bởi quat đơn vị q (T,4) -> (T,3)."""
    u = q[:, 1:]
    w = q[:, :1]
    c = np.cross(u, v[None, :])
    return v[None, :] + 2.0 * np.cross(u, c + w * v[None, :])


def _fk_tat_ca(clip: dict):
    """FK vector hoá theo thời gian: (mốc, vị trí world mọi khớp (T, J, 3), cột)."""
    B, n2i, par, rr, rp, _ = _rig()
    b = clip.get("bones") or {}
    moc = np.array(moc_goc(clip), dtype=float)
    T = len(moc)
    Q = {}
    for i in rr:
        Q[i] = np.tile(np.asarray(rr[i], float), (T, 1))
    for ten, i in n2i.items():
        tr = b.get(ten)
        if tr:
            arr = np.asarray(tr, dtype=float)
            if arr.shape[0] == T and np.allclose(arr[:, 0], moc):
                Q[i] = _q_euler_vec(arr[:, 1], arr[:, 2], arr[:, 3])
            else:
                # NỘI SUY TRÊN QUATERNION, KHÔNG TRÊN EULER (CLAUDE.md §4 bẫy 1):
                # np.interp từng kênh Euler qua chỗ Euler bọc ±π ra rác — đo
                # 28/08: FK từng khung đúng 0,0 mà thước báo bàn tay "dịch 73
                # cm" trên mọi khung. Lấy 2 khung kề, ép dấu quaternion liên
                # tục, nlerp rồi chuẩn hoá.
                qk = _q_euler_vec(arr[:, 1], arr[:, 2], arr[:, 3])
                doi = np.sign(np.einsum("ij,ij->i", qk[1:], qk[:-1])); doi[doi == 0] = 1.0
                qk[1:] *= np.cumprod(doi)[:, None]
                j = np.clip(np.searchsorted(arr[:, 0], moc, side="right") - 1, 0, len(arr) - 2)
                t0, t1 = arr[j, 0], arr[j + 1, 0]
                w = np.clip((moc - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)[:, None]
                q = (1 - w) * qk[j] + w * qk[j + 1]
                Q[i] = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    Wq, Wp = {}, {}
    con_lai = list(rr.keys())
    while con_lai:
        sau = []
        for i in con_lai:
            p = par.get(i)
            if p is None:
                Wq[i] = Q[i]
                Wp[i] = np.tile(np.asarray(rp[i], float), (T, 1))
            elif p in Wq:
                Wq[i] = _q_mul_vec(Wq[p], Q[i])
                Wp[i] = Wp[p] + _q_rot_vec(Wq[p], np.asarray(rp[i], float))
            else:
                sau.append(i)
        if len(sau) == len(con_lai):          # cha không có trong rr — coi như gốc
            for i in sau:
                Wq[i] = Q[i]
                Wp[i] = np.tile(np.asarray(rp[i], float), (T, 1))
            sau = []
        con_lai = sau
    cot = {i: k for k, i in enumerate(Wp)}
    return moc, np.stack([Wp[i] for i in Wp], axis=1), cot


_FK = {"khoa": None, "moc": None, "pos": None, "cot": None}


def _dau_van(clip: dict) -> tuple:
    """Dấu vân rẻ tiền của nội dung clip — đổi một số là đổi dấu. Dùng để
    dùng lại FK giữa các lần đo liên tiếp."""
    b = clip.get("bones") or {}
    tong = 0.0
    n = 0
    for tr in b.values():
        n += len(tr)
        for f in tr:
            tong += f[0] * 0.37 + f[1] + f[2] * 1.71 + f[3] * 2.29
    return (id(clip), n, round(tong, 6))


def vi_tri_world(clip: dict, diem: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(mốc thời gian, vị trí (T, n_diem, 3)) của các điểm đo.
    FK vector hoá, tính MỘT lần cho mọi khớp rồi cache theo dấu vân clip."""
    _, n2i, *_ = _rig()
    idx = [n2i[d] for d in diem if d in n2i]
    khoa = _dau_van(clip)
    if _FK["khoa"] != khoa:
        moc, pos, cot = _fk_tat_ca(clip)
        _FK.update(khoa=khoa, moc=moc, pos=pos, cot=cot)
    moc = _FK["moc"]
    if not idx or len(moc) < 3:
        return np.array(moc), np.zeros((len(moc), 0, 3))
    return moc.copy(), _FK["pos"][:, [_FK["cot"][h] for h in idx], :]


def toc_world(clip: dict, diem: tuple[str, ...]) -> dict:
    """v95, a95, v_max_khung (m/s) — bắt cả vọt một khung."""
    moc, P = vi_tri_world(clip, diem)
    if P.size == 0 or len(moc) < 3:
        return {"v95": 0.0, "a95": 0.0, "v_max": 0.0}
    dt = np.diff(moc)
    dt[dt <= 1e-9] = 1e-9
    v = np.linalg.norm(np.diff(P, axis=0), axis=2) / dt[:, None]
    a = np.linalg.norm(np.diff(np.diff(P, axis=0) / dt[:, None, None], axis=0), axis=2) / dt[:-1, None]
    return {"v95": float(np.percentile(v, 95)), "a95": float(np.percentile(a, 95)),
            "v_max": float(v.max())}


def toc_goc(clip: dict, xuong: tuple[str, ...]) -> dict:
    """Vận tốc/gia tốc GÓC (rad/s, rad/s²) gộp các xương của bộ phận."""
    B, *_ = _rig()
    b = clip.get("bones") or {}
    vs, as_ = [], []
    for ten in xuong:
        tr = b.get(ten)
        if not tr or len(tr) < 4:
            continue
        t = np.array([f[0] for f in tr])
        R, prev = [], None
        for f in tr:
            q = B.q_from_euler_xyz(f[1], f[2], f[3])
            w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            if prev is not None and w * prev[0] + x * prev[1] + y * prev[2] + z * prev[3] < 0:
                w, x, y, z = -w, -x, -y, -z
            prev = (w, x, y, z)
            R.append(_log_q((w, x, y, z)))
        R = np.array(R)
        d = np.diff(t)
        d[d <= 1e-9] = 1e-9
        w_ = np.diff(R, axis=0) / d[:, None]
        a_ = np.diff(w_, axis=0) / d[:-1, None]
        vs += list(np.linalg.norm(w_, axis=1))
        as_ += list(np.linalg.norm(a_, axis=1))
    if not vs:
        return {"v95": 0.0, "a95": 0.0, "v_max": 0.0}
    return {"v95": float(np.percentile(vs, 95)), "a95": float(np.percentile(as_, 95)) if as_ else 0.0,
            "v_max": float(max(vs))}


def chuoi_gia_toc(clip: dict, diem: tuple[str, ...]):
    """(mốc giữa, gia tốc world m/s² lớn nhất trong các điểm đo) — chuỗi theo
    thời gian, để tìm CHỖ vọt chứ không chỉ con số tổng."""
    moc, P = vi_tri_world(clip, diem)
    if P.size == 0 or len(moc) < 4:
        return np.array([]), np.array([])
    dt = np.diff(moc)
    dt[dt <= 1e-9] = 1e-9
    v = np.diff(P, axis=0) / dt[:, None, None]
    a = np.linalg.norm(np.diff(v, axis=0), axis=2) / dt[:-1, None]
    return np.array(moc[1:-1]), a.max(axis=1)


def chuoi_gia_toc_goc(clip: dict, xuong: tuple[str, ...]):
    """(mốc giữa, gia tốc GÓC rad/s² lớn nhất trong các xương) — cùng mục đích."""
    B, *_ = _rig()
    b = clip.get("bones") or {}
    ket = None
    for ten in xuong:
        tr = b.get(ten)
        if not tr or len(tr) < 4:
            continue
        t = np.array([f[0] for f in tr])
        R, prev = [], None
        for f in tr:
            q = B.q_from_euler_xyz(f[1], f[2], f[3])
            w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            if prev is not None and w * prev[0] + x * prev[1] + y * prev[2] + z * prev[3] < 0:
                w, x, y, z = -w, -x, -y, -z
            prev = (w, x, y, z)
            R.append(_log_q((w, x, y, z)))
        R = np.array(R)
        d = np.diff(t)
        d[d <= 1e-9] = 1e-9
        a = np.linalg.norm(np.diff(np.diff(R, axis=0) / d[:, None], axis=0), axis=1) / d[:-1]
        tm = t[1:-1]
        if ket is None:
            ket = (tm, a)
        else:
            n = min(len(ket[1]), len(a))
            ket = (ket[0][:n], np.maximum(ket[1][:n], a[:n]))
    return ket if ket else (np.array([]), np.array([]))
