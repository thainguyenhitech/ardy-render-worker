"""Render MỘT câu tiếng Anh thành clip đúng chuẩn kho: công thức `ardy_kho.render_mot` + tự kiểm theo `kiem_tai_ve`.

Không tra catalog/kho trước khi render (user 14/09: backend chỉ gửi câu khi tìm kho không thấy câu phù hợp).

Chạy thử cục bộ (Mac: Engine CPU, encoder MPS):
    cd ardy_server && ARDY_ENCODER_DEVICE=mps .venv/bin/python -m ardy_render.render --cau "A person waves." --ra /tmp/ra
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

from ardy_render.nguon import NguonTrong
import ardy_kho as AK

TAI_SAN = Path(__file__).with_name("tai_san")
_IDLE: tuple | None = None


def slug(cau: str) -> str:
    """Tên tag y quy tắc `ardy_kho --cau` (80 ký tự)."""
    return re.sub(r"[^a-z0-9]+", "_", cau.lower()).strip("_")[:80]


def _idle() -> tuple:
    global _IDLE
    if _IDLE is None:
        if not (TAI_SAN / "_idle_tho.json").exists():
            raise FileNotFoundError("thiếu tai_san/_idle_tho.json — idle phải CHÉP từ kho, không sinh lại (lệch tới 18°)")
        _IDLE = AK.idle_dong_bang(TAI_SAN)
    return _IDLE


def tu_kiem(clip: dict, idle_clip: dict) -> list[str]:
    """Đúng các phép của `kiem_tai_ve.cham_mot`: cấu trúc · lưới · mốc tăng · hữu hạn · đầu/cuối thân trên so idle."""
    loi = []
    b = clip.get("bones") or {}
    if not all(k in clip for k in ("fps", "duration_s", "bones")) or "Hips" not in b:
        return ["thiếu khoá fps/duration_s/bones/Hips"]
    n = len(b["Hips"])
    if any(len(tr) != n for tr in b.values()):
        loi.append("track lệch số khung")
    fps = float(clip["fps"])
    if abs(n - (round(float(clip["duration_s"]) * fps) + 1)) > 1:
        loi.append(f"{n} khung ≠ duration_s {clip['duration_s']} × {fps}")
    t = [f[0] for f in b["Hips"]]
    if any(t2 <= t1 or t2 - t1 > 1.5 / fps for t1, t2 in zip(t, t[1:])):
        loi.append("mốc Hips không tăng đều")
    if any(not math.isfinite(x) for tr in b.values() for f in tr for x in f) or \
            any(not math.isfinite(x) for f in clip.get("hips_pos") or [] for x in f):
        loi.append("có giá trị không hữu hạn")
    tren = lambda p: {x: v for x, v in p.items() if x not in AK.CHAN}  # noqa: E731
    idle_pose = tren(AK.tu_the(idle_clip, len(idle_clip["bones"]["Hips"]) - 1))
    dau = AK.lech(tren(AK.tu_the(clip, 0)), idle_pose)[0]
    cuoi = AK.lech(tren(AK.tu_the(clip, n - 1)), idle_pose)[0]
    if dau > AK.KIEM_DAU_MAX:
        loi.append(f"khung đầu lệch idle {dau:.1f}° > {AK.KIEM_DAU_MAX}")
    if cuoi > AK.KIEM_CUOI_MAX:
        loi.append(f"khung cuối lệch idle {cuoi:.1f}° > {AK.KIEM_CUOI_MAX}")
    return loi


def render_cau(cau: str, nguon: NguonTrong, giay: float = AK.GIAY_BIEN_THE[0], bien_the: int = 0) -> dict:
    """{cau, tag, tep, giay, seed, dat, loi?, meta?, clip?} — `clip` + `meta` (mục kho.json) chỉ có khi đạt."""
    cau = " ".join(str(cau).split())
    tag = slug(cau)
    ra = {"cau": cau, "tag": tag, "tep": AK._ten_tep(tag, bien_the), "giay": float(giay),
          "seed": nguon.seed(cau, giay, False), "dat": False}
    if not tag:
        ra["loi"] = "câu rỗng sau chuẩn hoá"
        return ra
    AK.dat_nguon(nguon)
    t0 = time.time()
    try:
        hist, dich, idle_clip = _idle()
        r = AK.render_mot(tag, cau, float(giay), hist, dich, idle_clip)
    except Exception as e:  # noqa: BLE001
        r = {"loi": f"hỏng: {type(e).__name__}: {str(e)[:160]}"}
    ra["sinh_s"] = round(time.time() - t0, 1)
    if "clip" not in r:
        ra["loi"] = r.get("loi")
        if r.get("meta"):
            ra["meta"] = r["meta"]
        return ra
    loi = tu_kiem(r["clip"], idle_clip)
    if loi:
        ra["loi"] = "tự kiểm: " + "; ".join(loi)
        ra["meta"] = r["meta"]
        return ra
    ra.update(dat=True, clip=r["clip"],
              meta={"tep": ra["tep"], "bien_the": bien_the, **r["meta"], "nguon_render": "ardy_render", "seed": ra["seed"]})
    return ra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cau", action="append", default=[], help="câu tiếng Anh (lặp lại được)")
    ap.add_argument("--tu-tep", help="tệp mỗi dòng một câu")
    ap.add_argument("--giay", type=float, default=AK.GIAY_BIEN_THE[0])
    ap.add_argument("--ra", required=True, help="thư mục ghi <tep> + <tep>.muc.json / .loi.json")
    a = ap.parse_args()
    caus = list(a.cau)
    if a.tu_tep:
        caus += [s.strip() for s in Path(a.tu_tep).read_text(encoding="utf-8").splitlines() if s.strip() and not s.startswith("#")]
    out = Path(a.ra)
    out.mkdir(parents=True, exist_ok=True)
    ng = NguonTrong()
    ng.nap()
    if not ng.san_sang:
        raise SystemExit(f"nạp hỏng: {ng.loi}")
    print(f"nạp {ng.nap_s}", flush=True)
    dat = 0
    for c in caus:
        kq = render_cau(c, ng, a.giay)
        if kq["dat"]:
            dat += 1
            (out / kq["tep"]).write_text(json.dumps(kq["clip"]))
            (out / (kq["tep"] + ".muc.json")).write_text(json.dumps(kq["meta"], ensure_ascii=False, indent=1))
        else:
            (out / (kq["tep"] + ".loi.json")).write_text(
                json.dumps({k: v for k, v in kq.items() if k != "clip"}, ensure_ascii=False, indent=1))
        m = kq.get("meta") or {}
        print(f"{'ĐẠT ' if kq['dat'] else 'TRƯỢT'} {kq['sinh_s']:5.1f}s · dài {m.get('dai_s')} · đầu {m.get('dau_lech_idle')}° "
              f"cuối {m.get('cuoi_lech_idle')}° · {kq.get('loi') or ''} · {c[:70]}", flush=True)
    print(f"XONG {dat}/{len(caus)} đạt → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
