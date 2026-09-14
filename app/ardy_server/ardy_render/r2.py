"""Kho production trên R2 — worker GHI, backend đồng bộ về.

    <prefix>/clip/<tep>              clip JSON đúng định dạng kho
    <prefix>/muc/<tep>.json          MỘT mục chỉ mục cho MỘT clip (đủ trường như kho.json)
    <prefix>/loi/<yyyymmdd>/<…>.json câu trượt cổng (lý do + số đo) để soi

MỖI CLIP MỘT TỆP CHỈ MỤC, không ghi chung `kho.json`: nhiều worker ghi cùng một khoá là đè nhau → clip về đủ mà thành
MỒ CÔI (CLAUDE.md §10b, 13/09: kéo 90 clip, hợp nhất +3, 88 mồ côi). Ghi CLIP TRƯỚC, MỤC SAU: mục không bao giờ trỏ vào
clip chưa có. Trùng tên (slug 80 ký tự của hai câu khác nhau) thì thêm hậu tố hash — không ghi đè nội dung khác lên cùng
tên (đo 13/09: 4 cặp khác câu cùng tên trong danh sách 34.799 câu).

Biến môi trường (cùng tên với `~/project/motion_trainer/cloud/r2.env`; trên RunPod đặt bằng secret):
    R2_BUCKET · RCLONE_CONFIG_R2_ENDPOINT · RCLONE_CONFIG_R2_ACCESS_KEY_ID · RCLONE_CONFIG_R2_SECRET_ACCESS_KEY
    ARDY_KHO_R2 (prefix, mặc định ardy_kho_prod)
"""
from __future__ import annotations

import hashlib
import json
import os
import time

PREFIX = os.getenv("ARDY_KHO_R2", "ardy_kho_prod").strip("/")


class KhoR2:
    def __init__(self, s3, bucket: str, prefix: str = PREFIX) -> None:
        self.s3, self.bucket, self.prefix = s3, bucket, prefix

    @classmethod
    def mo(cls) -> "KhoR2 | None":
        can = ("R2_BUCKET", "RCLONE_CONFIG_R2_ENDPOINT", "RCLONE_CONFIG_R2_ACCESS_KEY_ID", "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY")
        if not all(os.getenv(k) for k in can):
            return None
        import boto3
        s3 = boto3.client("s3", endpoint_url=os.environ["RCLONE_CONFIG_R2_ENDPOINT"], region_name="auto",
                          aws_access_key_id=os.environ["RCLONE_CONFIG_R2_ACCESS_KEY_ID"],
                          aws_secret_access_key=os.environ["RCLONE_CONFIG_R2_SECRET_ACCESS_KEY"])
        return cls(s3, os.environ["R2_BUCKET"])

    def _doc(self, key: str) -> dict | None:
        try:
            return json.loads(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read())
        except self.s3.exceptions.NoSuchKey:
            return None

    def _ghi(self, key: str, obj) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, ContentType="application/json",
                           Body=json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def day(self, kq: dict) -> dict:
        """Đẩy clip đạt. Trả {clip, muc, tep, ghi_de_cung_cau}."""
        tag, meta, bt = kq["tag"], dict(kq["meta"]), int(kq["meta"].get("bien_the", 0))
        tep = kq["tep"]
        cu = self._doc(f"{self.prefix}/muc/{tep}.json")
        if cu is not None and " ".join(str(cu.get("prompt", "")).lower().split()) != " ".join(kq["cau"].lower().split()):
            tag = f"{tag[:72]}_{hashlib.sha256(kq['cau'].encode('utf-8')).hexdigest()[:6]}"
            tep = f"{tag}__{bt}.json"
        meta.update(tep=tep, tag=tag, r2_luc=round(time.time(), 1))
        clip = kq["clip"]
        self._ghi(f"{self.prefix}/clip/{tep}", clip)
        self._ghi(f"{self.prefix}/muc/{tep}.json", meta)
        # SỔ THEO THỜI GIAN, ghi SAU mục: backend đồng bộ định kỳ chỉ liệt kê phần sổ sau khoá cuối nó đã xử lý —
        # không phải quét lại cả `muc/` (hàng chục nghìn khoá) mỗi chu kỳ. Khoá UTC tới mili giây nên tăng dần.
        ms = int(time.time() * 1000)
        self._ghi(f"{self.prefix}/nhat_ky/{time.strftime('%Y%m%dT%H%M%S', time.gmtime(ms / 1000))}{ms % 1000:03d}_{tep}",
                  {"muc": f"{self.prefix}/muc/{tep}.json"})
        return {"clip": f"{self.prefix}/clip/{tep}", "muc": f"{self.prefix}/muc/{tep}.json", "tep": tep,
                "ghi_de_cung_cau": cu is not None and tep == kq["tep"]}

    def ghi_loi(self, kq: dict) -> str:
        key = f"{self.prefix}/loi/{time.strftime('%Y%m%d')}/{int(time.time() * 1000)}_{kq['tag'][:60]}.json"
        self._ghi(key, {k: v for k, v in kq.items() if k != "clip"})
        return key
