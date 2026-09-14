"""Sổ đăng ký bộ phận — thứ tự chạy: thân trước (khoá cột sống ảnh hưởng FK
của mọi thứ phía trên), rồi đầu, hai tay, hông, chân."""
from app.modules.motion.bo_phan.chan import chan
from app.modules.motion.bo_phan.dau import dau
from app.modules.motion.bo_phan.hong import hong
from app.modules.motion.bo_phan.tay import tay_phai, tay_trai
from app.modules.motion.bo_phan.than import than

DANH_SACH = [than, dau, tay_trai, tay_phai, hong, chan]
THEO_TEN = {bp.ten: bp for bp in DANH_SACH}

__all__ = ["DANH_SACH", "THEO_TEN", "than", "dau", "tay_trai", "tay_phai", "hong", "chan"]
