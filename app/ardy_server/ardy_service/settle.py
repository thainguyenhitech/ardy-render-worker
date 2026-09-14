"""Đo xem một động tác đã KẾT THÚC hay chưa.

ARDY tự hồi quy: nó sinh 8 khung một cửa sổ và sinh mãi, không phát tín hiệu kết thúc nào. Nên độ
dài một động tác không phải con số người viết đặt ra, mà là thứ phải ĐO trên chính chuyển động.

Đo trên 6 loại động tác, mức chuyển động trung bình mỗi khung theo từng giây:

    "sits down on the floor"   3,26 -> 1,12 -> 0,24 0,23 0,20 0,22 ...   lắng thật ở giây 3
    "bows the head slightly"   0,51 0,55 0,74 0,17 0,99 0,87 0,37 ...    biên độ nhỏ, nhập nhằng
    "dances the cha-cha"       2,50 4,09 5,41 5,32 4,21 4,37 3,98 ...    tuần hoàn, không bao giờ lắng
    "runs in a circle"         3,56 3,19 1,04 0,62 4,22 5,14 4,92 ...    trũng 2 giây lúc vào cua

Hai loại sai mà một phép đo đơn lẻ mắc phải:

  - Chỉ nhìn VẬN TỐC KHUNG: cái cúi đầu chạy 27,5 giây không dứt, vì lúc nó đang diễn và lúc nó
    đứng yên có vận tốc gần bằng nhau.
  - Chỉ nhìn DỊCH CHUYỂN RÒNG: điệu nhảy quay lại đúng tư thế cũ sau mỗi chu kỳ nên bị coi là xong.

Nên phải đủ CẢ HAI: vận tốc khung thấp, VÀ tư thế gần như không đổi so với `hold` giây trước.
"""
from __future__ import annotations

import os

import numpy as np

FLOOR = float(os.getenv("MOTION_SETTLE_FLOOR", "0.9"))  # độ mỗi khung, sàn tuyệt đối
RATIO = float(os.getenv("MOTION_SETTLE_RATIO", "0.25"))  # so với đỉnh của chính động tác
HOLD = float(os.getenv("MOTION_SETTLE_HOLD", "3.0"))  # giây phải liên tục ở dưới ngưỡng
NET = float(os.getenv("MOTION_SETTLE_NET", "12.0"))  # độ, giữa hai đầu cửa sổ

__all__ = ["FLOOR", "RATIO", "HOLD", "NET", "frame_energy", "quats_of", "settled", "Settle"]


def quats_of(clip: dict) -> np.ndarray:
    """[N, J, 4] quaternion của một clip."""
    return np.asarray([f["data"]["quats"] for f in clip["frames"]], dtype=np.float32)


def frame_energy(q: np.ndarray) -> np.ndarray:
    """Mức chuyển động của từng khung: lệch góc trung bình trên các khớp, đơn vị độ."""
    if len(q) < 2:
        return np.zeros(0, dtype=np.float32)
    d = np.abs(np.sum(q[1:] * q[:-1], axis=-1)).clip(0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d)).mean(axis=1)


def settled(energy: np.ndarray, window: np.ndarray, peak: float, fps: int) -> bool:
    """Đã lắng chưa. Ngưỡng vận tốc lấy theo đỉnh của chính động tác đó chứ không cố định, vì biên
    độ một cái gật đầu và một điệu nhảy chênh nhau cả bậc độ lớn."""
    hold = int(HOLD * fps)
    if len(energy) < hold or len(window) < hold + 1:
        return False
    if float(energy[-hold:].mean()) >= max(FLOOR, RATIO * peak):
        return False
    d = np.abs(np.sum(window[0] * window[-1], axis=-1)).clip(0.0, 1.0)
    return bool(np.degrees(2.0 * np.arccos(d)).mean() < NET)


class Settle:
    """Bộ đếm trạng thái cho một động tác đang stream: nạp từng clip, hỏi đã xong chưa."""

    def __init__(self, fps: int):
        self.fps = fps
        self.keep = int(HOLD * fps) + 1
        self.peak = 0.0
        self.energy = np.zeros(0, dtype=np.float32)
        self.window = np.zeros((0, 0, 4), dtype=np.float32)
        self.last = 0.0  # mức chuyển động của clip vừa nạp

    def feed(self, clip: dict) -> bool:
        q = quats_of(clip)
        e = frame_energy(q)
        self.last = float(e.mean()) if len(e) else 0.0
        self.peak = max(self.peak, float(e.max()) if len(e) else 0.0)
        self.energy = np.concatenate([self.energy, e])[-self.keep :]
        self.window = (q if self.window.size == 0 else np.concatenate([self.window, q]))[-self.keep :]
        return settled(self.energy, self.window, self.peak, self.fps)
