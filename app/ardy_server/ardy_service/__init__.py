"""Một cục service: câu chữ + frame đầu vào -> motion streaming.

Project khác chỉ cần biết đúng một địa chỉ. Kho catalog, phép tra vector, phép hãm quán tính lịch
sử và việc nối clip đều nằm bên trong; bên gọi không phải biết ARDY tồn tại.

    python -m ardy_service                       # ARDY chạy luôn trong tiến trình này
    ARDY_URL=http://gpu-host:8100 python -m ardy_service   # ARDY chạy ở máy khác

Xem docs/ARDY_SERVER_GUIDE.md.
"""

from .app import create_app

__all__ = ["create_app"]
