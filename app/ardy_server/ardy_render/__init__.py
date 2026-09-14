"""Tool `ardy_render`: câu tiếng Anh (LLM sinh) → mã hoá LLM2Vec → ARDY → clip motion ĐÚNG CHUẨN kho → R2.

nguon.py   Engine ARDY + encoder trong tiến trình, cắm vào `ardy_kho.dat_nguon`
render.py  render một câu bằng công thức `ardy_kho.render_mot` + tự kiểm theo `kiem_tai_ve`; CLI chạy thử cục bộ
r2.py      đẩy clip + mục chỉ mục riêng từng clip lên kho production trên R2
handler.py worker RunPod serverless (queue)
"""
