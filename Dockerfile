# Tool ardy_render — worker RunPod serverless (queue): câu tiếng Anh → LLM2Vec → ARDY → clip chuẩn kho → R2.
#
# TRỌNG SỐ NƯỚNG VÀO IMAGE (user chốt 14/09): Llama-3-8B (LLM2Vec, ~15 GB) + ARDY core8 (~0,7 GB). Lớp trọng số đứng
# TRƯỚC lớp mã nên sửa mã chỉ đẩy lại vài MB — registry đã có blob thì không tải lại.
# Build context = thư mục do `dong_goi.sh` dựng (chỉ đúng các tệp tool cần), không phải gốc repo.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    TEXT_ENCODERS_DIR=/opt/text_encoders \
    LOCAL_CACHE=true

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends cmake build-essential git \
    && rm -rf /var/lib/apt/lists/*

# `cryptography` 41 của image nền là gói apt (không có RECORD) — pip không gỡ được ("uninstall-no-record-file") khi một
# phụ thuộc mới đòi bản cao hơn (build thứ 3 15/09 hỏng đúng chỗ này dù hai build trước qua); cài đè không gỡ.
RUN pip install --ignore-installed "cryptography>=42"
# ARDY (ghim transformers 5.8.1, numpy<2, dựng extension C++ MotionCorrection) + phần tool cần
RUN MAKEFLAGS=-j$(nproc) pip install "git+https://github.com/nv-tlabs/ardy.git" \
    "fastapi>=0.110" "httpx>=0.27" "runpod>=1.7" "boto3>=1.34"

# ---- lớp TRỌNG SỐ (hiếm đổi) ----
COPY ardy_server/prepare_text_encoder.py /opt/prepare_text_encoder.py
RUN python /opt/prepare_text_encoder.py --out /opt/text_encoders --hf-home /opt/hf --ardy-models core8 \
    && find /opt/text_encoders -type d -name .cache -prune -exec rm -rf {} + \
    && rm -rf /root/.cache
ENV HF_HUB_OFFLINE=1

# ---- lớp MÃ ----
COPY app/ /app/
WORKDIR /app/ardy_server
CMD ["python", "-u", "ardy_render/handler.py"]
