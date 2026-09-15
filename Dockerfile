# Tool ardy_render — worker RunPod serverless (queue): câu tiếng Anh → LLM2Vec → ARDY → clip chuẩn kho → R2.
#
# TRỌNG SỐ NƯỚNG VÀO IMAGE (user chốt 14/09): Llama-3-8B (LLM2Vec, ~15 GB) + ARDY core8 (~0,7 GB). Lớp trọng số đứng
# TRƯỚC lớp mã nên sửa mã chỉ đẩy lại vài MB — registry đã có blob thì không tải lại.
# Build context = thư mục do `dong_goi.sh` dựng (chỉ đúng các tệp tool cần), không phải gốc repo.
# IMAGE NỀN GỌN (15/09, user "làm gọn đi"): `runpod/pytorch` 10,6 GB nén (ssh, jupyter, đủ thứ pod) → `pytorch/pytorch`
# runtime 4,3 GB nén, CÙNG torch 2.8 cu128 nên hành vi số học không đổi; đã chứng minh cài được ardy trên họ image này
# (máy vast dùng pytorch/pytorch:2.5.1-…-runtime). Trọng số 8B bf16 ~14 GB nén là phần không giảm được.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    TEXT_ENCODERS_DIR=/opt/text_encoders \
    LOCAL_CACHE=true

# Bộ dựng C++ (extension MotionCorrection của ARDY) chỉ cần lúc cài — cài, pip, rồi gỡ trong CÙNG một lớp để không
# để lại ~300 MB. `cryptography` cài đè để tránh "uninstall-no-record-file" nếu image nền có bản apt.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends cmake build-essential git \
    && pip install --ignore-installed "cryptography>=42" \
    && MAKEFLAGS=-j$(nproc) pip install "git+https://github.com/nv-tlabs/ardy.git" \
       "fastapi>=0.110" "httpx>=0.27" "runpod>=1.7" "boto3>=1.34" \
    && apt-get purge -y -qq cmake build-essential && apt-get autoremove -y -qq \
    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

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
