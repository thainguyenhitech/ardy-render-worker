# IMAGE WORKER TTS GPU cho RunPod serverless — build bằng GitHub Actions (runner x86, Mac không cross-build CUDA được).
# HAI ENGINE (user 11/09): VieNeu-TTS v3 Turbo (tiếng Việt, venv_main = torch 2.8 của image) + CosyVoice 3 (ngoại ngữ,
# venv_cosy: Python 3.10 qua uv, torch 2.3.1 — không chung được). Chỉ mã + phụ thuộc trong image; WEIGHTS ở network
# volume (HF_HOME, COSY_MODEL). Giọng nhân vật (bản clone VieNeu, custom_voices.json) COPY sẵn.
FROM runpod/pytorch:1.2.0-rc.162-cu1281-torch280-ubuntu2404
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1 TOKENIZERS_PARALLELISM=false \
    UV_PYTHON_INSTALL_DIR=/opt/uv_python
RUN apt-get update && apt-get install -y --no-install-recommends git curl ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install -q --upgrade pip virtualenv uv
WORKDIR /app
# 1) venv_main — handler + VieNeu (dùng torch 2.8 của image; layer cache theo requirements_main.txt)
COPY requirements_main.txt .
RUN python3 -m virtualenv -q --system-site-packages /opt/venv_main && /opt/venv_main/bin/pip install -q -r requirements_main.txt
# 2) venv_cosy — CosyVoice 3; mã CosyVoice + FST wetext nằm trong image
COPY requirements_cosy.txt .
RUN uv venv -q --python 3.10 /opt/venv_cosy \
    && uv pip install -q --python /opt/venv_cosy/bin/python "setuptools<81" wheel \
    && uv pip install -q --python /opt/venv_cosy/bin/python --no-build-isolation openai-whisper==20231117 \
    && uv pip install -q --python /opt/venv_cosy/bin/python -r requirements_cosy.txt
RUN git clone -q --recursive --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice \
    && rm -rf /opt/CosyVoice/.git /opt/CosyVoice/third_party/Matcha-TTS/.git \
    && mkdir -p /opt/wetext/en/tn /opt/wetext/zh/tn \
    && for f in en/tn/tagger.fst en/tn/verbalizer.fst zh/tn/tagger.fst zh/tn/verbalizer.fst; do \
         curl -sfL -m 300 -o /opt/wetext/$f "https://www.modelscope.cn/models/pengzhendong/wetext/resolve/master/$f" || echo "thiếu FST $f"; done
# 3) mã worker + giọng nhân vật — đổi thường xuyên nên COPY sau cùng
COPY handler.py vieneu_server.py cosy_worker.py start.sh custom_voices.json ./
RUN chmod +x start.sh
ENV VOL=/runpod-volume COSY_DIR=/opt/CosyVoice COSY_WETEXT=/opt/wetext COSY_THIET_BI=cuda COSY_BUOC=10 COSY_CONG=8814 \
    VIENEU_CONG=8815 VIENEU_CUSTOM=/app/custom_voices.json
CMD ["/app/start.sh"]
