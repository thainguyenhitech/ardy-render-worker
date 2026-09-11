#!/usr/bin/env bash
# Khởi động trong IMAGE đóng gói sẵn (Dockerfile): venv/mã/FST đã có, chỉ WEIGHTS ở volume (HF_HOME, COSY_MODEL).
# Handler chạy NGAY (RunPod thu hồi worker chưa "ready" sau ~9 phút); hai engine con nạp ở nền:
#   vieneu_server (VieNeu v3 Turbo, CUDA, 8815) · cosy_worker (CosyVoice 3, 8814 — tải model vào volume nếu thiếu).
V=${VOL:-/runpod-volume}
export HF_HOME=$V/hf
mkdir -p "$HF_HOME"
exec > >(tee -a "$V/bootstrap.log") 2>&1
echo "===== $(date -u +%FT%TZ) start.sh (image) · GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
/opt/venv_main/bin/python -u /app/vieneu_server.py >> "$V/vieneu_server.log" 2>&1 &
COSY_MODEL=${COSY_MODEL:-$V/Fun-CosyVoice3-0.5B}
export COSY_MODEL
(
  if [ ! -f "$COSY_MODEL/llm.pt" ]; then
    echo "[nền] tải Fun-CosyVoice3-0.5B → $COSY_MODEL"
    /opt/venv_main/bin/python -c "from huggingface_hub import snapshot_download as s; s('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='$COSY_MODEL')" || echo "[nền] LỖI tải model Cosy"
  fi
  [ -f "$COSY_MODEL/llm.pt" ] && /opt/venv_cosy/bin/python -u /app/cosy_worker.py --cong "${COSY_CONG:-8814}" >> "$V/cosy_worker.log" 2>&1 &
  echo "[nền] cosy_worker bật"
) &
exec /opt/venv_main/bin/python -u /app/handler.py
