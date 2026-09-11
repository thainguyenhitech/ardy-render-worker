# tts-gpu-worker

Worker TTS GPU cho RunPod serverless của dự án 3d_chatbot: Qwen3-TTS + Chatterbox Multilingual + CosyVoice 3, clone giọng
zero-shot đa ngôn ngữ. Image chỉ chứa mã + phụ thuộc; weights nằm ở network volume (`/runpod-volume`).
Build tự động bằng GitHub Actions → `ghcr.io/thainguyenhitech/tts-gpu-worker:latest`.
Tài liệu đầy đủ ở repo chính (`tts_server/README.md`, `tts_server/runpod/`).
