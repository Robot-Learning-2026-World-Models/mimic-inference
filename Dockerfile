FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_CACHE_DIR=/root/.cache/uv \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    NVTE_FUSED_ATTN=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libglib2.0-0 \
    libgl1 \
    libvulkan1 \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace/so101-mimic-video-deploy
COPY . .

WORKDIR /workspace/so101-mimic-video-deploy/mimic-video/model
RUN uv sync --extra cu126
RUN . .venv/bin/activate && uv pip install "huggingface_hub[cli]" "lerobot[async]" && uv pip install -e /workspace/so101-mimic-video-deploy/lerobot_policy_mimic_video

WORKDIR /workspace/so101-mimic-video-deploy
ENV PATH="/workspace/so101-mimic-video-deploy/mimic-video/model/.venv/bin:${PATH}" \
    PYTHONPATH="/workspace/so101-mimic-video-deploy/mimic-video/model"

EXPOSE 8000 8080

ENTRYPOINT ["/workspace/so101-mimic-video-deploy/entrypoint.sh"]
CMD ["policy-server"]
