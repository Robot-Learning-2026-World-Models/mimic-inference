#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PATH="${REPO_ROOT}/mimic-video/model/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/lerobot_policy_mimic_video/src:${REPO_ROOT}/mimic-video/model"
export COSMOS_PREDICT2_ARGS="--checkpoints ${REPO_ROOT}/checkpoints"

exec python scripts/smoke_mimic_video_grpc_client.py \
  --server-address "${SERVER_ADDRESS:-127.0.0.1:8080}" \
  --raw-image-horizon 13 \
  --actions-per-chunk 60 \
  --timeout 600
