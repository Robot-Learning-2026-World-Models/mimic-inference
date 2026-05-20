#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
FPS="${FPS:-30}"
: "${VIDEO_MODEL_PATH:?Set VIDEO_MODEL_PATH}"
: "${ACTION_MODEL_PATH:?Set ACTION_MODEL_PATH}"
: "${TRAINING_CONFIG:?Set TRAINING_CONFIG}"

export PYTHONPATH="${PWD}/lerobot_policy_mimic_video/src:${PWD}/mimic-video/model:${PYTHONPATH:-}"

EXTRA_SERVER_ARGS=()
if [[ -n "${STATS_PATH:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--stats_path="${STATS_PATH}")
fi
if [[ -n "${ACTION_DATA_DIR:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--action_data_dir="${ACTION_DATA_DIR}")
fi
if [[ -n "${HF_STATS_REPO_ID:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--hf_stats_repo_id="${HF_STATS_REPO_ID}")
fi
if [[ -n "${HF_STATS_FILENAME:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--hf_stats_filename="${HF_STATS_FILENAME}")
fi
if [[ -n "${HF_STATS_REVISION:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--hf_stats_revision="${HF_STATS_REVISION}")
fi

exec python -m lerobot_policy_mimic_video.grpc_policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --video_model_path="${VIDEO_MODEL_PATH}" \
  --action_model_path="${ACTION_MODEL_PATH}" \
  --training_config="${TRAINING_CONFIG}" \
  "${EXTRA_SERVER_ARGS[@]}"
