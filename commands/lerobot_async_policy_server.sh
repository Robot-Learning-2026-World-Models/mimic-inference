#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
FPS="${FPS:-30}"
: "${TRAINING_CONFIG:?Set TRAINING_CONFIG}"
CHECKPOINT_CACHE_DIR="${CHECKPOINT_CACHE_DIR:-/checkpoints}"
MIMIC_VIDEO_HF_REPO_ID="${MIMIC_VIDEO_HF_REPO_ID:-rl26-world-models/mimic-finetune}"
PREPULL="${PREPULL:-1}"

export PYTHONPATH="${PWD}/lerobot_policy_mimic_video/src:${PWD}/mimic-video/model:${PYTHONPATH:-}"
export MIMIC_VIDEO_HF_REPO_ID

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
if [[ -n "${VIDEO_MODEL_PATH:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--video_model_path="${VIDEO_MODEL_PATH}")
fi
if [[ -n "${ACTION_MODEL_PATH:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--action_model_path="${ACTION_MODEL_PATH}")
fi
if [[ -n "${VIDEO_MODEL_HF_REPO_ID:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--video_model_hf_repo_id="${VIDEO_MODEL_HF_REPO_ID}")
fi
if [[ -n "${VIDEO_MODEL_HF_FILENAME:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--video_model_hf_filename="${VIDEO_MODEL_HF_FILENAME}")
fi
if [[ -n "${VIDEO_MODEL_HF_REVISION:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--video_model_hf_revision="${VIDEO_MODEL_HF_REVISION}")
fi
if [[ -n "${VIDEO_MODEL_HF_REPO_TYPE:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--video_model_hf_repo_type="${VIDEO_MODEL_HF_REPO_TYPE}")
fi
if [[ -n "${ACTION_MODEL_HF_REPO_ID:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--action_model_hf_repo_id="${ACTION_MODEL_HF_REPO_ID}")
fi
if [[ -n "${ACTION_MODEL_HF_FILENAME:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--action_model_hf_filename="${ACTION_MODEL_HF_FILENAME}")
fi
if [[ -n "${ACTION_MODEL_HF_REVISION:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--action_model_hf_revision="${ACTION_MODEL_HF_REVISION}")
fi
if [[ -n "${ACTION_MODEL_HF_REPO_TYPE:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--action_model_hf_repo_type="${ACTION_MODEL_HF_REPO_TYPE}")
fi
if [[ -n "${MIMIC_VIDEO_PROMPT:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--prompt="${MIMIC_VIDEO_PROMPT}")
fi
if [[ -n "${MIMIC_VIDEO_TASK:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--task="${MIMIC_VIDEO_TASK}")
fi

if [[ "${PREPULL}" != "0" ]]; then
  PREPULL_ENV_FILE="${PREPULL_ENV_FILE:-/tmp/mimic_video_assets.env}"
  export PREPULL_ENV_FILE CHECKPOINT_CACHE_DIR
  python scripts/prepull_mimic_video_assets.py --write-env="${PREPULL_ENV_FILE}"
  # shellcheck disable=SC1090
  source "${PREPULL_ENV_FILE}"
  if [[ -n "${MIMIC_VIDEO_PROMPT_EMBEDDING_PATH:-}" ]]; then
    export MIMIC_VIDEO_PROMPT_EMBEDDING_PATH
  fi
  EXTRA_SERVER_ARGS+=(--video_model_path="${VIDEO_MODEL_PATH}")
  EXTRA_SERVER_ARGS+=(--action_model_path="${ACTION_MODEL_PATH}")
  EXTRA_SERVER_ARGS+=(--stats_path="${STATS_PATH}")
fi

exec python -m lerobot_policy_mimic_video.grpc_policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --training_config="${TRAINING_CONFIG}" \
  --checkpoint_cache_dir="${CHECKPOINT_CACHE_DIR}" \
  "${EXTRA_SERVER_ARGS[@]}"
