#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
FPS="${FPS:-30}"
: "${TRAINING_CONFIG:?Set TRAINING_CONFIG}"
if [[ -z "${CHECKPOINT_CACHE_DIR:-}" ]]; then
  if [[ -d /checkpoints && -w /checkpoints ]]; then
    CHECKPOINT_CACHE_DIR=/checkpoints
  else
    CHECKPOINT_CACHE_DIR="${PWD}/checkpoints"
  fi
fi
MIMIC_VIDEO_HF_REPO_ID="${MIMIC_VIDEO_HF_REPO_ID:-rl26-world-models/mimic-finetune}"
PREPULL="${PREPULL:-1}"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_CMD=("${PYTHON}")
elif command -v uv >/dev/null 2>&1 && [[ -f "${PWD}/mimic-video/model/pyproject.toml" ]]; then
  UV_PROJECT_DIR="${UV_PROJECT_DIR:-${PWD}/mimic-video/model}"
  UV_EXTRA="${UV_EXTRA:-cu126}"
  UV_PYTHON="${UV_PROJECT_DIR}/.venv/bin/python"
  if [[ ! -x "${UV_PYTHON}" || "${UV_SYNC:-0}" == "1" ]]; then
    UV_SYNC_CMD=(uv sync --project "${UV_PROJECT_DIR}")
    if [[ -n "${UV_EXTRA}" ]]; then
      UV_SYNC_CMD+=(--extra "${UV_EXTRA}")
    fi
    "${UV_SYNC_CMD[@]}"
  fi
  if [[ "${UV_WITH_LEROBOT:-1}" == "1" ]] && ! "${UV_PYTHON}" -c 'import lerobot' >/dev/null 2>&1; then
    UV_LEROBOT_INSTALL_CMD=(
      uv pip install
      --python "${UV_PYTHON}"
      --torch-backend cu126
      "lerobot[async]"
      "torch==2.6.0"
      "torchvision==0.21.0"
    )
    "${UV_LEROBOT_INSTALL_CMD[@]}"
  fi
  PYTHON_CMD=("${UV_PYTHON}")
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
else
  PYTHON_CMD=(python3)
fi

export PYTHONPATH="${PWD}/lerobot_policy_mimic_video/src:${PWD}/mimic-video/model:${PYTHONPATH:-}"
export MIMIC_VIDEO_HF_REPO_ID

mkdir -p "${CHECKPOINT_CACHE_DIR}"
COSMOS_CHECKPOINT_DIR="${PWD}/mimic-video/model/checkpoints"
mkdir -p "${COSMOS_CHECKPOINT_DIR}"
for checkpoint_name in video_backbone text_encoder; do
  cache_checkpoint="${CHECKPOINT_CACHE_DIR}/${checkpoint_name}"
  if [[ "${checkpoint_name}" == "text_encoder" && -e "${CHECKPOINT_CACHE_DIR}/text_encoder/text_encoder" ]]; then
    cache_checkpoint="${CHECKPOINT_CACHE_DIR}/text_encoder/text_encoder"
  fi
  cosmos_checkpoint="${COSMOS_CHECKPOINT_DIR}/${checkpoint_name}"
  if [[ -L "${cosmos_checkpoint}" && "$(readlink "${cosmos_checkpoint}")" != "${cache_checkpoint}" ]]; then
    rm "${cosmos_checkpoint}"
  fi
  if [[ -e "${cache_checkpoint}" && ! -e "${cosmos_checkpoint}" ]]; then
    ln -s "${cache_checkpoint}" "${cosmos_checkpoint}"
  fi
done

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
if [[ -n "${DENOISED_VIDEO_SAVE_DIR:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--denoised_video_save_dir="${DENOISED_VIDEO_SAVE_DIR}")
fi
if [[ -n "${DENOISED_VIDEO_SAVE_EVERY_N:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--denoised_video_save_every_n="${DENOISED_VIDEO_SAVE_EVERY_N}")
fi
if [[ -n "${DENOISED_VIDEO_SAVE_FPS:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--denoised_video_save_fps="${DENOISED_VIDEO_SAVE_FPS}")
fi
if [[ -n "${DENOISED_VIDEO_SAVE_PREFIX:-}" ]]; then
  EXTRA_SERVER_ARGS+=(--denoised_video_save_prefix="${DENOISED_VIDEO_SAVE_PREFIX}")
fi

if [[ "${PREPULL}" != "0" ]]; then
  PREPULL_ENV_FILE="${PREPULL_ENV_FILE:-/tmp/mimic_video_assets.env}"
  export PREPULL_ENV_FILE CHECKPOINT_CACHE_DIR
  "${PYTHON_CMD[@]}" scripts/prepull_mimic_video_assets.py --write-env="${PREPULL_ENV_FILE}"
  # shellcheck disable=SC1090
  source "${PREPULL_ENV_FILE}"
  EXTRA_SERVER_ARGS+=(--video_model_path="${VIDEO_MODEL_PATH}")
  EXTRA_SERVER_ARGS+=(--action_model_path="${ACTION_MODEL_PATH}")
  EXTRA_SERVER_ARGS+=(--stats_path="${STATS_PATH}")
fi

exec "${PYTHON_CMD[@]}" -m lerobot_policy_mimic_video.grpc_policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --training_config="${TRAINING_CONFIG}" \
  --checkpoint_cache_dir="${CHECKPOINT_CACHE_DIR}" \
  "${EXTRA_SERVER_ARGS[@]}"
