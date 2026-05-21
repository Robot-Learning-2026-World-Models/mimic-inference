#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PATH="${REPO_ROOT}/mimic-video/model/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/lerobot_policy_mimic_video/src:${REPO_ROOT}/mimic-video/model"
export COSMOS_PREDICT2_ARGS="--checkpoints ${REPO_ROOT}/checkpoints"

export TRAINING_CONFIG="${TRAINING_CONFIG:-${REPO_ROOT}/mimic-video/configs/so101/idm_large_all_tasks_dirty_4500.yaml}"
export VIDEO_MODEL_PATH="${VIDEO_MODEL_PATH:-${REPO_ROOT}/checkpoints/rev3/iter_000004500_fused.pt}"
export ACTION_MODEL_PATH="${ACTION_MODEL_PATH:-${REPO_ROOT}/checkpoints/rev3/iter_000001200.pt}"
export STATS_PATH="${STATS_PATH:-${REPO_ROOT}/checkpoints/rev3/dataset_stats.json}"

export MIMIC_VIDEO_TASK="${MIMIC_VIDEO_TASK:-task3}"
if [[ -z "${MIMIC_VIDEO_PROMPT_EMBEDDING_PATH:-}" ]]; then
  case "${MIMIC_VIDEO_TASK,,}" in
    task1|task-1|task_1)
      MIMIC_VIDEO_PROMPT_EMBEDDING_PATH="${REPO_ROOT}/inference_t5_embeddings/the_robot_arm_gently_nudges_the_white_object_towards_the_target_circle.pickle"
      ;;
    task2|task-2|task_2)
      MIMIC_VIDEO_PROMPT_EMBEDDING_PATH="${REPO_ROOT}/inference_t5_embeddings/the_robot_arm_gently_nudges_the_white_object_towards_the_target_circle_while_avoiding_the_red_cylinder.pickle"
      ;;
    task3|task-3|task_3)
      MIMIC_VIDEO_PROMPT_EMBEDDING_PATH="${REPO_ROOT}/inference_t5_embeddings/the_robot_arm_gently_nudges_the_colored_cube_towards_the_target_circle.pickle"
      ;;
    task4|task-4|task_4)
      MIMIC_VIDEO_PROMPT_EMBEDDING_PATH="${REPO_ROOT}/inference_t5_embeddings/the_robot_arm_gently_nudges_the_colored_cube_towards_the_target_circle_while_avoiding_the_red_cylinder.pickle"
      ;;
    *)
      MIMIC_VIDEO_PROMPT_EMBEDDING_PATH="${REPO_ROOT}/inference_t5_embeddings/the_robot_arm_gently_nudges_the_white_object_towards_the_target_circle.pickle"
      ;;
  esac
fi
export MIMIC_VIDEO_PROMPT_EMBEDDING_PATH
export MIMIC_VIDEO_DEBUG_SAVE_DENOISED_VIDEO="${MIMIC_VIDEO_DEBUG_SAVE_DENOISED_VIDEO:-1}"
export MIMIC_VIDEO_DEBUG_PREVIEW_DENOISED_VIDEO="${MIMIC_VIDEO_DEBUG_PREVIEW_DENOISED_VIDEO:-1}"
export MIMIC_VIDEO_DEBUG_OUTPUT_DIR="${MIMIC_VIDEO_DEBUG_OUTPUT_DIR:-${REPO_ROOT}/outputs/debug_plans}"

export PREPULL="${PREPULL:-0}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8080}"
export FPS="${FPS:-30}"

exec commands/lerobot_async_policy_server.sh
