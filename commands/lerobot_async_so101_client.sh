#!/usr/bin/env bash
set -euo pipefail

: "${ROBOT_PORT:?Set ROBOT_PORT, for example /dev/tty.usbmodemXXXX}"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_KEY="${CAMERA_KEY:-front}"
PROMPT="${MIMIC_VIDEO_PROMPT:-the robot arm pushes the white object towards the goal circle}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-3}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-latest_only}"
: "${VIDEO_MODEL_PATH:?Set VIDEO_MODEL_PATH}"
: "${ACTION_MODEL_PATH:?Set ACTION_MODEL_PATH}"
: "${TRAINING_CONFIG:?Set TRAINING_CONFIG}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
EXTRA_POLICY_ARGS=()

if [[ -n "${STATS_PATH:-}" ]]; then
  EXTRA_POLICY_ARGS+=(--policy.stats_path="${STATS_PATH}")
fi
if [[ -n "${ACTION_DATA_DIR:-}" ]]; then
  EXTRA_POLICY_ARGS+=(--policy.action_data_dir="${ACTION_DATA_DIR}")
fi
if [[ -n "${HF_STATS_REPO_ID:-}" ]]; then
  EXTRA_POLICY_ARGS+=(--policy.hf_stats_repo_id="${HF_STATS_REPO_ID}")
fi
if [[ -n "${HF_STATS_FILENAME:-}" ]]; then
  EXTRA_POLICY_ARGS+=(--policy.hf_stats_filename="${HF_STATS_FILENAME}")
fi
if [[ -n "${HF_STATS_REVISION:-}" ]]; then
  EXTRA_POLICY_ARGS+=(--policy.hf_stats_revision="${HF_STATS_REVISION}")
fi


exec python -m lerobot.async_inference.robot_client \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="{ ${CAMERA_KEY}: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: 640, height: 480, fps: 10}}" \
  --task="${PROMPT}" \
  --policy_type=mimic_video \
  --pretrained_name_or_path=. \
  --policy_device="${POLICY_DEVICE}" \
  --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
  --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}" \
  --aggregate_fn_name="${AGGREGATE_FN_NAME}" \
  --policy.video_model_path="${VIDEO_MODEL_PATH}" \
  --policy.action_model_path="${ACTION_MODEL_PATH}" \
  --policy.training_config="${TRAINING_CONFIG}" \
  --policy.n_action_steps="${ACTIONS_PER_CHUNK}" \
  "${EXTRA_POLICY_ARGS[@]}"
