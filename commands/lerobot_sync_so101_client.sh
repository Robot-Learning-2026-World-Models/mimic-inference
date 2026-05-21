#!/usr/bin/env bash
set -euo pipefail

: "${ROBOT_PORT:?Set ROBOT_PORT, for example /dev/tty.usbmodemXXXX}"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
ROBOT_ID="${ROBOT_ID:-follower_arm}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_KEY="${CAMERA_KEY:-front}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
RAW_IMAGE_HORIZON="${RAW_IMAGE_HORIZON:-13}"
PROMPT="${MIMIC_VIDEO_PROMPT:-the robot arm pushes the white object towards the goal circle}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-60}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
SYNC_NUM_CYCLES="${SYNC_NUM_CYCLES:-0}"
SYNC_LOG_ACTIONS="${SYNC_LOG_ACTIONS:-true}"
SYNC_ACTION_LOG_DIR="${SYNC_ACTION_LOG_DIR:-outputs/sync_client_actions}"

export PYTHONPATH="${PWD}/lerobot_policy_mimic_video/src:${PYTHONPATH:-}"

exec python -m lerobot_policy_mimic_video.grpc_robot_client \
  --client_mode=sync \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="{ ${CAMERA_KEY}: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}}" \
  --task="${PROMPT}" \
  --policy_type=mimic_video \
  --pretrained_name_or_path=. \
  --policy_device="${POLICY_DEVICE}" \
  --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
  --raw_image_horizon="${RAW_IMAGE_HORIZON}" \
  --sync_num_cycles="${SYNC_NUM_CYCLES}" \
  --sync_log_actions="${SYNC_LOG_ACTIONS}" \
  --sync_action_log_dir="${SYNC_ACTION_LOG_DIR}" \
  --aggregate_fn="conservative"
