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
PROMPT="${MIMIC_VIDEO_PROMPT:-the_robot_arm_gently_nudges_the_white_object_towards_the_target_circle_while_avoiding_the_red_cylinder}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-60}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.8}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-latest_only}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

export PYTHONPATH="${PWD}/lerobot_policy_mimic_video/src:${PYTHONPATH:-}"

exec python -m lerobot_policy_mimic_video.grpc_robot_client \
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
  --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}" \
  --aggregate_fn_name="${AGGREGATE_FN_NAME}" \
  --raw_image_horizon="${RAW_IMAGE_HORIZON}" \
  --aggregate_fn="conservative"
