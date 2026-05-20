#!/usr/bin/env bash
set -euo pipefail

: "${ROBOT_PORT:?Set ROBOT_PORT, for example /dev/tty.usbmodemXXXX}"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_KEY="${CAMERA_KEY:-front}"
CAMERA_WIDTH="${CAMERA_WIDTH:-1280}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-720}"
CAMERA_FPS="${CAMERA_FPS:-30}"
RAW_IMAGE_HORIZON="${RAW_IMAGE_HORIZON:-13}"
PROMPT="${MIMIC_VIDEO_PROMPT:-the robot arm pushes the white object towards the goal circle}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-3}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-1.0}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-latest_only}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

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

export PYTHONPATH="${PWD}/lerobot_policy_mimic_video/src:${PYTHONPATH:-}"

exec "${PYTHON_CMD[@]}" -m lerobot_policy_mimic_video.grpc_robot_client \
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
  --raw_image_horizon="${RAW_IMAGE_HORIZON}"
