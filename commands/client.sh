#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PATH="${REPO_ROOT}/mimic-video/model/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/lerobot_policy_mimic_video/src:${REPO_ROOT}/mimic-video/model"

export ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
export SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
export ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-60}"
export RAW_IMAGE_HORIZON="${RAW_IMAGE_HORIZON:-13}"

exec commands/lerobot_sync_so101_client.sh
