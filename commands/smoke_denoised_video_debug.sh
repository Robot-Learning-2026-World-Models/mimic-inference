#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:${PORT}}"
START_POLICY_SERVER="${START_POLICY_SERVER:-1}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-600}"
DENOISED_VIDEO_SAVE_DIR="${DENOISED_VIDEO_SAVE_DIR:-/tmp/mimic-denoised-video-smoke}"
DENOISED_VIDEO_SAVE_EVERY_N="${DENOISED_VIDEO_SAVE_EVERY_N:-1}"
DENOISED_VIDEO_SAVE_FPS="${DENOISED_VIDEO_SAVE_FPS:-10}"
DENOISED_VIDEO_SAVE_PREFIX="${DENOISED_VIDEO_SAVE_PREFIX:-debug_smoke_$(date +%Y%m%d_%H%M%S)_$$}"
DENOISED_VIDEO_VERIFY_GLOB="${DENOISED_VIDEO_VERIFY_GLOB:-*_obs*_pred*.mp4}"
SMOKE_SERVER_LOG="${SMOKE_SERVER_LOG:-${DENOISED_VIDEO_SAVE_DIR}/${DENOISED_VIDEO_SAVE_PREFIX}_policy_server.log}"

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
export DENOISED_VIDEO_SAVE_DIR DENOISED_VIDEO_SAVE_EVERY_N DENOISED_VIDEO_SAVE_FPS DENOISED_VIDEO_SAVE_PREFIX
export HOST PORT

mkdir -p "${DENOISED_VIDEO_SAVE_DIR}"
verify_marker="${DENOISED_VIDEO_SAVE_DIR}/.${DENOISED_VIDEO_SAVE_PREFIX}_verify_marker"
touch "${verify_marker}"

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" >/dev/null 2>&1; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_for_port() {
  local address="$1"
  local timeout_s="$2"
  local host="${address%:*}"
  local port="${address##*:}"
  local deadline=$((SECONDS + timeout_s))

  while (( SECONDS < deadline )); do
    if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "Policy server exited before opening ${address}. Log: ${SMOKE_SERVER_LOG}" >&2
      tail -n 80 "${SMOKE_SERVER_LOG}" >&2 || true
      return 1
    fi
    if (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "Timed out waiting for policy server at ${address}. Log: ${SMOKE_SERVER_LOG}" >&2
  tail -n 80 "${SMOKE_SERVER_LOG}" >&2 || true
  return 1
}

if [[ "${START_POLICY_SERVER}" != "0" ]]; then
  : "${TRAINING_CONFIG:?Set TRAINING_CONFIG, or set START_POLICY_SERVER=0 to test an already-running debug-enabled server.}"
  echo "Starting debug policy server on ${HOST}:${PORT}"
  echo "Saving denoised videos to ${DENOISED_VIDEO_SAVE_DIR} with prefix ${DENOISED_VIDEO_SAVE_PREFIX}"
  commands/lerobot_async_policy_server.sh >"${SMOKE_SERVER_LOG}" 2>&1 &
  server_pid="$!"
  wait_for_port "${SERVER_ADDRESS}" "${SERVER_START_TIMEOUT}"
else
  echo "Using existing policy server at ${SERVER_ADDRESS}"
  echo "Expecting denoised videos in ${DENOISED_VIDEO_SAVE_DIR} with prefix ${DENOISED_VIDEO_SAVE_PREFIX}"
fi

if ! "${PYTHON_CMD[@]}" scripts/smoke_mimic_video_grpc_client.py \
  --server-address="${SERVER_ADDRESS}" \
  --timeout="${SMOKE_TIMEOUT}" \
  "$@"; then
  if [[ -n "${server_pid}" ]]; then
    echo "Smoke client failed. Policy server log: ${SMOKE_SERVER_LOG}" >&2
    tail -n 80 "${SMOKE_SERVER_LOG}" >&2 || true
  fi
  exit 1
fi

mapfile -t saved_videos < <(
  find "${DENOISED_VIDEO_SAVE_DIR}" \
    -maxdepth 1 \
    -type f \
    -newer "${verify_marker}" \
    -name "${DENOISED_VIDEO_VERIFY_GLOB}" \
    -print \
    | sort
)

if (( ${#saved_videos[@]} == 0 )); then
  echo "Smoke client received actions, but no denoised debug video was saved." >&2
  echo "Checked for new files matching: ${DENOISED_VIDEO_SAVE_DIR}/${DENOISED_VIDEO_VERIFY_GLOB}" >&2
  if [[ -n "${server_pid}" ]]; then
    echo "Policy server log: ${SMOKE_SERVER_LOG}" >&2
    tail -n 80 "${SMOKE_SERVER_LOG}" >&2 || true
  fi
  exit 1
fi

echo "Saved denoised debug video(s):"
printf '  %s\n' "${saved_videos[@]}"
if [[ -n "${server_pid}" ]]; then
  echo "Policy server log: ${SMOKE_SERVER_LOG}"
fi
