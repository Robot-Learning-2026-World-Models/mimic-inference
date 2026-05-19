#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
FPS="${FPS:-10}"

export PYTHONPATH="${PWD}/mimic-video/model:${PYTHONPATH:-}"

exec python -m lerobot.async_inference.policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}"
