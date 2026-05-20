#!/usr/bin/env bash
set -euo pipefail

cd /workspace/mimic-inference
source mimic-video/model/.venv/bin/activate
export PYTHONPATH="/workspace/mimic-inference/mimic-video/model:${PYTHONPATH:-}"

mode="${1:-policy-server}"
shift || true
case "${mode}" in
  prepull)
    exec python scripts/prepull_mimic_video_assets.py "$@"
    ;;
  policy-server)
    exec commands/lerobot_async_policy_server.sh "$@"
    ;;
  shell)
    exec bash "$@"
    ;;
  *)
    exec "${mode}" "$@"
    ;;
esac
