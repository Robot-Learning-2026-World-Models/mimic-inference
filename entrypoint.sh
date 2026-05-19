#!/usr/bin/env bash
set -euo pipefail

cd /workspace/so101-mimic-video-deploy
source mimic-video/model/.venv/bin/activate
export PYTHONPATH="/workspace/so101-mimic-video-deploy/mimic-video/model:${PYTHONPATH:-}"

mode="${1:-policy-server}"
shift || true
case "${mode}" in
  policy-server)
    exec commands/lerobot_async_policy_server.sh "$@"
    ;;
  video-server)
    exec python scripts/so101_inference_server.py "$@"
    ;;
  shell)
    exec bash "$@"
    ;;
  *)
    exec "${mode}" "$@"
    ;;
esac
