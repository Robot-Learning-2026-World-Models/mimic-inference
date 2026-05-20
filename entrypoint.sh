#!/usr/bin/env bash
set -euo pipefail

cd /workspace/so101-mimic-video-deploy
source mimic-video/model/.venv/bin/activate
export PYTHONPATH="/workspace/so101-mimic-video-deploy/mimic-video/model:${PYTHONPATH:-}"

mode="${1:-policy-server}"
shift || true
case "${mode}" in
  prepull)
    exec python scripts/prepull_mimic_video_assets.py "$@"
    ;;
  policy-server)
    exec commands/lerobot_async_policy_server.sh "$@"
    ;;
  debug-smoke|smoke-debug)
    exec commands/smoke_denoised_video_debug.sh "$@"
    ;;
  shell)
    exec bash "$@"
    ;;
  *)
    exec "${mode}" "$@"
    ;;
esac
