REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PATH="$PWD/mimic-video/model/.venv/bin:$PATH"
export PYTHONPATH="$PWD/lerobot_policy_mimic_video/src:$PWD/mimic-video/model"
export COSMOS_PREDICT2_ARGS="--checkpoints $PWD/checkpoints"

export TRAINING_CONFIG="$PWD/mimic-video/configs/so101/idm_large_all_tasks.yaml"
export VIDEO_MODEL_PATH="$PWD/checkpoints/iter_000003600_fused.pt"
export ACTION_MODEL_PATH="$PWD/checkpoints/iter_000001450.pt"
export STATS_PATH="$PWD/checkpoints/iter_1450_dataset_stats.json"
export MIMIC_VIDEO_PROMPT_EMBEDDING_PATH="$PWD/inference_t5_embeddings/the_robot_arm_gently_nudges_the_white_object_towards_the_target_circle.pickle"
export PREPULL=0
export HOST=127.0.0.1
export PORT=8080
export FPS=30


OUTPUT_DEBUG=TRUE
if [[ "${OUTPUT_DEBUG}" == "TRUE" ]]; then
  rm -rf "$REPO_ROOT/outputs"
  export MIMIC_VIDEO_DEBUG_SAVE_DENOISED_VIDEO=1
  export MIMIC_VIDEO_DEBUG_OUTPUT_DIR="$PWD/outputs"
else
  export MIMIC_VIDEO_DEBUG_SAVE_DENOISED_VIDEO=0
fi

FILTER_RECEIVER_LOG=TRUE
if [[ "${FILTER_RECEIVER_LOG}" == "TRUE" ]]; then
  commands/lerobot_async_policy_server.sh 2>&1 | grep --line-buffered -v "Starting receiver"
else
  commands/lerobot_async_policy_server.sh
fi
