#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PATH="$PWD/mimic-video/model/.venv/bin:$PATH"
export PYTHONPATH="$PWD/lerobot_policy_mimic_video/src:$PWD/mimic-video/model"
export COSMOS_PREDICT2_ARGS="--checkpoints $PWD/checkpoints"

export TRAINING_CONFIG="/media/baraq/Transfer/idm_long_700_config.yaml"
export VIDEO_MODEL_PATH="/media/baraq/Transfer/idm_long_700_backbone.pt"
export ACTION_MODEL_PATH="/media/baraq/Transfer/idm_long_700.pt"
export STATS_PATH="/media/baraq/Transfer/idm_long_700_stats.json"
export PROMPT="${PROMPT:-task2}"
export PREPULL=0
export HOST=127.0.0.1
export PORT=8080
export FPS=30

PROMPT_KEY="${PROMPT,,}"
PROMPT_KEY="${PROMPT_KEY// /}"
PROMPT_KEY="${PROMPT_KEY//-/}"
PROMPT_KEY="${PROMPT_KEY//_/}"
case "${PROMPT_KEY}" in
  1|task1)
    export MIMIC_VIDEO_TASK=task1
    export MIMIC_VIDEO_PROMPT="the robot arm gently nudges the white object towards the target circle"
    export MIMIC_VIDEO_PROMPT_EMBEDDING_PATH="$PWD/inference_t5_embeddings/the_robot_arm_gently_nudges_the_white_object_towards_the_target_circle.pickle"
    ;;
  2|task2)
    export MIMIC_VIDEO_TASK=task2
    export MIMIC_VIDEO_PROMPT="the robot arm gently nudges the white object towards the target circle while avoiding the red cylinder"
    export MIMIC_VIDEO_PROMPT_EMBEDDING_PATH="$PWD/inference_t5_embeddings/the_robot_arm_gently_nudges_the_white_object_towards_the_target_circle_while_avoiding_the_red_cylinder.pickle"
    ;;
  3|task3)
    export MIMIC_VIDEO_TASK=task3
    export MIMIC_VIDEO_PROMPT="the robot arm gently nudges the colored cube towards the target circle"
    export MIMIC_VIDEO_PROMPT_EMBEDDING_PATH="$PWD/inference_t5_embeddings/the_robot_arm_gently_nudges_the_colored_cube_towards_the_target_circle.pickle"
    ;;
  4|task4)
    export MIMIC_VIDEO_TASK=task4
    export MIMIC_VIDEO_PROMPT="the robot arm gently nudges the colored cube towards the target circle while avoiding the red cylinder"
    export MIMIC_VIDEO_PROMPT_EMBEDDING_PATH="$PWD/inference_t5_embeddings/the_robot_arm_gently_nudges_the_colored_cube_towards_the_target_circle_while_avoiding_the_red_cylinder.pickle"
    ;;
  *)
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! WARNING: PROMPT DID NOT RESOLVE TO TASK1/TASK2/TASK3/TASK4 !!!"
    echo "!!! USING RAW PROMPT: ${PROMPT}"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    export MIMIC_VIDEO_PROMPT="${PROMPT}"
    ;;
esac

OUTPUT_DEBUG=FALSE
if [[ "${OUTPUT_DEBUG}" == "TRUE" ]]; then
  # rm -rf "$REPO_ROOT/outputs"
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
