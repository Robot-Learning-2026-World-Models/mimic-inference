#!/usr/bin/env bash

export PATH="$PWD/mimic-video/model/.venv/bin:$PATH"
export PYTHONPATH="$PWD/lerobot_policy_mimic_video/src:$PWD/mimic-video/model"
export ROBOT_PORT=/dev/ttyACM0
export SERVER_ADDRESS=127.0.0.1:8080
export PROMPT="${PROMPT:-task2}"

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

commands/lerobot_async_so101_client.sh
