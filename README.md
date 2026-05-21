# SO101 mimic-video inference

Uv-only setup for running mimic-video inference on a Blackwell GPU. Do not use
sudo or Docker.

## Setup

```bash
git clone --recurse-submodules git@github.com:Robot-Learning-2026-World-Models/mimic-inference.git
cd mimic-inference
git submodule update --init --recursive

cd mimic-video/model
uv sync
cd ../..
```

`uv sync` creates `mimic-video/model/.venv` with Python 3.12, Torch CUDA 12.8,
flash-attn, NATTEN, Transformer Engine, LeRobot, and the local policy plugin.

## Pull weights and tokenizer

Set `HF_TOKEN` if the Hugging Face repo is private. This skips T5 and uses the
repo-local prompt embedding.

```bash
export HF_HOME="$PWD/hf-cache"
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300

export TRAINING_CONFIG="$PWD/mimic-video/configs/so101/idm_train_dual_task_low_rank.yaml"
export CHECKPOINT_CACHE_DIR="$PWD/checkpoints"
export MIMIC_VIDEO_HF_REPO_ID="mimicinferencegobrrrr/mimic-finetune"
export VIDEO_MODEL_HF_FILENAME="video_backbone/iter_000002800_fused.pt"
export ACTION_MODEL_HF_FILENAME="action_decoder/iter_000001850.pt"
export HF_STATS_FILENAME="dataset_statistics/so101_task1_2_stats.json"
export HF_STATS_REPO_TYPE="model"
export PREPULL_TEXT_ENCODER=0
export PYTHONPATH="$PWD/lerobot_policy_mimic_video/src:$PWD/mimic-video/model"

mimic-video/model/.venv/bin/python scripts/prepull_mimic_video_assets.py \
  --write-env="$PWD/checkpoints/mimic_video_assets.env"

mimic-video/model/.venv/bin/hf download "$MIMIC_VIDEO_HF_REPO_ID" \
  video_backbone/tokenizer/tokenizer.pth \
  --repo-type model \
  --local-dir "$PWD/checkpoints"
```

## Run server

Terminal A:

```bash
export PATH="$PWD/mimic-video/model/.venv/bin:$PATH"
export PYTHONPATH="$PWD/lerobot_policy_mimic_video/src:$PWD/mimic-video/model"
export COSMOS_PREDICT2_ARGS="--checkpoints $PWD/checkpoints"

export TRAINING_CONFIG="$PWD/mimic-video/configs/so101/idm_train_dual_task_low_rank.yaml"
export VIDEO_MODEL_PATH="$PWD/checkpoints/iter_000002800_fused.pt"
export ACTION_MODEL_PATH="$PWD/checkpoints/iter_000001850.pt"
export STATS_PATH="$PWD/checkpoints/dataset_statistics/so101_task1_2_stats.json"
export MIMIC_VIDEO_PROMPT_EMBEDDING_PATH="$PWD/inference_t5_embeddings/the_robot_arm_gently_nudges_the_white_object_towards_the_target_circle.pickle"
export MIMIC_VIDEO_TASK=task1
export PREPULL=0
export HOST=127.0.0.1
export PORT=8080
export FPS=30

commands/lerobot_async_policy_server.sh
```

### Long-context IDM from `/Volumes/Transfer`

The `idm_long_700*` artifacts use `experiment: so101_90`, so deployment derives
`state_t=16`, 5 visual observation frames, 56 future video frames, a 90-action
decoder horizon, `max_horizon=91`, and `xattn_layer_idx=16` from the exported
YAML.

```bash
export PATH="$PWD/mimic-video/model/.venv/bin:$PATH"
export PYTHONPATH="$PWD/lerobot_policy_mimic_video/src:$PWD/mimic-video/model"
export COSMOS_PREDICT2_ARGS="--checkpoints $PWD/checkpoints"

export TRAINING_CONFIG="/Volumes/Transfer/idm_long_700_config.yaml"
export VIDEO_MODEL_PATH="/Volumes/Transfer/idm_long_700_backbone.pt"
export ACTION_MODEL_PATH="/Volumes/Transfer/idm_long_700.pt"
export STATS_PATH="/Volumes/Transfer/idm_long_700_stats.json"
export MIMIC_VIDEO_PROMPT_EMBEDDING_PATH="$PWD/inference_t5_embeddings/the_robot_arm_gently_nudges_the_white_object_towards_the_target_circle.pickle"
export MIMIC_VIDEO_TASK=task1
export PREPULL=0
export HOST=127.0.0.1
export PORT=8080
export FPS=30

commands/lerobot_async_policy_server.sh
```

## Smoke client

Terminal B, no robot required:

```bash
export PATH="$PWD/mimic-video/model/.venv/bin:$PATH"
export PYTHONPATH="$PWD/lerobot_policy_mimic_video/src:$PWD/mimic-video/model"
export COSMOS_PREDICT2_ARGS="--checkpoints $PWD/checkpoints"

python scripts/smoke_mimic_video_grpc_client.py \
  --server-address 127.0.0.1:8080 \
  --raw-image-horizon 13 \
  --actions-per-chunk 3 \
  --timeout 600
```

## Robot client

Terminal B, when ready to run the SO101:

```bash
export PATH="$PWD/mimic-video/model/.venv/bin:$PATH"
export PYTHONPATH="$PWD/lerobot_policy_mimic_video/src:$PWD/mimic-video/model"
export ROBOT_PORT=/dev/ttyACM0
export SERVER_ADDRESS=127.0.0.1:8080

commands/lerobot_async_so101_client.sh
```
