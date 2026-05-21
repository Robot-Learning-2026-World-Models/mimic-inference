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
commands/server.sh
```

## Smoke client

Terminal B, no robot required:

```bash
commands/smoke_client.sh
```

## Sync debug robot client

Terminal B, when ready to run the SO101. This is the recommended debug path:
it warms a fresh 13-frame camera window at 30 Hz, sends one observation, waits
while the server fully denoises and previews the 41-frame plan, then executes
the returned 60-action chunk at 30 Hz.

```bash
commands/client.sh
```

Debug artifacts are written per inference under `MIMIC_VIDEO_DEBUG_OUTPUT_DIR`
on the server. The sync client also writes returned/applied action chunks under
`SYNC_ACTION_LOG_DIR` (default `outputs/sync_client_actions`).

## Async robot client

The async launcher is still available for later smoother real-time experiments,
but it is not the default debug path because the server-side preview pauses
inference.

```bash
commands/lerobot_async_so101_client.sh
```
