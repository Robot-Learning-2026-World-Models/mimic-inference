# SO101 mimic-video Deployment

LeRobot async deployment wrapper for `mimic-video` on SO101.

This repo keeps deployment separate from the training/research repo and vendors `mimic-video` as a git submodule. The submodule should track `video_finetune_improvements`.

## Container

Build the robot action deployment image:

```bash
docker build -t so101-mimic-video-deploy .
```

For RTX 5090 / CUDA 12.8 hosts, build the Blackwell image instead:

```bash
docker build -f Dockerfile.blackwell -t so101-mimic-video-deploy:blackwell .
```

The Blackwell image defaults to RTX 5090 (`sm_120`) to keep CUDA extension
builds small. For B200/GB200-style Blackwell, override both arch args:

```bash
docker build -f Dockerfile.blackwell \
  --build-arg CUDA_ARCH_LIST="10.0" \
  --build-arg NVTE_CUDA_ARCHS="100" \
  -t so101-mimic-video-deploy:blackwell .
```

Pull the dataset stats (NOT NEEDED UNLESS WE NEED TO FUCKING RECOMPUTE THE MOTHERFUCKINGS STATS):
```
docker run --rm --gpus all --ipc=host --shm-size=32g \
    -v $PWD/hf-cache:/root/.cache/huggingface \
    -v $PWD/data:/workspace/so101-mimic-video-deploy/data \
    -e HF_HOME=/root/.cache/huggingface \
    so101-mimic-video-deploy \
    python scripts/prepare_so101_inference_stats.py \
      --config mimic-video/configs/so101/idm_train_dual_task_low_rank.yaml
```

Pull weights (DO THIS IN A FUCKING REPO WITH SPACE):
```
docker run --rm --gpus all --ipc=host --shm-size=32g \
    -v $PWD/scripts/prepull_mimic_video_assets.py:/workspace/so101-mimic-video-deploy/scripts/prepull_mimic_video_assets.py:ro \
    -v $PWD/lerobot_policy_mimic_video/src/lerobot_policy_mimic_video/training_config.py:/workspace/so101-mimic-video-deploy/lerobot_policy_mimic_video/src/lerobot_policy_mimic_video/training_config.py:ro \
    -v $PWD/checkpoints:/checkpoints \
    -v $PWD/hf-cache:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN \
    -e HF_HUB_ETAG_TIMEOUT=60 \
    -e HF_HUB_DOWNLOAD_TIMEOUT=300 \
    -e TRAINING_CONFIG=/workspace/so101-mimic-video-deploy/mimic-video/configs/so101/idm_train_dual_task_low_rank.yaml \
    -e MIMIC_VIDEO_HF_REPO_ID=mimicinferencegobrrrr/mimic-finetune \
    -e VIDEO_MODEL_HF_FILENAME=video_backbone/iter_000002800_fused.pt \
    -e ACTION_MODEL_HF_FILENAME=action_decoder/iter_000001850.pt \
    -e HF_STATS_FILENAME=dataset_statistics/so101_task1_2_stats.json \
    -e HF_STATS_REPO_TYPE=model \
    so101-mimic-video-deploy \
    python scripts/prepull_mimic_video_assets.py
```

Prepull skips T5 by default. Only opt into the text encoder if you need to generate prompt embeddings on the host:
```
PREPULL_TEXT_ENCODER=1
```

Generate prompt embeddings once, then use those at server startup instead of loading T5:
```
docker run --rm --runtime=nvidia --ipc=host --shm-size=64g \
    -v $PWD/checkpoints:/checkpoints \
    -v $PWD/lerobot_policy_mimic_video/src/lerobot_policy_mimic_video:/workspace/so101-mimic-video-deploy/lerobot_policy_mimic_video/src/lerobot_policy_mimic_video:ro \
    -v $PWD/scripts/cache_mimic_video_prompt_embedding.py:/workspace/so101-mimic-video-deploy/scripts/cache_mimic_video_prompt_embedding.py:ro \
    so101-mimic-video-deploy \
    python scripts/cache_mimic_video_prompt_embedding.py \
      --tasks task1 task2 \
      --prompts "alternate language prompt text here" \
      --text-encoder-path /checkpoints/text_encoder/text_encoder/t5-11b \
      --output-dir /checkpoints/prompt_embeddings
```

Pull tokenizer since I fucked up:
```
docker run --rm \
    -v $PWD/checkpoints:/checkpoints \
    -v $PWD/hf-cache:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN \
    so101-mimic-video-deploy \
    hf download mimicinferencegobrrrr/mimic-finetune \
      video_backbone/tokenizer/tokenizer.pth \
      --repo-type model \
      --local-dir /checkpoints
```

Start the server:
```
docker run --rm --runtime=nvidia --ipc=host --shm-size=64g \
    -p 8080:8080 \
    -v $PWD/lerobot_policy_mimic_video/src/lerobot_policy_mimic_video:/workspace/so101-mimic-video-deploy/lerobot_policy_mimic_video/src/lerobot_policy_mimic_video:ro \
    -v $PWD/mimic-video/configs/so101:/workspace/so101-mimic-video-deploy/mimic-video/configs/so101:ro \
    -v $PWD/checkpoints:/checkpoints \
    -v $PWD/checkpoints/video_backbone:/workspace/so101-mimic-video-deploy/mimic-video/model/checkpoints/video_backbone:ro \
    -v $PWD/checkpoints/text_encoder/text_encoder:/workspace/so101-mimic-video-deploy/mimic-video/model/checkpoints/text_encoder:ro \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e TRAINING_CONFIG=/workspace/so101-mimic-video-deploy/mimic-video/configs/so101/idm_train_dual_task_low_rank.yaml \
    -e VIDEO_MODEL_PATH=/checkpoints/iter_000002800_fused.pt \
    -e ACTION_MODEL_PATH=/checkpoints/iter_000001900.pt \
    -e STATS_PATH=/checkpoints/dataset_statistics/so101_task1_2_stats.json \
    -e MIMIC_VIDEO_PROMPT_EMBEDDING_PATH=/checkpoints/prompt_embeddings/task1.pt \
    -e MIMIC_VIDEO_TASK=task1 \
    -e PREPULL=0 \
    -e FPS=30 \
    so101-mimic-video-deploy \
    policy-server
```

Smoke test client:
```
docker run --rm --network host     -v $PWD/scripts/smoke_mimic_video_grpc_client.py:/workspace/so101-mimic-video-deploy/scripts/smoke_mimic_video_grpc_client.py:ro     -v $PWD/lerobot_policy_mimic_video/src/lerobot_policy_mimic_video:/workspace/so101-mimic-video-deploy/lerobot_policy_mimic_video/src/lerobot_policy_mimic_video:ro     so101-mimic-video-deploy     python scripts/smoke_mimic_video_grpc_client.py       --server-address 127.0.0.1:8080       --raw-image-horizon 13       --actions-per-chunk 3       --timeout 600
```

Run the LeRobot async policy server on the GPU machine:

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g -p 8080:8080 \
  -v /host/checkpoints:/checkpoints:ro \
  -v /host/data:/data:ro \
  so101-mimic-video-deploy policy-server
```

## Run LeRobot Async

Terminal A on the GPU machine:

```bash
commands/lerobot_async_policy_server.sh
```

Terminal B on the robot machine:

```bash
ROBOT_PORT=/dev/tty.usbmodemXXXX \
commands/lerobot_async_so101_client.sh
```

Pass `STATS_PATH=/abs/path/to/statistics.json` instead of `ACTION_DATA_DIR` if you already know the exact statistics file. If the file is on Hugging Face, set `STATS_PATH` to the local destination plus `HF_STATS_REPO_ID`, `HF_STATS_FILENAME`, and optionally `HF_STATS_REVISION`; the plugin downloads it before loading the normalizer.

Checkpoints are resolved by the policy server in this order:

1. `VIDEO_MODEL_PATH` / `ACTION_MODEL_PATH`
2. `training.action_decoder.model.video_dit_path` / `action_dit_path` in `TRAINING_CONFIG`
3. Hugging Face download envs. By default, `MIMIC_VIDEO_HF_REPO_ID=rl26-world-models/mimic-finetune` is used as the shared repo for video/action/text assets:

```bash
VIDEO_MODEL_HF_REPO_ID=...
VIDEO_MODEL_HF_FILENAME=...
VIDEO_MODEL_HF_REVISION=...
ACTION_MODEL_HF_REPO_ID=...
ACTION_MODEL_HF_FILENAME=...
ACTION_MODEL_HF_REVISION=...
TEXT_ENCODER_HF_ALLOW_PATTERN='text_encoder/**'
CHECKPOINT_CACHE_DIR=/checkpoints
```

The container runs `scripts/prepull_mimic_video_assets.py` automatically before starting the policy server (`PREPULL=1` by default). This downloads HF checkpoint/text artifacts into `CHECKPOINT_CACHE_DIR` and fails before model startup if a config path points to an unavailable training-machine path. Normalization stats can come from the same artifact repo, but `HF_STATS_FILENAME` must be explicit because there are separate task stats files.

Prompt selection is server-side and embedded once at startup:

```bash
MIMIC_VIDEO_TASK=task1   # or task2
# or
MIMIC_VIDEO_PROMPT="custom prompt text"
```

See [docs/LEROBOT_ASYNC.md](docs/LEROBOT_ASYNC.md) for the full bring-up checklist.

## Network

LeRobot async is gRPC between the custom mimic-video robot client and policy server on port `8080` by default. For no-robot safety testing, use `scripts/smoke_mimic_video_grpc_client.py`; it exercises the same gRPC server without opening the robot serial port.
