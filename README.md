# SO101 mimic-video Deployment

LeRobot async deployment wrapper for `mimic-video` on SO101.

This repo keeps deployment separate from the training/research repo and vendors `mimic-video` as a git submodule. The submodule should track `video_finetune_improvements`.

## Clone

```bash
git clone <this-repo-url>
cd so101-mimic-video-deploy
git submodule update --init --recursive
git -C mimic-video checkout video_finetune_improvements
```

## Install On The GPU Policy Server

```bash
cd mimic-video/model
uv sync --extra cu126
source .venv/bin/activate
cd ../..

uv pip install "lerobot[async]"
uv pip install -e lerobot_policy_mimic_video
```

The LeRobot policy loads `mimic-video/model` directly and reads the same SO101 training YAML used for training. For the current dual-task video backbone, use:

```bash
TRAINING_CONFIG=$PWD/mimic-video/configs/so101-task1-mini-v1-birdview/video_finetune_subsampled_dual_task.yaml
```

That path is swappable, but the replacement YAML must carry the matching `preprocessing.image` fields and live in a checkout with the matching `model/cosmos_predict2/configs/dataloading/policy_io/so101.yaml`.

## Container

Build the robot action deployment image:

```bash
docker build -t so101-mimic-video-deploy .
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

See [docs/LEROBOT_ASYNC.md](docs/LEROBOT_ASYNC.md) for the full bring-up checklist.

## Network

LeRobot async is gRPC between the custom mimic-video robot client and policy server on port `8080` by default. For no-robot safety testing, use `scripts/smoke_mimic_video_grpc_client.py`; it exercises the same gRPC server without opening the robot serial port.
