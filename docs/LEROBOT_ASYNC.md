# LeRobot Async Deployment

This is the robot deployment path. It is gRPC end-to-end between the custom mimic-video LeRobot-compatible `RobotClient` and `PolicyServer`; the mimic-video policy is loaded directly inside the policy server.

## Topology

```text
GPU machine
  lerobot_policy_mimic_video.grpc_policy_server  (gRPC server)
    loads policy_type=mimic_video
    loads Cosmos video backbone + SO101 action decoder + stats
    returns action chunks

Robot machine
  lerobot_policy_mimic_video.grpc_robot_client  (gRPC client)
    reads SO101 camera/state
    sends a raw camera frame window with every observation
    executes returned action chunks one action per control tick at 30 Hz
```

## 1. Install The LeRobot Policy Plugin

Run this in the LeRobot environment on the machine that starts the LeRobot async policy server:

```bash
cd mimic-video/model
uv sync --extra cu126
source .venv/bin/activate
cd ../..

uv pip install "lerobot[async]"
uv pip install -e lerobot_policy_mimic_video
```

That environment must also be able to import the mimic-video `model/` package and load the Cosmos CUDA dependencies. In practice, run the LeRobot `PolicyServer` from this repository root or add `mimic-video/model/` to `PYTHONPATH`.

## 2. Start LeRobot Async Policy Server

In terminal A on the GPU machine:

```bash
commands/lerobot_async_policy_server.sh
```

The default server port is `8080`.

## 3. Start SO101 Robot Client

In terminal B on the robot machine, replace paths and robot/camera settings with your setup:

```bash
ROBOT_PORT=/dev/tty.usbmodemXXXX \
SERVER_ADDRESS=<policy-server-host>:8080 \
VIDEO_MODEL_PATH=/abs/path/to/video_backbone.pt \
ACTION_MODEL_PATH=/abs/path/to/action_decoder.pt \
ACTION_DATA_DIR=/abs/path/to/action_decoder \
TRAINING_CONFIG=/abs/path/to/mimic-video/configs/so101-task1-mini-v1-birdview/video_finetune_subsampled_dual_task.yaml \
ACTIONS_PER_CHUNK=3 \
commands/lerobot_async_so101_client.sh
```

Use `STATS_PATH` instead of `ACTION_DATA_DIR` when you have a specific statistics JSON. To download stats from Hugging Face at startup, set `STATS_PATH` to the desired local path and provide `HF_STATS_REPO_ID`, `HF_STATS_FILENAME`, and optionally `HF_STATS_REVISION`.

## Training YAML Contract

`TRAINING_CONFIG` is the source of truth for deployment preprocessing. The plugin reads:

- `preprocessing.sources[0].instruction` or `preprocessing.instruction` for the prompt.
- `preprocessing.sources[0].video_feature` or `preprocessing.video_feature` for the camera key.
- `preprocessing.image.crop_xywh` and `preprocessing.image.output_size_hw` for crop/resize.
- `training.action_decoder.experiment`, if present, to choose the Cosmos action experiment name.
- `training.action_decoder.model`, `network`, and `action_scheduler`, if present, for action-decoder parameters.
- `model/cosmos_predict2/configs/dataloading/policy_io/so101.yaml` beside the YAML checkout for image, state, and action horizons.

The current intended config is:

```text
mimic-video/configs/so101-task1-mini-v1-birdview/video_finetune_subsampled_dual_task.yaml
```

That YAML currently has `training.action_decoder.enabled: false`, so the action checkpoint/statistics still need to be supplied explicitly.

## Bring-Up Order

1. Prepare action statistics with `scripts/prepare_so101_inference_stats.py`.
2. Start `commands/lerobot_async_policy_server.sh` on the GPU machine.
3. Run `scripts/smoke_mimic_video_grpc_client.py` and confirm it receives action chunks without the robot.
4. Start with the robot disabled or lifted; log returned action chunks.
5. Enable the robot and execute one short chunk.
6. Increase to continuous async only after action sign, scale, and joint order look correct.

## Contract

- Image stream must match training camera/crop/resize.
- State must be the same 6D `observation.state` order used in training.
- Actions returned by mimic-video are 6D SO101 action vectors. Confirm whether your LeRobot SO101 action path expects absolute joint targets, deltas, or another representation before motor execution.
- Use the custom gRPC client/server in this repo for robot action deployment.
