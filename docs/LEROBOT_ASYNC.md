# LeRobot Async Deployment

This is the robot deployment path. It is gRPC end-to-end between LeRobot's `RobotClient` and `PolicyServer`; the mimic-video policy is loaded directly inside the LeRobot `PolicyServer`.

The standalone HTTP server is only for video debugging with `/generate_video`; it is stateless and each request carries explicit real image/state context.

## Topology

```text
GPU machine
  lerobot.async_inference.policy_server  (gRPC server)
    loads policy_type=mimic_video
    loads Cosmos video backbone + SO101 action decoder + stats
    returns action chunks

Robot machine
  lerobot.async_inference.robot_client  (gRPC client)
    reads SO101 camera/state
    executes returned action chunks at 10 Hz
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

If your LeRobot CLI version does not accept `--policy.*` overrides from `robot_client`, create a local policy config/checkpoint directory for `mimic_video` with these fields and pass that as `--pretrained_name_or_path`.

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

1. Use the standalone HTTP server only for video debugging with `scripts/video_debug_client.py`.
2. Start LeRobot `policy_server` on the GPU machine and confirm the robot client can connect.
3. Start with the robot disabled or lifted; log returned action chunks.
4. Enable the robot and execute one short chunk.
5. Increase to continuous async only after action sign, scale, and joint order look correct.

## Contract

- Image stream must match training camera/crop/resize.
- State must be the same 6D `observation.state` order used in training.
- Actions returned by mimic-video are 6D SO101 action vectors. Confirm whether your LeRobot SO101 action path expects absolute joint targets, deltas, or another representation before motor execution.
- Use LeRobot async for robot action deployment. Keep the standalone HTTP server for video debug only.
