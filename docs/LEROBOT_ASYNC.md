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

Checkpoint resolution happens on the policy server. Explicit `VIDEO_MODEL_PATH` / `ACTION_MODEL_PATH` wins. If unset, the server uses `training.action_decoder.model.video_dit_path` and `action_dit_path` from `TRAINING_CONFIG`. If those paths are missing or unavailable, provide HF download envs. By default, `MIMIC_VIDEO_HF_REPO_ID=rl26-world-models/mimic-finetune` is used as the shared repo for video/action/text assets:

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

`commands/lerobot_async_policy_server.sh` prepulls these assets by default before starting gRPC (`PREPULL=1`). Set `PREPULL=0` only when all paths are already local and validated. Dataset/action normalization stats can come from the same artifact repo, but `HF_STATS_FILENAME` must be explicit because there are separate task stats files.

Prompt selection is server-side. Set `MIMIC_VIDEO_TASK=task1` or `MIMIC_VIDEO_TASK=task2`, or provide `MIMIC_VIDEO_PROMPT="..."`. The selected prompt is embedded once during startup; runtime robot observations do not re-run T5.

## Training YAML Contract

`TRAINING_CONFIG` is the source of truth for deployment preprocessing. The plugin reads:

- `preprocessing.sources[0].instruction` or `preprocessing.instruction` for the prompt.
- `preprocessing.sources[0].video_feature` or `preprocessing.video_feature` for the camera key.
- `preprocessing.image.crop_xywh` and `preprocessing.image.output_size_hw` for crop/resize.
- `training.action_decoder.experiment`, if present, to choose the Cosmos action experiment name.
- `training.action_decoder.model`, `network`, and `action_scheduler`, if present, for action-decoder parameters.
- `model/cosmos_predict2/configs/dataloading/policy_io/<experiment>.yaml` for image, state, and action horizons. The loader first checks beside the YAML checkout, then `MIMIC_VIDEO_REPO_ROOT`, the current working tree, and the installed package location, so exported YAMLs outside the repo can still use repo-local policy IO files.

For the `/Volumes/Transfer/idm_long_700*` checkpoint set:

```bash
TRAINING_CONFIG=/Volumes/Transfer/idm_long_700_config.yaml
VIDEO_MODEL_PATH=/Volumes/Transfer/idm_long_700_backbone.pt
ACTION_MODEL_PATH=/Volumes/Transfer/idm_long_700.pt
STATS_PATH=/Volumes/Transfer/idm_long_700_stats.json
PREPULL=0
```

That config routes through `policy_io/so101_90.yaml`: 5 image observation
frames at 10 Hz, 56 future video frames, `video_pipe_state_t=16`, 90 lowdim
actions at 30 Hz, `max_horizon=91`, and `xattn_layer_idx=16`.

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
