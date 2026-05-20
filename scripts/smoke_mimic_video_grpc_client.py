#!/usr/bin/env python3
"""No-robot gRPC smoke client for mimic-video LeRobot async inference."""

from __future__ import annotations

import argparse
import pickle  # nosec
import time
from pathlib import Path
from typing import Any

import grpc
import numpy as np
from PIL import Image

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks


def _load_image(path: Path | None, *, width: int, height: int) -> np.ndarray:
    if path is None:
        yy, xx = np.mgrid[:height, :width]
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[..., 0] = (xx % 256).astype(np.uint8)
        image[..., 1] = (yy % 256).astype(np.uint8)
        image[..., 2] = 127
        return image
    return np.asarray(Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR))


def _make_frame_window(image: np.ndarray, raw_image_horizon: int) -> np.ndarray:
    frames = []
    for i in range(raw_image_horizon):
        frame = image.copy()
        frame[..., 2] = np.clip(frame[..., 2].astype(np.int16) + i, 0, 255).astype(np.uint8)
        frames.append(frame)
    return np.stack(frames, axis=0)


def _lerobot_features(camera_key: str, *, width: int, height: int, action_dim: int) -> dict[str, dict[str, Any]]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"joint_{i}" for i in range(action_dim)],
        },
        f"observation.images.{camera_key}": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--camera-key", default="front")
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--raw-image-horizon", type=int, default=13)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--actions-per-chunk", type=int, default=3)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    channel = grpc.insecure_channel(args.server_address, grpc_channel_options(initial_backoff="0.0333s"))
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    stub.Ready(services_pb2.Empty(), timeout=args.timeout)

    policy_config = RemotePolicyConfig(
        "mimic_video",
        ".",
        _lerobot_features(args.camera_key, width=args.width, height=args.height, action_dim=args.action_dim),
        args.actions_per_chunk,
        args.policy_device,
    )
    stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy_config)), timeout=args.timeout)

    image = _load_image(args.image_path, width=args.width, height=args.height)
    raw_observation = {
        "observation.state": np.zeros((args.action_dim,), dtype=np.float32),
        f"{args.camera_key}_frame_window": _make_frame_window(image, args.raw_image_horizon),
        "task": "smoke test",
    }
    timed_observation = TimedObservation(
        timestamp=time.time(),
        observation=raw_observation,
        timestep=0,
    )
    timed_observation.must_go = True
    obs_bytes = pickle.dumps(timed_observation)
    stub.SendObservations(
        send_bytes_in_chunks(obs_bytes, services_pb2.Observation, log_prefix="[SMOKE] Observation", silent=True),
        timeout=args.timeout,
    )

    actions_msg = stub.GetActions(services_pb2.Empty(), timeout=args.timeout)
    if not actions_msg.data:
        raise RuntimeError("Server returned no actions. Check policy server logs.")
    timed_actions = pickle.loads(actions_msg.data)  # nosec
    print(f"received_actions={len(timed_actions)}")
    for idx, action in enumerate(timed_actions[: min(3, len(timed_actions))]):
        tensor = action.get_action()
        print(f"action[{idx}] timestep={action.get_timestep()} shape={tuple(tensor.shape)} values={tensor.tolist()}")


if __name__ == "__main__":
    main()
