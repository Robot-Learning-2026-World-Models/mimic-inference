#!/usr/bin/env python3
"""Client helpers for the mimic-video SO101 inference server."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SERVER_IMAGE_SIZE = (640, 480)


def resize_rgb(image: np.ndarray, size: tuple[int, int] = SERVER_IMAGE_SIZE) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return np.asarray(Image.fromarray(image.astype(np.uint8), "RGB").resize(size, Image.Resampling.BILINEAR))


def encode_jpeg(image: np.ndarray, quality: int = 90) -> str:
    buffer = io.BytesIO()
    Image.fromarray(resize_rgb(image), "RGB").save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_jpeg(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


@dataclass
class SO101MimicVideoClient:
    server_url: str
    timeout: float = 1800.0
    jpeg_quality: int = 90

    def health(self) -> dict[str, Any]:
        return self._get_json("/health")

    def reset(self, prompt: str = "") -> dict[str, Any]:
        return self._post_json("/reset", {"prompt": prompt})

    def plan_actions(
        self,
        *,
        prompt: str,
        image: np.ndarray | None = None,
        state: np.ndarray | list[float] | None = None,
        images: list[np.ndarray] | None = None,
        states: np.ndarray | list[list[float]] | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "seed": int(seed),
        }
        if images is not None:
            payload["images_base64"] = [encode_jpeg(frame, self.jpeg_quality) for frame in images]
        elif image is not None:
            payload["image_base64"] = encode_jpeg(image, self.jpeg_quality)
        else:
            raise ValueError("plan_actions requires image or images.")
        if states is not None:
            payload["states"] = np.asarray(states, dtype=np.float32).tolist()
        elif state is not None:
            payload["state"] = np.asarray(state, dtype=np.float32).tolist()
        else:
            raise ValueError("plan_actions requires state or states.")
        return self._post_json(
            "/plan_actions",
            payload,
        )

    def generate_video(
        self,
        *,
        prompt: str,
        image: np.ndarray | None = None,
        images: list[np.ndarray] | None = None,
        seed: int = 0,
        num_sampling_steps: int | None = None,
        max_return_frames: int = 8,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "seed": int(seed),
            "max_return_frames": int(max_return_frames),
        }
        if images is not None:
            payload["images_base64"] = [encode_jpeg(frame, self.jpeg_quality) for frame in images]
        elif image is not None:
            payload["image_base64"] = encode_jpeg(image, self.jpeg_quality)
        else:
            raise ValueError("generate_video requires image or images.")
        if num_sampling_steps is not None:
            payload["num_sampling_steps"] = int(num_sampling_steps)
        return self._post_json("/generate_video", payload)

    def generate_video_frames(self, **kwargs: Any) -> list[np.ndarray]:
        response = self.generate_video(**kwargs)
        return [decode_jpeg(frame_b64) for frame_b64 in response["frames_jpeg_base64"]]

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.server_url.rstrip("/") + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{path} failed with HTTP {exc.code}: {body}") from exc

    def _get_json(self, path: str) -> dict[str, Any]:
        with urllib.request.urlopen(self.server_url.rstrip("/") + path, timeout=self.timeout) as response:
            return json.loads(response.read())


class SO101ActionChunkPolicy:
    """Tiny robot-side policy adapter that owns the action chunk buffer.

    This is intentionally framework-light: LeRobot integration code can call
    `select_action(image=..., state=...)` from its control loop and pass the
    returned 6D action to the SO101 action path.
    """

    def __init__(
        self,
        *,
        client: SO101MimicVideoClient,
        prompt: str,
        seed: int = 0,
        replan_remaining: int = 1,
    ) -> None:
        self.client = client
        self.prompt = prompt
        self.seed = seed
        self.replan_remaining = replan_remaining
        self._actions: list[list[float]] = []

    def reset(self) -> None:
        self._actions.clear()
        self.client.reset(self.prompt)

    def select_action(self, *, image: np.ndarray, state: np.ndarray | list[float]) -> np.ndarray:
        if len(self._actions) <= self.replan_remaining:
            response = self.client.plan_actions(image=image, state=state, prompt=self.prompt, seed=self.seed)
            execute_n = int(response.get("num_execute_actions", len(response["actions"])))
            self._actions = response["actions"][:execute_n]
        if not self._actions:
            raise RuntimeError("No actions returned by mimic-video server.")
        return np.asarray(self._actions.pop(0), dtype=np.float32)


def save_frames_as_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264")
