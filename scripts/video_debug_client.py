#!/usr/bin/env python3
"""Compare mimic-video predicted frames against a real demo video."""

from __future__ import annotations

import argparse
import time
import urllib.error
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

from so101_client import SERVER_IMAGE_SIZE, SO101MimicVideoClient, resize_rgb, save_frames_as_mp4


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = Image.fromarray(image.astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, 220, 28), fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255))
    return np.asarray(out, dtype=np.uint8)


def side_by_side(predicted: np.ndarray, actual: np.ndarray, step: int) -> np.ndarray:
    predicted = resize_rgb(predicted, SERVER_IMAGE_SIZE)
    actual = resize_rgb(actual, SERVER_IMAGE_SIZE)
    predicted = label(predicted, f"predicted +{step}")
    actual = label(actual, f"actual +{step}")
    return np.concatenate([predicted, actual], axis=1)


def context_window(frames: list[np.ndarray], end: int, horizon: int) -> list[np.ndarray]:
    start = max(0, end - horizon + 1)
    context = frames[start : end + 1]
    if len(context) < horizon:
        context = [context[0]] * (horizon - len(context)) + context
    return context


def load_video(path: Path, max_frames: int | None) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for frame in iio.imiter(path):
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frames.append(resize_rgb(frame, SERVER_IMAGE_SIZE))
        if max_frames is not None and len(frames) >= max_frames:
            break
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True, help="Example: http://127.0.0.1:8000")
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--horizon", type=int, default=8, help="Predicted frames to request per reconditioning step.")
    parser.add_argument("--stride", type=int, default=8, help="Advance this many real frames before reconditioning again.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num-sampling-steps", type=int, default=35)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=5.0)
    parser.add_argument("--max-input-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def generate_with_retries(
    *,
    client: SO101MimicVideoClient,
    args: argparse.Namespace,
    frames: list[np.ndarray],
    start: int,
    image_horizon: int,
    max_return_frames: int,
) -> tuple[list[np.ndarray], float]:
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        t0 = time.perf_counter()
        try:
            predicted_frames = client.generate_video_frames(
                image=None,
                images=context_window(frames, start, image_horizon),
                prompt=args.prompt,
                seed=args.seed,
                num_sampling_steps=args.num_sampling_steps,
                max_return_frames=max_return_frames,
            )
            return predicted_frames, time.perf_counter() - t0
        except (TimeoutError, urllib.error.URLError, ConnectionError, RuntimeError) as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            delay = args.retry_backoff * (attempt + 1)
            print(
                f"recondition frame {start}: request failed ({exc}); retrying in {delay:.1f}s "
                f"({attempt + 1}/{args.retries})",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"recondition frame {start} failed after {args.retries + 1} attempts") from last_error


def main() -> None:
    args = parse_args()
    client = SO101MimicVideoClient(
        server_url=args.server_url,
        timeout=args.timeout,
        jpeg_quality=args.jpeg_quality,
    )
    health = client.health()
    print(f"Server health: {health}", flush=True)

    frames = load_video(args.input_video, args.max_input_frames)
    output_frames: list[np.ndarray] = []
    image_horizon = int(health.get("image_horizon", 5))
    for start in range(0, max(1, len(frames) - 1), args.stride):
        actual_future = frames[start + 1 : start + 1 + args.horizon]
        if not actual_future:
            break

        predicted_frames, elapsed = generate_with_retries(
            client=client,
            args=args,
            frames=frames,
            start=start,
            image_horizon=image_horizon,
            max_return_frames=len(actual_future),
        )
        print(
            f"recondition frame {start}: got {len(predicted_frames)} predicted frames in {elapsed:.2f}s",
            flush=True,
        )

        for offset, (predicted, actual) in enumerate(zip(predicted_frames, actual_future), start=1):
            output_frames.append(side_by_side(predicted, actual, offset))

    if not output_frames:
        raise RuntimeError("No comparison frames were produced.")

    save_frames_as_mp4(args.output_video, output_frames, args.fps)
    print(f"Wrote {args.output_video}", flush=True)


if __name__ == "__main__":
    main()
