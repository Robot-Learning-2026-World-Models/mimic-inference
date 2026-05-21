#!/usr/bin/env python3
"""Capture a camera window and save the frames after mimic-video image transforms."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lerobot_policy_mimic_video/src"))

from lerobot_policy_mimic_video.training_config import load_policy_io  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--camera-index", default="0", help="OpenCV camera index or video path.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--raw-image-horizon", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/camera_transform_debug"))
    return parser.parse_args()


def _load_transform_config(training_config: Path) -> tuple[tuple[int, int, int, int], tuple[int, int], int, int]:
    with training_config.expanduser().open() as f:
        data = yaml.safe_load(f)

    preprocessing = data["preprocessing"]
    image_cfg = preprocessing["image"]
    action_cfg = data["training"].get("action_decoder") or {}
    dataset_name = str(action_cfg.get("experiment") or "so101")
    policy_io = load_policy_io(training_config, dataset_name)

    crop_xywh = tuple(int(v) for v in image_cfg["crop_xywh"])
    output_size_hw = tuple(int(v) for v in image_cfg["output_size_hw"])
    image_horizon = int(policy_io.image_horizon)
    image_frequency = int(policy_io.image_frequency)
    return crop_xywh, output_size_hw, image_horizon, image_frequency


def _process_image(image: np.ndarray, crop_xywh: tuple[int, int, int, int], output_size_hw: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), "RGB")
    x, y, w, h = crop_xywh
    pil_image = pil_image.crop((x, y, x + w, y + h))
    out_h, out_w = output_size_hw
    pil_image = pil_image.resize((out_w, out_h), Image.Resampling.BILINEAR)
    return np.asarray(pil_image)


def _contact_sheet(frames: list[np.ndarray], labels: list[str]) -> Image.Image:
    pil_frames = [Image.fromarray(frame.astype(np.uint8), "RGB") for frame in frames]
    widths, heights = zip(*(frame.size for frame in pil_frames), strict=True)
    label_h = 24
    sheet = Image.new("RGB", (sum(widths), max(heights) + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    x = 0
    for frame, label in zip(pil_frames, labels, strict=True):
        sheet.paste(frame, (x, label_h))
        draw.text((x + 4, 4), label, fill=(0, 0, 0))
        x += frame.size[0]
    return sheet


def _camera_source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    args = _parse_args()
    crop_xywh, output_size_hw, image_horizon, image_frequency = _load_transform_config(args.training_config)
    if args.fps % image_frequency != 0:
        raise ValueError(f"--fps must be a multiple of image_frequency={image_frequency}, got {args.fps}.")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required. Try running this inside the mimic-video venv.") from exc

    cap = cv2.VideoCapture(_camera_source(args.camera_index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source {args.camera_index!r}.")

    frames: list[np.ndarray] = []
    try:
        while len(frames) < args.raw_image_horizon:
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError("Camera read failed.")
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            time.sleep(max(0.0, 1.0 / args.fps))
    finally:
        cap.release()

    step = args.fps // image_frequency
    raw_horizon = 1 + (image_horizon - 1) * step
    if len(frames) < raw_horizon:
        frames = [frames[0].copy() for _ in range(raw_horizon - len(frames))] + frames
    selected = frames[-raw_horizon::step]
    transformed = [_process_image(frame, crop_xywh, output_size_hw) for frame in selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frames[-1]).save(args.output_dir / "raw_latest.jpg")
    selected_dir = args.output_dir / "selected_raw_png"
    transformed_dir = args.output_dir / "transformed_png"
    selected_dir.mkdir(exist_ok=True)
    transformed_dir.mkdir(exist_ok=True)
    for idx, frame in enumerate(selected):
        Image.fromarray(frame).save(selected_dir / f"frame_{idx:02d}.png")
    for idx, frame in enumerate(transformed):
        Image.fromarray(frame).save(transformed_dir / f"frame_{idx:02d}.png")
    _contact_sheet(selected, [f"selected_{idx}" for idx in range(len(selected))]).save(
        args.output_dir / "selected_raw_contact_sheet.jpg"
    )
    _contact_sheet(transformed, [f"transformed_{idx}" for idx in range(len(transformed))]).save(
        args.output_dir / "transformed_contact_sheet.jpg"
    )
    np.save(args.output_dir / "transformed_frames_rgb.npy", np.stack(transformed, axis=0))

    print(f"crop_xywh={crop_xywh}")
    print(f"output_size_hw={output_size_hw}")
    print(f"image_horizon={image_horizon} image_frequency={image_frequency} input_fps={args.fps} step={step}")
    print(f"selected_frames={len(selected)} transformed_shape={np.stack(transformed, axis=0).shape}")
    print(f"selected_png_dir={selected_dir.resolve()}")
    print(f"transformed_png_dir={transformed_dir.resolve()}")
    print(f"wrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
