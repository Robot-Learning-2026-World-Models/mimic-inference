#!/usr/bin/env python3
"""Small HTTP inference server for SO101 mimic-video policies."""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.types import NormalizationType
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.utils.config_helper import override

SO101_EXPERIMENT = "w2a_so101_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1"
IMAGE_SIZE_HW = (480, 640)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _decode_image(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    if image.size != (IMAGE_SIZE_HW[1], IMAGE_SIZE_HW[0]):
        image = image.resize((IMAGE_SIZE_HW[1], IMAGE_SIZE_HW[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _decode_images(payload: dict[str, Any]) -> list[np.ndarray]:
    if payload.get("images_base64"):
        images = payload["images_base64"]
        if not isinstance(images, list):
            raise ValueError("images_base64 must be a list.")
        return [_decode_image(str(image_b64)) for image_b64 in images]
    if payload.get("image_base64"):
        return [_decode_image(str(payload["image_base64"]))]
    return []


def _encode_jpeg(image: np.ndarray, quality: int = 90) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image.astype(np.uint8), "RGB").save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _process_image(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}.")
    if image.shape != (*IMAGE_SIZE_HW, 3):
        raise ValueError(f"Expected image shape {(IMAGE_SIZE_HW[0], IMAGE_SIZE_HW[1], 3)}, got {image.shape}.")
    tensor = rearrange(image, "h w c -> c h w")[:, None, :, :]
    return 2.0 * (tensor.astype(np.float32) / 255.0 - 0.5)


def _prepare_image_context(images: list[np.ndarray], horizon: int) -> np.ndarray:
    if not images:
        raise ValueError("At least one image is required.")
    processed = [_process_image(image) for image in images]
    if len(processed) < horizon:
        processed = [processed[0].copy() for _ in range(horizon - len(processed))] + processed
    return np.concatenate(processed[-horizon:], axis=1)


def _prepare_state_context(states: np.ndarray, horizon: int) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states[None]
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError(f"SO101 states must have shape (T, 6), got {states.shape}.")
    if len(states) < horizon:
        states = np.concatenate([np.repeat(states[:1], horizon - len(states), axis=0), states], axis=0)
    return states[-horizon:]


def _video_tensor_to_jpegs(video: torch.Tensor, max_return_frames: int) -> list[str]:
    video = video[0].detach().float().cpu().clamp(-1.0, 1.0)
    frames = ((video + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).numpy()
    if max_return_frames <= 0:
        return []
    return [_encode_jpeg(frame) for frame in frames[:max_return_frames]]


def _build_so101_normalizer(pipe: World2ActionPipeline, stats_path: Path, dtype: torch.dtype) -> None:
    stats = _load_json(stats_path)
    pipe.normalizer.build_from_stats(
        stats,
        normalization_types={
            "action/joint_action_lowdim": NormalizationType.VARIANCE,
            "obs/joint_state_lowdim": NormalizationType.VARIANCE,
        },
        concat_groups={
            "action/lowdim_concat": ["action/joint_action_lowdim"],
            "obs/lowdim_concat": ["obs/joint_state_lowdim"],
        },
        device="cuda",
        dtype=dtype,
    )


def load_pipeline(
    *,
    experiment_name: str,
    video_model_path: Path,
    action_model_path: Path | None,
    stats_path: Path | None,
    dtype: torch.dtype,
) -> Video2World2ActionPipeline | Video2WorldPipeline:
    config = make_config()
    config = override(config, ["--", f"experiment={experiment_name}"])
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    video2world_pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path=str(video_model_path),
        device="cuda",
        torch_dtype=dtype,
        load_ema_to_reg=False,
    )
    if action_model_path is None or stats_path is None:
        return video2world_pipe.cuda().eval()

    world2action_pipe = World2ActionPipeline.from_config(
        config.model.config.pipe_config,
        dit_path=str(action_model_path),
        device="cuda",
        dtype=dtype,
    )
    _build_so101_normalizer(world2action_pipe, stats_path, dtype)
    return Video2World2ActionPipeline(video2world_pipe, world2action_pipe).cuda().eval()


class SO101PolicyServer:
    def __init__(
        self,
        *,
        pipeline: Video2World2ActionPipeline | Video2WorldPipeline,
        image_horizon: int,
        lowdim_horizon: int,
        stop_video_denoising_step: int,
        num_execute_actions: int,
        num_sampling_steps: int,
        video_fps: float,
        action_hz: float,
        use_cuda_graphs: bool,
    ) -> None:
        self.pipeline = pipeline
        self.image_horizon = image_horizon
        self.lowdim_horizon = lowdim_horizon
        self.stop_video_denoising_step = stop_video_denoising_step
        self.num_execute_actions = num_execute_actions
        self.num_sampling_steps = num_sampling_steps
        self.video_fps = video_fps
        self.action_hz = action_hz
        self.use_cuda_graphs = use_cuda_graphs
        self.lock = threading.Lock()

    def reset(self, prompt: str) -> None:
        return None

    def plan_actions(
        self,
        *,
        images: list[np.ndarray],
        states: np.ndarray,
        prompt: str,
        seed: int,
    ) -> dict[str, Any]:
        with self.lock:
            input_images = _prepare_image_context(images, self.image_horizon)
            lowdims = _prepare_state_context(states, self.lowdim_horizon)
            input_vid = torch.from_numpy(input_images[None]).cuda().to(dtype=torch.bfloat16)
            state_tensor = torch.from_numpy(lowdims[None]).cuda().to(dtype=torch.bfloat16)

            if not isinstance(self.pipeline, Video2World2ActionPipeline):
                raise RuntimeError("This server was started video-only; /plan_actions requires action weights.")

            pred_actions = self.pipeline(
                input_vid=input_vid,
                state_B_HO_O=state_tensor,
                prompt=prompt,
                num_sampling_step=self.num_sampling_steps,
                stop_after_step=self.stop_video_denoising_step,
                seed=seed,
                use_cuda_graphs=self.use_cuda_graphs,
            )
            action_chunk = pred_actions[0].float().cpu().numpy()

            return {
                "actions": action_chunk.tolist(),
                "action_hz": float(self.action_hz),
                "num_execute_actions": int(self.num_execute_actions),
                "image_horizon": int(self.image_horizon),
                "lowdim_horizon": int(self.lowdim_horizon),
            }

    def generate_video(
        self,
        *,
        images: list[np.ndarray],
        prompt: str,
        seed: int,
        num_sampling_steps: int,
        max_return_frames: int,
    ) -> dict[str, Any]:
        with self.lock:
            input_images = _prepare_image_context(images, self.image_horizon)
            input_vid = torch.from_numpy(input_images[None]).cuda().to(dtype=torch.bfloat16)
            num_latent_conditional_frames = 1 if input_vid.shape[2] == 1 else 2

            video_pipeline = (
                self.pipeline.video2world_pipeline
                if isinstance(self.pipeline, Video2World2ActionPipeline)
                else self.pipeline
            )
            video = video_pipeline.generate_video(
                vid_input=input_vid,
                num_latent_conditional_frames=num_latent_conditional_frames,
                prompt=prompt,
                negative_prompt="",
                guidance=0.0,
                num_sampling_step=num_sampling_steps,
                seed=seed,
                use_cuda_graphs=self.use_cuda_graphs,
            )
            return {
                "frames_jpeg_base64": _video_tensor_to_jpegs(video, max_return_frames),
                "num_returned_frames": min(max_return_frames, int(video.shape[2])),
                "total_generated_frames": int(video.shape[2]),
            }


def make_handler(server_state: SO101PolicyServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length))

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "cuda": torch.cuda.is_available(),
                        "image_horizon": server_state.image_horizon,
                        "lowdim_horizon": server_state.lowdim_horizon,
                        "video_fps": server_state.video_fps,
                        "action_hz": server_state.action_hz,
                        "num_execute_actions": server_state.num_execute_actions,
                        "mode": "actions" if isinstance(server_state.pipeline, Video2World2ActionPipeline) else "video",
                    },
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json()
                if self.path == "/reset":
                    server_state.reset(str(payload.get("prompt", "")))
                    self._send_json(HTTPStatus.OK, {"status": "reset"})
                    return

                if self.path == "/plan_actions":
                    response = server_state.plan_actions(
                        images=_decode_images(payload),
                        states=np.asarray(payload.get("states", payload.get("state")), dtype=np.float32),
                        prompt=str(payload["prompt"]),
                        seed=int(payload.get("seed", 0)),
                    )
                    self._send_json(HTTPStatus.OK, response)
                    return

                if self.path == "/generate_video":
                    response = server_state.generate_video(
                        images=_decode_images(payload),
                        prompt=str(payload["prompt"]),
                        seed=int(payload.get("seed", 0)),
                        num_sampling_steps=int(payload.get("num_sampling_steps", server_state.num_sampling_steps)),
                        max_return_frames=int(payload.get("max_return_frames", 8)),
                    )
                    self._send_json(HTTPStatus.OK, response)
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path {self.path}"})
            except Exception as exc:  # noqa: BLE001
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}", flush=True)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--experiment-name", default=SO101_EXPERIMENT)
    parser.add_argument("--video-model-path", required=True, type=Path)
    parser.add_argument("--action-model-path", type=Path)
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument("--image-horizon", type=int, default=5)
    parser.add_argument("--lowdim-horizon", type=int, default=1)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--action-hz", type=float, default=10.0)
    parser.add_argument("--stop-video-denoising-step", type=int, default=0)
    parser.add_argument("--num-execute-actions", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=35)
    parser.add_argument("--disable-cuda-graphs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SO101 inference requires an NVIDIA GPU with CUDA available.")
    if (args.action_model_path is None) != (args.stats_path is None):
        raise ValueError("--action-model-path and --stats-path must be provided together, or both omitted.")

    pipeline = load_pipeline(
        experiment_name=args.experiment_name,
        video_model_path=args.video_model_path,
        action_model_path=args.action_model_path,
        stats_path=args.stats_path,
        dtype=torch.bfloat16,
    )
    state = SO101PolicyServer(
        pipeline=pipeline,
        image_horizon=args.image_horizon,
        lowdim_horizon=args.lowdim_horizon,
        stop_video_denoising_step=args.stop_video_denoising_step,
        num_execute_actions=args.num_execute_actions,
        num_sampling_steps=args.num_sampling_steps,
        video_fps=args.video_fps,
        action_hz=args.action_hz,
        use_cuda_graphs=not args.disable_cuda_graphs,
    )
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Serving SO101 inference on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
