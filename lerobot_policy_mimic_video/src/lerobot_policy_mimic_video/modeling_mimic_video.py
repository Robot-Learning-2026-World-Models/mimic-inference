from __future__ import annotations

import json
import os
import pickle  # nosec
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
try:
    from lerobot.policies import PreTrainedPolicy
except ImportError:
    from lerobot.policies.pretrained import PreTrainedPolicy
from PIL import Image

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.types import NormalizationType
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.utils.config_helper import override

from .configuration_mimic_video import MimicVideoPolicyConfig
from .training_config import build_deployment_config, load_stats


FRAME_WINDOW_TIMESTAMPS_SUFFIX = "_frame_window_timestamps"
CLIENT_OBSERVATION_ELAPSED_KEY = "client_observation_elapsed_s"


class MimicVideoPolicy(PreTrainedPolicy):
    """LeRobot policy that runs mimic-video directly in the async PolicyServer."""

    config_class = MimicVideoPolicyConfig
    name = "mimic_video"

    def __init__(self, config: MimicVideoPolicyConfig, dataset_stats: dict[str, Any] | None = None):
        self.deployment_config = build_deployment_config(
            training_config=config.training_config,
            stats_path=config.stats_path,
            action_data_dir=config.action_data_dir,
            hf_stats_repo_id=config.hf_stats_repo_id,
            hf_stats_filename=config.hf_stats_filename,
            hf_stats_revision=config.hf_stats_revision,
        )
        self._apply_deployment_config(config)
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        self.pipeline = self._load_pipeline(config)
        self.prompt_embedding = self._load_prompt_embedding(config)
        self._preprocessed_input_notice_printed = False
        self._debug_run_root: Path | None = None
        self._debug_run_start: float | None = None

    def _apply_deployment_config(self, config: MimicVideoPolicyConfig) -> None:
        dc = self.deployment_config
        config.prompt = dc.prompt or config.prompt
        config.camera_key = dc.camera_key
        config.image_height = dc.image_height
        config.image_width = dc.image_width
        config.crop_x, config.crop_y, config.crop_width, config.crop_height = dc.crop_xywh
        config.experiment_name = dc.experiment_name
        config.xattn_layer_idx = dc.xattn_layer_idx
        config.action_dim = dc.action_dim
        config.horizon = dc.action_horizon
        config.image_horizon = dc.image_horizon
        config.lowdim_horizon = dc.lowdim_horizon
        config.image_frequency = dc.image_frequency
        config.lowdim_frequency = dc.lowdim_frequency
        config.action_frequency = dc.action_frequency
        config.max_horizon = dc.max_horizon
        config.video_pipe_state_t = dc.video_pipe_state_t
        config.action_denoising_steps = dc.action_denoising_steps
        config.stats_path = dc.stats_path
        config._mimic_video_network_config = dc.network_config

    def reset(self) -> None:
        return None

    def get_optim_params(self) -> dict:
        return {"params": []}

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        prompt = self.config.prompt
        images, seed_frames, seed_frame_timestamps = self._image_context_and_seed_frames_from_batch(batch)
        client_observation_elapsed_s = self._client_observation_elapsed_from_batch(batch)
        lowdims = self._state_context_from_batch(batch)

        input_vid = torch.from_numpy(images[None]).cuda().to(dtype=torch.bfloat16)
        state_tensor = torch.from_numpy(lowdims[None]).cuda().to(dtype=torch.bfloat16)

        with torch.no_grad():
            pipeline_result = self.pipeline(
                input_vid=input_vid,
                state_B_HO_O=state_tensor,
                prompt=prompt,
                prompt_embedding=self.prompt_embedding if prompt == self.config.prompt else None,
                num_sampling_step=self.config.num_sampling_steps,
                stop_after_step=self.config.stop_video_denoising_step,
                seed=self.config.seed,
                use_cuda_graphs=self.config.use_cuda_graphs,
                return_debug_outputs=self.config.debug_save_denoised_video,
            )

        if self.config.debug_save_denoised_video:
            full_action_batch = pipeline_result["actions"].float()
            actions = full_action_batch[0]
            self._save_debug_outputs(
                video_denoiser_output=pipeline_result["video_denoiser_output"],
                full_action_batch=full_action_batch,
                returned_actions=actions[: self.config.horizon],
                server_chunk_actions=actions[: self.config.n_action_steps],
                seed_frames=seed_frames,
                seed_frame_timestamps=seed_frame_timestamps,
                client_observation_elapsed_s=client_observation_elapsed_s,
                prompt=prompt,
                context_sigma=pipeline_result["context_sigma"],
                context_step=pipeline_result["context_step"],
            )
        else:
            actions = pipeline_result[0].float()

        actions = actions[: self.config.horizon].cpu()
        if actions.ndim != 2 or actions.shape[1] != self.config.action_dim:
            raise RuntimeError(f"Expected action chunk shape (H, {self.config.action_dim}), got {tuple(actions.shape)}.")
        return actions.unsqueeze(0)

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return self.predict_action_chunk(batch, **kwargs)[0, 0]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("mimic_video is an inference-only policy.")

    def _save_debug_outputs(
        self,
        *,
        video_denoiser_output: dict[str, Any],
        full_action_batch: torch.Tensor,
        returned_actions: torch.Tensor,
        server_chunk_actions: torch.Tensor,
        seed_frames: list[np.ndarray],
        seed_frame_timestamps: list[float] | None,
        client_observation_elapsed_s: float | None,
        prompt: str,
        context_sigma: torch.Tensor,
        context_step: torch.Tensor,
    ) -> None:
        run_dir, server_elapsed_s = self._make_debug_run_dir(client_observation_elapsed_s=client_observation_elapsed_s)
        video_denoiser_output = dict(video_denoiser_output)
        video_denoiser_output["video_model_path"] = str(self.config.video_model_path)
        video_denoiser_output["experiment_name"] = self.config.experiment_name
        video_denoiser_output["video_pipe_state_t"] = self.config.video_pipe_state_t
        video_denoiser_output["debug_run_dir"] = str(run_dir)
        torch.save(video_denoiser_output, run_dir / "video_denoiser_output.pt")

        seed_frame_dir = run_dir / "seed_frames"
        seed_frame_dir.mkdir(parents=True, exist_ok=False)
        for idx, frame in enumerate(seed_frames):
            if seed_frame_timestamps is None:
                filename = f"{idx}.png"
            else:
                filename = f"{idx}_{seed_frame_timestamps[idx]:.3f}s.png"
            Image.fromarray(frame).save(seed_frame_dir / filename)

        full_actions_np = full_action_batch.detach().float().cpu().numpy()
        returned_actions_np = returned_actions.detach().float().cpu().numpy()
        server_chunk_actions_np = server_chunk_actions.detach().float().cpu().numpy()

        sigma = context_sigma.detach().float().cpu()
        actions_json = {
            "full_batch": full_actions_np.tolist(),
            "returned": returned_actions_np.tolist(),
            "server_chunk": server_chunk_actions_np.tolist(),
            "shapes": {
                "full_batch": list(full_actions_np.shape),
                "returned": list(returned_actions_np.shape),
                "server_chunk": list(server_chunk_actions_np.shape),
            },
            "seed": self.config.seed,
            "prompt": prompt,
            "elapsed_s": server_elapsed_s,
            "server_elapsed_s": server_elapsed_s,
            "client_observation_elapsed_s": client_observation_elapsed_s,
            "scheduler_step": int(video_denoiser_output.get("scheduler_step", 0)),
            "num_sampling_steps": self.config.num_sampling_steps,
            "context_step": int(context_step.detach().cpu().item()),
            "context_sigma": sigma.tolist(),
            "video_denoiser_output": "video_denoiser_output.pt",
            "seed_frames": [
                f"seed_frames/{idx}.png"
                if seed_frame_timestamps is None
                else f"seed_frames/{idx}_{seed_frame_timestamps[idx]:.3f}s.png"
                for idx in range(len(seed_frames))
            ],
            "seed_frame_client_elapsed_s": seed_frame_timestamps,
        }
        with open(run_dir / "actions.json", "w", encoding="utf-8") as f:
            json.dump(actions_json, f, indent=2)

    def _make_debug_run_dir(self, *, client_observation_elapsed_s: float | None) -> tuple[Path, float]:
        run_root = self._get_debug_run_root()
        elapsed = self._debug_elapsed_s()
        if client_observation_elapsed_s is None:
            folder_name = f"server_{elapsed:.3f}s"
        else:
            folder_name = f"server_{elapsed:.3f}s_client_{client_observation_elapsed_s:.3f}s"
        run_dir = run_root / folder_name
        if run_dir.exists():
            run_dir = run_root / f"{folder_name}_{time.time_ns()}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir, elapsed

    def _get_debug_run_root(self) -> Path:
        if self._debug_run_root is not None:
            return self._debug_run_root

        output_dir = Path(self.config.debug_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.debug_run_name:
            run_root = output_dir / self.config.debug_run_name
            run_root.mkdir(parents=True, exist_ok=False)
        else:
            next_id = 1
            while True:
                run_root = output_dir / str(next_id)
                try:
                    run_root.mkdir(parents=True, exist_ok=False)
                    break
                except FileExistsError:
                    next_id += 1

        self._debug_run_root = run_root
        self._debug_run_start = time.monotonic()
        return run_root

    def _debug_elapsed_s(self) -> float:
        if self._debug_run_start is None:
            return 0.0
        return time.monotonic() - self._debug_run_start

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        if self._frame_matches_target_resolution(image):
            resized = image.astype(np.uint8, copy=False)
        else:
            pil_image = Image.fromarray(image.astype(np.uint8), "RGB")
            x, y = self.config.crop_x, self.config.crop_y
            w, h = self.config.crop_width, self.config.crop_height
            pil_image = pil_image.crop((x, y, x + w, y + h))
            pil_image = pil_image.resize((self.config.image_width, self.config.image_height), Image.Resampling.BILINEAR)
            resized = np.asarray(pil_image)
        tensor = rearrange(resized, "h w c -> c h w")[:, None, :, :]
        return 2.0 * (tensor.astype(np.float32) / 255.0 - 0.5)

    def _process_image_for_visualization(self, image: np.ndarray) -> np.ndarray:
        processed = self._process_image(image)
        frame = processed[:, 0]
        frame = np.moveaxis(frame, 0, -1)
        return np.clip((frame + 1.0) * 127.5, 0, 255).astype(np.uint8)

    def _image_context_from_batch(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        images, _, _ = self._image_context_and_seed_frames_from_batch(batch)
        return images

    def _image_context_and_seed_frames_from_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[np.ndarray, list[np.ndarray], list[float] | None]:
        images = self._images_from_batch(batch)
        timestamps = self._frame_timestamps_from_batch(batch)
        selected, selected_timestamps = self._select_image_context_frames_and_timestamps(images, timestamps)
        processed_tensors = [self._process_image(image) for image in selected]
        seed_frames = [self._processed_image_to_uint8_frame(frame) for frame in processed_tensors]
        return np.concatenate(processed_tensors, axis=1), seed_frames, selected_timestamps

    @staticmethod
    def _processed_image_to_uint8_frame(processed: np.ndarray) -> np.ndarray:
        frame = processed[:, 0]
        frame = np.moveaxis(frame, 0, -1)
        return np.clip((frame + 1.0) * 127.5, 0, 255).astype(np.uint8)

    def image_debug_frames_from_batch(self, batch: dict[str, torch.Tensor]) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
        images = self._images_from_batch(batch)
        selected = self._select_image_context_frames(images)
        processed = [self._process_image_for_visualization(image) for image in selected]
        return images, processed, self._can_use_preprocessed_image_window(images)

    def _select_image_context_frames(self, images: list[np.ndarray]) -> list[np.ndarray]:
        selected, _ = self._select_image_context_frames_and_timestamps(images, None)
        return selected

    def _select_image_context_frames_and_timestamps(
        self, images: list[np.ndarray], timestamps: list[float] | None
    ) -> tuple[list[np.ndarray], list[float] | None]:
        if timestamps is not None and len(timestamps) != len(images):
            timestamps = None

        if self._can_use_preprocessed_image_window(images):
            if not self._preprocessed_input_notice_printed:
                print(
                    "mimic-video: received frame window already matches target preprocessing "
                    f"({self.config.image_width}x{self.config.image_height} at {self.config.image_frequency}fps); "
                    "skipping crop/resize and temporal subsampling.",
                    file=sys.stdout,
                    flush=True,
                )
                self._preprocessed_input_notice_printed = True
            return images, timestamps

        step = self.config.input_frequency // self.config.image_frequency
        raw_horizon = 1 + (self.config.image_horizon - 1) * step
        if len(images) < raw_horizon:
            if timestamps is not None:
                timestamps = [timestamps[0] for _ in range(raw_horizon - len(images))] + timestamps
            images = [images[0].copy() for _ in range(raw_horizon - len(images))] + images
        selected = images[-raw_horizon::step]
        selected_timestamps = timestamps[-raw_horizon::step] if timestamps is not None else None
        if len(selected) != self.config.image_horizon:
            raise RuntimeError(
                f"Expected {self.config.image_horizon} selected image frames, got {len(selected)} "
                f"from raw_horizon={raw_horizon}, step={step}."
            )
        return selected, selected_timestamps

    def _frame_timestamps_from_batch(self, batch: dict[str, torch.Tensor]) -> list[float] | None:
        key = f"{self.config.camera_key}{FRAME_WINDOW_TIMESTAMPS_SUFFIX}"
        if key not in batch:
            return None
        timestamps = batch[key]
        if isinstance(timestamps, torch.Tensor):
            timestamps = timestamps.detach().cpu().numpy()
        arr = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        return arr.tolist()

    @staticmethod
    def _client_observation_elapsed_from_batch(batch: dict[str, torch.Tensor]) -> float | None:
        if CLIENT_OBSERVATION_ELAPSED_KEY not in batch:
            return None
        value = batch[CLIENT_OBSERVATION_ELAPSED_KEY]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        return float(value)

    def _can_use_preprocessed_image_window(self, images: list[np.ndarray]) -> bool:
        return len(images) == self.config.image_horizon and all(self._frame_matches_target_resolution(image) for image in images)

    def _frame_matches_target_resolution(self, image: np.ndarray) -> bool:
        return image.shape[:2] == (self.config.image_height, self.config.image_width)

    def _images_from_batch(self, batch: dict[str, torch.Tensor]) -> list[np.ndarray]:
        if self.config.camera_key not in batch:
            raise KeyError(f"Missing camera key {self.config.camera_key}; available keys: {sorted(batch)}")
        image = batch[self.config.camera_key]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            arr = image.numpy()
        else:
            arr = np.asarray(image)

        frames = self._split_image_frames(arr)
        return [self._normalize_image_array(frame) for frame in frames]

    @staticmethod
    def _split_image_frames(arr: np.ndarray) -> list[np.ndarray]:
        arr = np.asarray(arr)
        if arr.ndim == 5:
            arr = arr[0]
        if arr.ndim == 4:
            if arr.shape[0] == 1 and arr.shape[1] in {1, 3}:
                return [arr[0]]
            return [arr[i] for i in range(arr.shape[0])]
        if arr.ndim == 3:
            return [arr]
        raise ValueError(f"Expected image tensor with 3, 4, or 5 dims, got {arr.shape}.")

    @staticmethod
    def _normalize_image_array(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 3:
            raise ValueError(f"Expected one image frame with 3 dims, got {arr.shape}.")
        if arr.shape[0] in {1, 3}:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected RGB image frame, got shape {arr.shape}.")
        if arr.dtype != np.uint8:
            max_value = float(np.nanmax(arr)) if arr.size else 1.0
            if max_value <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _state_context_from_batch(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        if self.config.state_key not in batch:
            raise KeyError(f"Missing state key {self.config.state_key}; available keys: {sorted(batch)}")
        state = batch[self.config.state_key]
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu()
            arr = state.numpy()
        else:
            arr = np.asarray(state)
        arr = np.asarray(arr, dtype=np.float32)
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr[None]
        if arr.ndim != 2 or arr.shape[1] != self.config.action_dim:
            raise ValueError(f"Expected states with shape (T, {self.config.action_dim}), got {arr.shape}.")
        if len(arr) < self.config.lowdim_horizon:
            arr = np.concatenate([np.repeat(arr[:1], self.config.lowdim_horizon - len(arr), axis=0), arr], axis=0)
        return arr[-self.config.lowdim_horizon :]

    @staticmethod
    def _load_prompt_embedding(config: MimicVideoPolicyConfig) -> torch.Tensor:
        prompt_embedding_path = os.environ.get("MIMIC_VIDEO_PROMPT_EMBEDDING_PATH")
        if prompt_embedding_path:
            with open(prompt_embedding_path, "rb") as f:
                embedding = pickle.load(f)  # nosec
            embedding = torch.from_numpy(embedding)
            return embedding.to(device="cuda", dtype=torch.bfloat16)

        if os.environ.get("MIMIC_VIDEO_ZERO_PROMPT_EMBEDDING") == "1":
            return torch.zeros((1, 512, 1024), device="cuda", dtype=torch.bfloat16)

        text_pipe_cfg = make_config()
        text_pipe_cfg = override(text_pipe_cfg, ["--", f"experiment={config.experiment_name}"])
        text_pipe_cfg.model.config.video_pipe_config.guardrail_config.enabled = False
        text_pipe_cfg.model.config.video_pipe_config.state_t = config.video_pipe_state_t
        text_pipe = Video2WorldPipeline.from_config(
            config=text_pipe_cfg.model.config.video_pipe_config,
            dit_path=str(config.video_model_path),
            use_text_encoder=True,
            offload_text_encoder=False,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=False,
        )
        with torch.no_grad():
            embedding = text_pipe.encode_prompt(config.prompt).to(device="cuda", dtype=torch.bfloat16)
        del text_pipe
        torch.cuda.empty_cache()
        return embedding

    @staticmethod
    def _load_pipeline(config: MimicVideoPolicyConfig) -> Video2World2ActionPipeline:
        cfg = make_config()
        cfg = override(cfg, ["--", f"experiment={config.experiment_name}"])
        cfg.model.config.video_pipe_config.guardrail_config.enabled = False
        cfg.model.config.video_pipe_config.state_t = config.video_pipe_state_t
        cfg.model.config.pipe_config.xattn_layer_idx = config.xattn_layer_idx
        cfg.model.config.pipe_config.scheduler.num_denoising_steps = config.action_denoising_steps
        network_config = getattr(config, "_mimic_video_network_config", {})
        for key, value in network_config.items():
            if hasattr(cfg.model.config.pipe_config.net, key):
                setattr(cfg.model.config.pipe_config.net, key, value)
        cfg.model.config.pipe_config.net.max_horizon = config.max_horizon
        cfg.model.config.pipe_config.net.in_channels = config.action_dim
        cfg.model.config.pipe_config.net.out_channels = config.action_dim

        video2world_pipe = Video2WorldPipeline.from_config(
            config=cfg.model.config.video_pipe_config,
            dit_path=str(config.video_model_path),
            use_text_encoder=False,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=False,
        )
        world2action_pipe = World2ActionPipeline.from_config(
            config=cfg.model.config.pipe_config,
            dit_path=str(config.action_model_path),
            device="cuda",
            dtype=torch.bfloat16,
        )
        stats = load_stats(config.stats_path)
        world2action_pipe.normalizer.build_from_stats(
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
            dtype=torch.bfloat16,
        )
        return Video2World2ActionPipeline(video2world_pipe, world2action_pipe).cuda().eval()
