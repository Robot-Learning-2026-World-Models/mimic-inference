from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
from einops import rearrange
from lerobot.policies import PreTrainedPolicy
from PIL import Image

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.types import NormalizationType
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.utils.config_helper import override

from .configuration_mimic_video import MimicVideoPolicyConfig
from .training_config import build_deployment_config, load_stats



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
        self.image_history: deque[np.ndarray] = deque(maxlen=config.image_horizon)
        self.lowdim_history: deque[np.ndarray] = deque(maxlen=config.lowdim_horizon)
        self._action_queue: list[torch.Tensor] = []

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
        config.max_horizon = dc.max_horizon
        config.action_denoising_steps = dc.action_denoising_steps
        config.stats_path = dc.stats_path

    def reset(self) -> None:
        self.image_history.clear()
        self.lowdim_history.clear()
        self._action_queue.clear()

    def get_optim_params(self) -> dict:
        return {"params": []}

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        prompt = str(kwargs.get("task", None) or kwargs.get("prompt", None) or self.config.prompt)
        image = self._image_from_batch(batch)
        state = self._state_from_batch(batch)
        self._append_history(image, state)

        images = np.concatenate(list(self.image_history), axis=1)
        lowdims = np.stack(list(self.lowdim_history), axis=0)
        input_vid = torch.from_numpy(images[None]).cuda().to(dtype=torch.bfloat16)
        state_tensor = torch.from_numpy(lowdims[None]).cuda().to(dtype=torch.bfloat16)

        with torch.no_grad():
            actions = self.pipeline(
                input_vid=input_vid,
                state_B_HO_O=state_tensor,
                prompt=prompt,
                num_sampling_step=self.config.num_sampling_steps,
                stop_after_step=self.config.stop_video_denoising_step,
                seed=self.config.seed,
                use_cuda_graphs=self.config.use_cuda_graphs,
            )[0].float()

        actions = actions[: self.config.horizon].cpu()
        if actions.ndim != 2 or actions.shape[1] != self.config.action_dim:
            raise RuntimeError(f"Expected action chunk shape (H, {self.config.action_dim}), got {tuple(actions.shape)}.")
        return actions.unsqueeze(0)

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)[0]
            self._action_queue = [action for action in chunk[: self.config.n_action_steps]]
        return self._action_queue.pop(0)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("mimic_video is an inference-only policy.")

    def _append_history(self, image: np.ndarray, state: np.ndarray) -> None:
        self.image_history.append(self._process_image(image))
        while len(self.image_history) < self.config.image_horizon:
            self.image_history.append(self.image_history[-1].copy())

        self.lowdim_history.append(state.astype(np.float32))
        while len(self.lowdim_history) < self.config.lowdim_horizon:
            self.lowdim_history.append(self.lowdim_history[-1].copy())

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image.astype(np.uint8), "RGB")
        x, y = self.config.crop_x, self.config.crop_y
        w, h = self.config.crop_width, self.config.crop_height
        pil_image = pil_image.crop((x, y, x + w, y + h))
        pil_image = pil_image.resize((self.config.image_width, self.config.image_height), Image.Resampling.BILINEAR)
        image = np.asarray(pil_image)
        tensor = rearrange(image, "h w c -> c h w")[:, None, :, :]
        return 2.0 * (tensor.astype(np.float32) / 255.0 - 0.5)

    def _image_from_batch(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        if self.config.camera_key not in batch:
            raise KeyError(f"Missing camera key {self.config.camera_key}; available keys: {sorted(batch)}")
        image = batch[self.config.camera_key]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            while image.ndim > 3:
                image = image[0]
            arr = image.numpy()
        else:
            arr = np.asarray(image)

        if arr.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dims after batch squeeze, got {arr.shape}.")
        if arr.shape[0] in {1, 3}:
            arr = np.moveaxis(arr, 0, -1)
        if arr.dtype != np.uint8:
            max_value = float(np.nanmax(arr)) if arr.size else 1.0
            if max_value <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _state_from_batch(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        if self.config.state_key not in batch:
            raise KeyError(f"Missing state key {self.config.state_key}; available keys: {sorted(batch)}")
        state = batch[self.config.state_key]
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu()
            while state.ndim > 1:
                state = state[0]
            arr = state.numpy()
        else:
            arr = np.asarray(state)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.shape != (self.config.action_dim,):
            raise ValueError(f"Expected {self.config.action_dim}D state, got shape {arr.shape}.")
        return arr

    @staticmethod
    def _load_pipeline(config: MimicVideoPolicyConfig) -> Video2World2ActionPipeline:
        cfg = make_config()
        cfg = override(cfg, ["--", f"experiment={config.experiment_name}"])
        cfg.model.config.video_pipe_config.guardrail_config.enabled = False
        cfg.model.config.pipe_config.xattn_layer_idx = config.xattn_layer_idx
        cfg.model.config.pipe_config.scheduler.num_denoising_steps = config.action_denoising_steps
        cfg.model.config.pipe_config.net.max_horizon = config.max_horizon
        cfg.model.config.pipe_config.net.in_channels = config.action_dim
        cfg.model.config.pipe_config.net.out_channels = config.action_dim

        video2world_pipe = Video2WorldPipeline.from_config(
            config=cfg.model.config.video_pipe_config,
            dit_path=str(config.video_model_path),
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
