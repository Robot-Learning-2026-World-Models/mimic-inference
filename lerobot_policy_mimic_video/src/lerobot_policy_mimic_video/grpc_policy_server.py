import logging
import pickle  # nosec
import platform
import threading
import time
from concurrent import futures
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import grpc
import numpy as np
import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2
from lerobot.transport import services_pb2_grpc

from .configuration_mimic_video import MimicVideoPolicyConfig
from .modeling_mimic_video import MimicVideoPolicy
from .training_config import build_deployment_config, maybe_download_file

try:
    from lerobot.async_inference.helpers import make_lerobot_observation as _raw_to_lerobot_observation
except ImportError:
    from lerobot.async_inference.helpers import raw_observation_to_observation as _raw_to_lerobot_observation


FRAME_WINDOW_SUFFIX = "_frame_window"
RAW_STATE_KEY = "observation.state"


@dataclass
class MimicVideoPolicyServerConfig(PolicyServerConfig):
    video_model_path: Path | None = None
    action_model_path: Path | None = None
    video_model_hf_repo_id: str | None = None
    video_model_hf_filename: str | None = None
    video_model_hf_revision: str | None = None
    video_model_hf_repo_type: str = "model"
    action_model_hf_repo_id: str | None = None
    action_model_hf_filename: str | None = None
    action_model_hf_revision: str | None = None
    action_model_hf_repo_type: str = "model"
    checkpoint_cache_dir: Path = Path("/checkpoints")
    training_config: Path | None = None
    stats_path: Path | None = None
    action_data_dir: Path | None = None
    hf_stats_repo_id: str | None = None
    hf_stats_filename: str | None = None
    hf_stats_revision: str | None = None
    policy_device: str = "cuda"
    prompt: str | None = None
    task: str | None = None
    num_sampling_steps: int = 35
    stop_video_denoising_step: int = 0
    use_cuda_graphs: bool = True
    seed: int = 0
    debug_save_denoised_video: bool = False
    debug_output_dir: Path = Path("runs")
    debug_run_name: str | None = None
    visualize_frames: bool = False
    visualize_every_n: int = 1


def _raw_camera_key(policy_camera_key: str) -> str:
    prefix = "observation.images."
    if policy_camera_key.startswith(prefix):
        return policy_camera_key[len(prefix) :]
    return policy_camera_key.rsplit(".", 1)[-1]


def _select_prompt(default_prompt: str, *, task: str | None, prompt: str | None) -> str:
    if prompt:
        return prompt
    task_prompts = {
        "task1": "the robot arm gently nudges the white object towards the target circle",
        "task2": "the robot arm gently nudges the white object towards the target circle while avoiding the red cylinder",
    }
    if task:
        key = task.strip().lower().replace("-", "").replace("_", "")
        if key in task_prompts:
            return task_prompts[key]
        return task
    return default_prompt


def _resolve_model_path(
    *,
    explicit_path: Path | None,
    config_path: Path | None,
    cache_dir: Path,
    hf_repo_id: str | None,
    hf_filename: str | None,
    hf_revision: str | None,
    hf_repo_type: str,
    label: str,
) -> Path:
    path = explicit_path or config_path
    if path is None and hf_filename:
        path = cache_dir / hf_filename
    if path is None:
        raise ValueError(
            f"{label} checkpoint is required. Set {label.upper()}_MODEL_PATH, "
            f"put it in the training YAML, or provide {label.upper()}_MODEL_HF_REPO_ID/FILENAME."
        )
    return maybe_download_file(
        path=path.expanduser(),
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
        hf_revision=hf_revision,
        repo_type=hf_repo_type,
    ).resolve()


class MimicVideoFrameVisualizer:
    def __init__(self, *, raw_fps: int, processed_fps: int):
        self.raw_fps = raw_fps
        self.processed_fps = processed_fps
        self._lock = threading.Lock()
        self._fig = None
        self._axes = None
        self._artists = []
        self._column_count = 0

    def update(
        self,
        *,
        raw_frames: list[np.ndarray],
        processed_frames: list[np.ndarray],
        timestep: int,
        preprocessed_input: bool,
    ) -> None:
        if not raw_frames and not processed_frames:
            return
        with self._lock:
            try:
                self._ensure_figure(max(len(raw_frames), len(processed_frames), 1))
                self._draw(raw_frames, processed_frames, timestep, preprocessed_input)
            except Exception as exc:  # pragma: no cover - visual aid only
                logging.warning("Disabling mimic-video frame visualizer after error: %s", exc)
                self.close()
                raise

    def _ensure_figure(self, column_count: int) -> None:
        if self._fig is not None and self._column_count == column_count:
            return
        if self._fig is not None:
            self.close()
        if platform.system() == "Darwin":
            import matplotlib

            matplotlib.use("Qt5Agg")
        import matplotlib.pyplot as plt

        plt.ion()
        self._fig, self._axes = plt.subplots(2, column_count, figsize=(max(8, column_count * 1.9), 5.0))
        self._axes = np.asarray(self._axes).reshape(2, column_count)
        if self._fig.canvas.manager is not None:
            self._fig.canvas.manager.set_window_title("mimic-video frame visualizer")
        self._fig.set_facecolor((0.12, 0.12, 0.12))
        self._artists = []
        for row in range(2):
            row_artists = []
            for ax in self._axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor((0.05, 0.05, 0.05))
                artist = ax.imshow(np.zeros((8, 8, 3), dtype=np.uint8))
                row_artists.append(artist)
            self._artists.append(row_artists)
        self._column_count = column_count

    def _draw(
        self,
        raw_frames: list[np.ndarray],
        processed_frames: list[np.ndarray],
        timestep: int,
        preprocessed_input: bool,
    ) -> None:
        assert self._fig is not None
        assert self._axes is not None
        raw_resolution = self._resolution_text(raw_frames)
        processed_resolution = self._resolution_text(processed_frames)
        status = "preprocessed input: crop/resize/subsample skipped" if preprocessed_input else "server preprocessing active"
        self._fig.suptitle(f"mimic-video frame window | timestep {timestep} | {status}", color="white")
        self._axes[0, 0].set_ylabel(
            f"Raw\n{raw_resolution}\n{self.raw_fps} fps\n{len(raw_frames)} frames",
            color="white",
            rotation=0,
            ha="right",
            va="center",
            labelpad=42,
        )
        self._axes[1, 0].set_ylabel(
            f"Processed\n{processed_resolution}\n{self.processed_fps} fps\n{len(processed_frames)} frames",
            color="white",
            rotation=0,
            ha="right",
            va="center",
            labelpad=42,
        )
        self._draw_row(0, raw_frames)
        self._draw_row(1, processed_frames)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        import matplotlib.pyplot as plt

        plt.pause(0.001)

    def _draw_row(self, row: int, frames: list[np.ndarray]) -> None:
        assert self._axes is not None
        for column in range(self._column_count):
            ax = self._axes[row, column]
            artist = self._artists[row][column]
            if column < len(frames):
                artist.set_data(frames[column])
                ax.set_visible(True)
            else:
                ax.set_visible(False)

    @staticmethod
    def _resolution_text(frames: list[np.ndarray]) -> str:
        if not frames:
            return "no frames"
        height, width = frames[0].shape[:2]
        return f"{width}x{height}"

    def close(self) -> None:
        if self._fig is None:
            return
        import matplotlib.pyplot as plt

        plt.close(self._fig)
        self._fig = None
        self._axes = None
        self._artists = []
        self._column_count = 0


class MimicVideoPolicyServer(PolicyServer):
    """LeRobot async gRPC server that preserves mimic-video's frame-window input.

    The stock LeRobot server converts each raw camera frame to one resized
    `C,H,W` tensor before calling the policy. mimic-video needs the raw temporal
    frame window so it can apply the YAML crop/resize and downsample 30Hz -> 10Hz
    itself.
    """

    config: MimicVideoPolicyServerConfig

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()
        policy_specs = pickle.loads(request.data)  # nosec
        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")
        if policy_specs.policy_type != "mimic_video":
            raise ValueError(f"MimicVideoPolicyServer only supports policy_type='mimic_video', got {policy_specs.policy_type!r}.")

        self.device = self.config.policy_device or policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        deployment = build_deployment_config(
            training_config=self.config.training_config,
            stats_path=self.config.stats_path,
            action_data_dir=self.config.action_data_dir,
            hf_stats_repo_id=self.config.hf_stats_repo_id,
            hf_stats_filename=self.config.hf_stats_filename,
            hf_stats_revision=self.config.hf_stats_revision,
        )
        video_model_path = _resolve_model_path(
            explicit_path=self.config.video_model_path,
            config_path=deployment.video_model_path,
            cache_dir=self.config.checkpoint_cache_dir,
            hf_repo_id=self.config.video_model_hf_repo_id,
            hf_filename=self.config.video_model_hf_filename,
            hf_revision=self.config.video_model_hf_revision,
            hf_repo_type=self.config.video_model_hf_repo_type,
            label="video",
        )
        action_model_path = _resolve_model_path(
            explicit_path=self.config.action_model_path,
            config_path=deployment.action_model_path,
            cache_dir=self.config.checkpoint_cache_dir,
            hf_repo_id=self.config.action_model_hf_repo_id,
            hf_filename=self.config.action_model_hf_filename,
            hf_revision=self.config.action_model_hf_revision,
            hf_repo_type=self.config.action_model_hf_repo_type,
            label="action",
        )
        policy_config = MimicVideoPolicyConfig(
            video_model_path=video_model_path,
            action_model_path=action_model_path,
            stats_path=deployment.stats_path,
            action_data_dir=None,
            hf_stats_repo_id=self.config.hf_stats_repo_id,
            hf_stats_filename=self.config.hf_stats_filename,
            hf_stats_revision=self.config.hf_stats_revision,
            training_config=self.config.training_config,
            n_action_steps=policy_specs.actions_per_chunk,
            input_frequency=self.config.fps,
            num_sampling_steps=self.config.num_sampling_steps,
            stop_video_denoising_step=self.config.stop_video_denoising_step,
            use_cuda_graphs=self.config.use_cuda_graphs,
            seed=self.config.seed,
            debug_save_denoised_video=self.config.debug_save_denoised_video,
            debug_output_dir=self.config.debug_output_dir,
            debug_run_name=self.config.debug_run_name,
        )
        policy_config.prompt = _select_prompt(
            policy_config.prompt,
            task=self.config.task,
            prompt=self.config.prompt,
        )
        start = time.perf_counter()
        self.policy = MimicVideoPolicy(policy_config).to(self.device).eval()
        self.frame_visualizer = (
            MimicVideoFrameVisualizer(raw_fps=self.config.fps, processed_fps=self.policy.config.image_frequency)
            if self.config.visualize_frames
            else None
        )
        self.preprocessor = None
        self.postprocessor = None
        self.logger.info(
            f"Loaded mimic_video on {self.device} in {time.perf_counter() - start:.4f}s | "
            f"actions_per_chunk={self.actions_per_chunk}"
        )
        return services_pb2.Empty()

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        start_prepare = time.perf_counter()
        raw_observation: dict[str, Any] = observation_t.get_observation()
        if RAW_STATE_KEY in raw_observation:
            observation = {RAW_STATE_KEY: torch.as_tensor(raw_observation[RAW_STATE_KEY])}
        else:
            observation = _raw_to_lerobot_observation(raw_observation, self.lerobot_features)

        camera_key = self.policy.config.camera_key
        window_key = f"{_raw_camera_key(camera_key)}{FRAME_WINDOW_SUFFIX}"
        if window_key in raw_observation:
            observation[camera_key] = torch.as_tensor(raw_observation[window_key])
        elif camera_key in observation:
            observation[camera_key] = torch.as_tensor(observation[camera_key])
        else:
            raise KeyError(
                f"Missing camera window {window_key!r} and policy camera key {camera_key!r}; "
                f"raw keys={sorted(raw_observation)}, observation keys={sorted(observation)}"
            )

        if "task" in raw_observation:
            observation["task"] = raw_observation["task"]
        self.last_processed_obs = observation_t
        prepare_time = time.perf_counter() - start_prepare

        visualizer = getattr(self, "frame_visualizer", None)
        if visualizer is not None and observation_t.get_timestep() % max(1, self.config.visualize_every_n) == 0:
            try:
                raw_frames, processed_frames, preprocessed_input = self.policy.image_debug_frames_from_batch(observation)
                visualizer.update(
                    raw_frames=raw_frames,
                    processed_frames=processed_frames,
                    timestep=observation_t.get_timestep(),
                    preprocessed_input=preprocessed_input,
                )
            except Exception:
                self.frame_visualizer = None

        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_inference
        if action_tensor.ndim != 3:
            raise RuntimeError(f"Expected action chunk shape (B, H, D), got {tuple(action_tensor.shape)}")

        action_tensor = action_tensor.squeeze(0).detach().cpu()
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(),
            list(action_tensor),
            observation_t.get_timestep(),
        )
        total_time = time.perf_counter() - start_prepare
        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare: {1000 * prepare_time:.2f}ms | "
            f"Inference: {1000 * inference_time:.2f}ms | "
            f"Total: {1000 * total_time:.2f}ms | "
            f"Action shape: {tuple(action_tensor.shape)}"
        )
        return action_chunk


@draccus.wrap()
def serve(cfg: MimicVideoPolicyServerConfig):
    logging.info(pformat(asdict(cfg)))
    policy_server = MimicVideoPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    policy_server.logger.info(f"MimicVideoPolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()
    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
