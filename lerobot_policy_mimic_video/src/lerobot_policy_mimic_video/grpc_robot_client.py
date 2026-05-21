import json
import logging
import pickle  # nosec
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import numpy as np

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import (
    RawObservation,
    TimedAction,
    TimedObservation,
    visualize_action_queue_size,
)
from lerobot.async_inference.robot_client import RobotClient
from lerobot.transport import services_pb2
from lerobot.utils.import_utils import register_third_party_plugins


FRAME_WINDOW_SUFFIX = "_frame_window"


@dataclass
class MimicVideoRobotClientConfig(RobotClientConfig):
    client_mode: str = "async"
    raw_image_horizon: int = 13
    frame_window_suffix: str = FRAME_WINDOW_SUFFIX
    sync_num_cycles: int = 0
    sync_log_actions: bool = True
    sync_action_log_dir: Path = Path("outputs/sync_client_actions")


class MimicVideoRobotClient(RobotClient):
    """LeRobot robot client that includes a camera frame window per obs."""

    def __init__(self, config: MimicVideoRobotClientConfig):
        super().__init__(config)
        if config.raw_image_horizon < 1:
            raise ValueError("raw_image_horizon must be >= 1.")
        self.frame_windows: dict[str, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=config.raw_image_horizon)
        )

    @staticmethod
    def _looks_like_image(value: Any) -> bool:
        arr = np.asarray(value)
        return arr.ndim == 3 and arr.shape[-1] in {1, 3}

    def _attach_frame_windows(self, raw_observation: RawObservation, *, pad: bool = True) -> None:
        for key, value in list(raw_observation.items()):
            if not self._looks_like_image(value):
                continue
            frame = np.asarray(value)
            window = self.frame_windows[key]
            window.append(frame)
            while pad and len(window) < self.config.raw_image_horizon:
                window.appendleft(frame.copy())
            if len(window) == self.config.raw_image_horizon:
                raw_observation[f"{key}{self.config.frame_window_suffix}"] = np.stack(
                    list(window), axis=0
                )

    def _clear_frame_windows(self) -> None:
        self.frame_windows.clear()

    def _next_observation_timestep(self) -> int:
        with self.latest_action_lock:
            latest_action = self.latest_action
        return max(latest_action + 1, 0)

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            start_time = time.perf_counter()
            raw_observation: RawObservation = self.robot.get_observation()
            self._attach_frame_windows(raw_observation)
            raw_observation["task"] = task
            with self.latest_action_lock:
                latest_action = self.latest_action
            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )
            obs_capture_time = time.perf_counter() - start_time

            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()
            _ = self.send_observation(observation)
            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                self.must_go.clear()
            if verbose:
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )
                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | "
                    f"Capturing observation took {obs_capture_time:.6f}s"
                )
            return raw_observation
        except Exception as exc:
            self.logger.error(f"Error in observation sender: {exc}")
            raise

    def warmup_frame_window(self, task: str, verbose: bool = False) -> tuple[RawObservation, float]:
        """Collect one fresh 30Hz raw window without sending intermediate observations."""
        self._clear_frame_windows()
        raw_observation: RawObservation | None = None
        start_time = time.perf_counter()
        for frame_idx in range(self.config.raw_image_horizon):
            frame_start = time.perf_counter()
            raw_observation = self.robot.get_observation()
            self._attach_frame_windows(raw_observation, pad=False)
            elapsed = time.perf_counter() - frame_start
            if frame_idx < self.config.raw_image_horizon - 1:
                time.sleep(max(0.0, self.config.environment_dt - elapsed))

        if raw_observation is None:
            raise RuntimeError("Failed to collect a warm-up observation.")
        self._attach_missing_complete_windows(raw_observation)
        raw_observation["task"] = task
        warmup_time = time.perf_counter() - start_time
        if verbose:
            frame_keys = sorted(k for k in raw_observation if k.endswith(self.config.frame_window_suffix))
            self.logger.info(
                f"Warm-up collected {self.config.raw_image_horizon} raw frames in {warmup_time:.3f}s | "
                f"frame_windows={frame_keys}"
            )
        return raw_observation, warmup_time

    def _attach_missing_complete_windows(self, raw_observation: RawObservation) -> None:
        if not self.frame_windows:
            raise RuntimeError("Warm-up did not find any image keys in robot observations.")
        for key, window in self.frame_windows.items():
            if len(window) != self.config.raw_image_horizon:
                raise RuntimeError(
                    f"Expected {self.config.raw_image_horizon} warm-up frames for {key!r}, got {len(window)}."
                )
            raw_observation[f"{key}{self.config.frame_window_suffix}"] = np.stack(
                list(window), axis=0
            )

    def send_sync_observation(self, raw_observation: RawObservation) -> TimedObservation:
        observation = TimedObservation(
            timestamp=time.time(),
            observation=raw_observation,
            timestep=self._next_observation_timestep(),
        )
        observation.must_go = True
        if not self.send_observation(observation):
            raise RuntimeError(f"Failed to send sync observation #{observation.get_timestep()}.")
        return observation

    def request_action_chunk_sync(self, verbose: bool = False) -> tuple[list[TimedAction], float]:
        start_time = time.perf_counter()
        while self.running:
            actions_chunk = self.stub.GetActions(services_pb2.Empty())
            if len(actions_chunk.data) == 0:
                self.logger.warning(
                    "Policy server returned an empty action chunk; retrying without sending a new observation."
                )
                time.sleep(self.config.environment_dt)
                continue

            timed_actions: list[TimedAction] = pickle.loads(actions_chunk.data)  # nosec
            client_device = self.config.client_device
            if client_device != "cpu":
                for timed_action in timed_actions:
                    if timed_action.get_action().device.type != client_device:
                        timed_action.action = timed_action.get_action().to(client_device)
            wait_time = time.perf_counter() - start_time
            if verbose and timed_actions:
                self.logger.info(
                    f"Received sync action chunk {timed_actions[0].get_timestep()}:{timed_actions[-1].get_timestep()} "
                    f"({len(timed_actions)} actions) after {wait_time:.3f}s"
                )
            return timed_actions, wait_time
        raise RuntimeError("Client stopped before receiving sync actions.")

    def execute_action_chunk_sync(
        self, timed_actions: list[TimedAction], verbose: bool = False
    ) -> tuple[list[Any], float]:
        performed_actions: list[Any] = []
        start_time = time.perf_counter()
        for timed_action in timed_actions:
            action_start = time.perf_counter()
            performed_action = self.robot.send_action(
                self._action_tensor_to_action_dict(timed_action.get_action())
            )
            performed_actions.append(performed_action)
            with self.latest_action_lock:
                self.latest_action = timed_action.get_timestep()
            elapsed = time.perf_counter() - action_start
            time.sleep(max(0.0, self.config.environment_dt - elapsed))
            if verbose:
                self.logger.debug(f"Sync action #{timed_action.get_timestep()} performed in {elapsed * 1000:.2f}ms")
        execute_time = time.perf_counter() - start_time
        return performed_actions, execute_time

    def sync_control_loop(self, task: str, verbose: bool = False) -> None:
        cycle_idx = 0
        self.logger.info(
            f"Starting sync debug loop | raw_image_horizon={self.config.raw_image_horizon} | "
            f"actions_per_chunk={self.config.actions_per_chunk}"
        )
        while self.running and (self.config.sync_num_cycles <= 0 or cycle_idx < self.config.sync_num_cycles):
            cycle_idx += 1
            raw_observation, warmup_time = self.warmup_frame_window(task, verbose=verbose)
            timed_observation = self.send_sync_observation(raw_observation)
            timed_actions, wait_time = self.request_action_chunk_sync(verbose=verbose)
            if len(timed_actions) != self.config.actions_per_chunk:
                self.logger.warning(
                    f"Expected {self.config.actions_per_chunk} sync actions, received {len(timed_actions)}."
                )
            performed_actions, execute_time = self.execute_action_chunk_sync(timed_actions, verbose=verbose)
            if self.config.sync_log_actions:
                self._log_sync_actions(
                    cycle_idx=cycle_idx,
                    timed_observation=timed_observation,
                    raw_observation=raw_observation,
                    timed_actions=timed_actions,
                    performed_actions=performed_actions,
                    warmup_time=warmup_time,
                    wait_time=wait_time,
                    execute_time=execute_time,
                )
            self.logger.info(
                f"Sync cycle {cycle_idx} complete | warmup={warmup_time:.3f}s | "
                f"wait={wait_time:.3f}s | execute={execute_time:.3f}s | actions={len(timed_actions)}"
            )

    def _log_sync_actions(
        self,
        *,
        cycle_idx: int,
        timed_observation: TimedObservation,
        raw_observation: RawObservation,
        timed_actions: list[TimedAction],
        performed_actions: list[Any],
        warmup_time: float,
        wait_time: float,
        execute_time: float,
    ) -> None:
        log_root = Path(self.config.sync_action_log_dir)
        log_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = log_root / f"{timestamp}_cycle{cycle_idx:04d}_obs{timed_observation.get_timestep()}"
        if run_dir.exists():
            run_dir = log_root / f"{run_dir.name}_{time.time_ns()}"
        run_dir.mkdir(parents=True, exist_ok=False)

        action_tensors = [action.get_action().detach().float().cpu().numpy() for action in timed_actions]
        actions_np = np.stack(action_tensors, axis=0) if action_tensors else np.empty((0,), dtype=np.float32)
        np.save(run_dir / "actions_returned.npy", actions_np)
        np.save(run_dir / "actions_applied.npy", actions_np)
        with open(run_dir / "actions_applied.json", "w", encoding="utf-8") as f:
            json.dump([self._to_jsonable(action) for action in performed_actions], f, indent=2)

        frame_window_shapes = {
            key: list(np.asarray(value).shape)
            for key, value in raw_observation.items()
            if key.endswith(self.config.frame_window_suffix)
        }
        metadata = {
            "cycle_idx": cycle_idx,
            "observation_timestep": timed_observation.get_timestep(),
            "observation_timestamp": timed_observation.get_timestamp(),
            "must_go": timed_observation.must_go,
            "raw_image_horizon": self.config.raw_image_horizon,
            "actions_per_chunk": self.config.actions_per_chunk,
            "fps": self.config.fps,
            "environment_dt": self.config.environment_dt,
            "returned_action_count": len(timed_actions),
            "applied_action_count": len(performed_actions),
            "first_action_timestep": timed_actions[0].get_timestep() if timed_actions else None,
            "last_action_timestep": timed_actions[-1].get_timestep() if timed_actions else None,
            "frame_window_shapes": frame_window_shapes,
            "warmup_time_s": warmup_time,
            "server_wait_time_s": wait_time,
            "execute_time_s": execute_time,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: MimicVideoRobotClient._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [MimicVideoRobotClient._to_jsonable(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().tolist()
        return value


def async_client(cfg: MimicVideoRobotClientConfig):
    logging.info(pformat(asdict(cfg)))
    client = MimicVideoRobotClient(cfg)
    if client.start():
        client.logger.info("Starting action receiver thread...")
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()
        try:
            client.control_loop(task=cfg.task)
        except Exception as e:
            print(e)
        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


def sync_client(cfg: MimicVideoRobotClientConfig):
    logging.info(pformat(asdict(cfg)))
    client = MimicVideoRobotClient(cfg)
    if client.start():
        try:
            client.sync_control_loop(task=cfg.task)
        except Exception as e:
            print(e)
            raise
        finally:
            client.stop()
            client.logger.info("Sync client stopped")


@draccus.wrap()
def client(cfg: MimicVideoRobotClientConfig):
    match cfg.client_mode:
        case "async":
            async_client(cfg)
        case "sync":
            sync_client(cfg)
        case _:
            raise ValueError(f"Unknown client_mode={cfg.client_mode!r}; expected 'async' or 'sync'.")


if __name__ == "__main__":
    register_third_party_plugins()
    client()
