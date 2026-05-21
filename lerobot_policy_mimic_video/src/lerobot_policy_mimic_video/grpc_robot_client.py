import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

import draccus
import numpy as np

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import RawObservation, TimedObservation, visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.import_utils import register_third_party_plugins


FRAME_WINDOW_SUFFIX = "_frame_window"
FRAME_WINDOW_TIMESTAMPS_SUFFIX = "_frame_window_timestamps"
CLIENT_OBSERVATION_ELAPSED_KEY = "client_observation_elapsed_s"


@dataclass
class MimicVideoRobotClientConfig(RobotClientConfig):
    raw_image_horizon: int = 13
    frame_window_suffix: str = FRAME_WINDOW_SUFFIX


class MimicVideoRobotClient(RobotClient):
    """LeRobot async robot client that includes a camera frame window per obs."""

    def __init__(self, config: MimicVideoRobotClientConfig):
        super().__init__(config)
        if config.raw_image_horizon < 1:
            raise ValueError("raw_image_horizon must be >= 1.")
        self.client_start_timestamp = time.time()
        self.frame_windows: dict[str, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=config.raw_image_horizon)
        )
        self.frame_window_timestamps: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=config.raw_image_horizon)
        )

    @staticmethod
    def _looks_like_image(value: Any) -> bool:
        arr = np.asarray(value)
        return arr.ndim == 3 and arr.shape[-1] in {1, 3}

    def _attach_frame_windows(self, raw_observation: RawObservation, *, frame_timestamp: float) -> None:
        for key, value in list(raw_observation.items()):
            if not self._looks_like_image(value):
                continue
            frame = np.asarray(value)
            window = self.frame_windows[key]
            timestamps = self.frame_window_timestamps[key]
            window.append(frame)
            timestamps.append(frame_timestamp)
            while len(window) < self.config.raw_image_horizon:
                window.appendleft(frame.copy())
                timestamps.appendleft(frame_timestamp)
            raw_observation[f"{key}{self.config.frame_window_suffix}"] = np.stack(list(window), axis=0)
            raw_observation[f"{key}{FRAME_WINDOW_TIMESTAMPS_SUFFIX}"] = np.asarray(list(timestamps), dtype=np.float64)

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            start_time = time.perf_counter()
            raw_observation: RawObservation = self.robot.get_observation()
            frame_timestamp = time.time()
            frame_elapsed_s = frame_timestamp - self.client_start_timestamp
            self._attach_frame_windows(raw_observation, frame_timestamp=frame_elapsed_s)
            raw_observation[CLIENT_OBSERVATION_ELAPSED_KEY] = frame_elapsed_s
            raw_observation["task"] = task
            with self.latest_action_lock:
                latest_action = self.latest_action
            observation = TimedObservation(
                timestamp=frame_timestamp,
                observation=raw_observation,
                timestep=max(latest_action + 1, 0),
                must_go=True,
            )
            obs_capture_time = time.perf_counter() - start_time

            with self.action_queue_lock:
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


@draccus.wrap()
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


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
