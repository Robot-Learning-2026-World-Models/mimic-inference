"""LeRobot plugin for running mimic-video inside LeRobot async PolicyServer."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError("lerobot is required to use lerobot_policy_mimic_video.") from exc

from .configuration_mimic_video import MimicVideoPolicyConfig
from .modeling_mimic_video import MimicVideoPolicy
from .processor_mimic_video import make_mimic_video_pre_post_processors

__all__ = [
    "MimicVideoPolicy",
    "MimicVideoPolicyConfig",
    "make_mimic_video_pre_post_processors",
]
