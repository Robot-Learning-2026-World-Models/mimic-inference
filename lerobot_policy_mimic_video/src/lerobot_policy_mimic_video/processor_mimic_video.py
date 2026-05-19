from __future__ import annotations

from typing import Any

import torch


def make_mimic_video_pre_post_processors(
    config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
):
    """Return identity processors.

    The policy reads LeRobot's raw observation keys directly and returns
    already unnormalized SO101 actions.
    """

    del config, dataset_stats
    try:
        from lerobot.processor import IdentityProcessor, PolicyProcessorPipeline

        return PolicyProcessorPipeline([IdentityProcessor()]), PolicyProcessorPipeline([IdentityProcessor()])
    except Exception:
        return None, None
