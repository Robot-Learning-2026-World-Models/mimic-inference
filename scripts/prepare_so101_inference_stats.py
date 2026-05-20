#!/usr/bin/env python3
"""Preprocess SO101 data and compute action normalizer stats exactly as training does."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MIMIC_ROOT = REPO_ROOT / "mimic-video"
MIMIC_MODEL = MIMIC_ROOT / "model"
MIMIC_SCRIPTS = MIMIC_ROOT / "scripts"

for path in (MIMIC_MODEL, MIMIC_SCRIPTS):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _optional_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _stats_file(action_data_dir: Path) -> Path:
    stats_dir = action_data_dir / ".statistics_cache"
    candidates = sorted(p for p in stats_dir.glob("*") if p.is_file())
    if len(candidates) != 1:
        names = ", ".join(p.name for p in candidates) or "<none>"
        raise RuntimeError(f"Expected exactly one statistics cache file under {stats_dir}, found: {names}")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import hydra
    from cosmos_predict2.configs.defaults.data_action import get_data_config
    from omegaconf import OmegaConf
    from so101_pipeline import load_yaml, preprocess_if_needed

    config = load_yaml(args.config)
    revision_dir = preprocess_if_needed(config)
    action_data_dir = revision_dir / "action_decoder"

    training = _optional_mapping(config.get("training"))
    action_cfg = _optional_mapping(training.get("action_decoder"))
    dataset_cfg = _optional_mapping(action_cfg.get("dataset"))
    experiment = str(action_cfg.get("experiment") or "so101")

    data_config = get_data_config(
        config_name=experiment,
        data_dir_override=str(action_data_dir),
        num_val_episodes=dataset_cfg.get("num_val_episodes"),
        val_episode_ranges=dataset_cfg.get("val_episode_ranges"),
        val_ratio=dataset_cfg.get("val_ratio"),
    )
    OmegaConf.resolve(data_config)
    dataset = hydra.utils.instantiate(data_config.dataset.dataset, train=True, verbose=True)
    stats = dataset.get_statistics()
    stats_path = _stats_file(action_data_dir)

    print(f"ACTION_DATA_DIR={action_data_dir}")
    print(f"STATS_PATH={stats_path}")
    print(f"stats_id={dataset.stats_id}")
    print(f"stats_keys={sorted(stats)}")


if __name__ == "__main__":
    main()
